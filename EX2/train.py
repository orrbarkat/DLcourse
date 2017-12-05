from __future__ import print_function

import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset

from utils import to_variables


class Net12Dataset(Dataset):
    def __init__(self, fg_data, bg_data, test=False):
        self.fg_data = fg_data
        self.bg_data = bg_data
        self.test = test

    def __len__(self):
        return len(self.fg_data) + len(self.bg_data)

    def __getitem__(self, idx):
        if idx < len(self.fg_data):
            image, label = self.fg_data[idx], 1
        else:
            image, label = self.bg_data[idx - len(self.fg_data)], 0

        if not self.test:
            # random flip-lr
            if random.random() < 0.5:
                image = image[:, :, ::-1]

            # random brightness
            if random.random() < 0.5:
                image += np.random.uniform(-0.3, 0.3)

            if random.random() < 0.5:
                mu = image.mean()
                image = (image - mu) * np.random.uniform(0.8, 1.25) + mu

            image = image.clip(0, 1)

        return image, label


class Net12(nn.Module):
    def __init__(self):
        super(Net12, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3), # 10x10
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # (10-3+2)/2 + 1 = 5.5
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=1)
        )

        self.loss_weight = torch.Tensor([1, 20])

    def forward(self, x):
        x = self.net(x)
        return x

    def loss(self, output, target):
        return F.cross_entropy(output, target, weight=self.loss_weight)

    def cuda(self, device=None):
        super(Net12, self).cuda(device)
        self.loss_weight = self.loss_weight.cuda()
        return self


class Net24(nn.Module):
    def __init__(self):
        super(Net24, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),  # 20x20
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 10, 10
            nn.ReLU(inplace=True)
        )

        self.fc1 = nn.Linear(64*10*10, 128)
        self.fc2 = nn.Linear(128, 2)

        self.loss_weight = torch.Tensor([1, 20])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 64*10*10)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.fc2(x)
        return x

    def loss(self, output, target):
        return F.cross_entropy(output, target, weight=self.loss_weight)

    def cuda(self, device=None):
        super(Net24, self).cuda(device)
        self.loss_weight = self.loss_weight.cuda()
        return self


def calc_stats(output, target):
    fn = (target.eq(1).float() * output.max(-1)[1].eq(0).float()).sum()
    p = target.eq(1).float().sum()
    fnr = fn / p.clamp(min=1)

    fp = (target.eq(0).float() * output.max(-1)[1].eq(1).float()).sum()
    n = target.eq(0).float().sum()
    fpr = fp / n.clamp(min=1)

    tp = (target.eq(1).float() * output.max(-1)[1].eq(1).float()).sum()
    recall = tp / p.clamp(min=1)

    return dict(fnr=fnr, fpr=fpr, recall=recall)


def train(model, optimizer, data_loader, summary_writer, epoch):
    model.train()

    avg_stats = defaultdict(float)
    for batch_i, (x, y) in enumerate(data_loader):
        x, y = to_variables(x, y, cuda=args.cuda)
        optimizer.zero_grad()

        y_ = model(x).view(-1, 2)
        loss = model.loss(y_, y)
        stats = calc_stats(y_, y)

        loss.backward()
        optimizer.step()

        avg_stats['loss'] += loss.data[0]
        for k, v in stats.items():
            avg_stats[k] += v.data[0]

    str_out = '[train] {}/{} '.format(epoch, args.epochs)
    for k, v in avg_stats.items():
        avg = v / len(data_loader)
        summary_writer.add_scalar(k, avg, epoch)
        str_out += '{}: {:.6f}  '.format(k, avg)

    print(str_out)


def test(model, data_loader, summary_writer, epoch):
    model.eval()

    avg_stats = defaultdict(float)
    for batch_i, (x, y) in enumerate(data_loader):
        x, y = to_variables(x, y, cuda=args.cuda)

        y_ = model(x).view(-1, 2)
        loss = model.loss(y_, y)
        stats = calc_stats(y_, y)

        avg_stats['loss'] += loss.data[0]
        for k, v in stats.items():
            avg_stats[k] += v.data[0]

    str_out = '[test ] {}/{} '.format(epoch, args.epochs)
    for k, v in avg_stats.items():
        avg = v / len(data_loader)
        summary_writer.add_scalar(k, avg, epoch)
        str_out += '{}: {:.6f}  '.format(k, avg)

    print(str_out)


def main():
    faces_data = np.load(args.faces_data_path)['data']
    bg_data = np.load(args.background_data_path)['data']

    n_train_faces = int(len(faces_data) * 0.9)
    n_train_bg = int(len(bg_data) * 0.9)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        Net12Dataset(faces_data[:n_train_faces], bg_data[:n_train_bg]),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        Net12Dataset(faces_data[n_train_faces:], bg_data[n_train_bg:], test=True),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    train_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'train'))
    test_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'test'))

    if args.model == '12net':
        model = Net12()
    else:
        model = Net24()

    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(model, optimizer, train_loader, train_writer, epoch)
        test(model, test_loader, test_writer, epoch)

    torch.save(model.state_dict(), os.path.join(args.log_dir, 'model.checkpoint'))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--faces-data-path', required=True)
    parser.add_argument('--background-data-path', required=True)
    parser.add_argument('--log-dir', required=True)
    parser.add_argument('--model', choices=['12net', '24net'], required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--no-cuda', action='store_true')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    main()
