from __future__ import print_function

import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchfile
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import Dataset


class Net12Dataset(Dataset):
    def __init__(self, fg_data, bg_data):
        self.fg_data = fg_data
        self.bg_data = bg_data

    def __len__(self):
        return len(self.fg_data) + len(self.bg_data)

    def __getitem__(self, idx):
        if idx < len(self.fg_data):
            return self.fg_data[idx], 1
        else:
            return self.bg_data[idx - len(self.fg_data)], 0


class Net12(nn.Module):
    def __init__(self):
        super(Net12, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=1)
        )

    def forward(self, x):
        x = self.net(x)
        return x

    def loss(self, output, target):
        return F.cross_entropy(output, target)

    def stats(self, output, target):
        fn = (target.eq(1).float() * output.max(-1)[1].eq(0).float()).sum()
        p = target.eq(1).float().sum()
        fnr = fn / p.clamp(min=1)

        fp = (target.eq(0).float() * output.max(-1)[1].eq(1).float()).sum()
        n = target.eq(0).float().sum()
        fpr = fp / n.clamp(min=1)

        return dict(fnr=fnr, fpr=fpr)


def to_variables(*tensors, cuda):
    variables = []
    for t in tensors:
        if cuda:
            t = t.cuda()
        variables.append(Variable(t))

    return variables


def train(model, optimizer, data_loader, summary_writer, epoch):
    model.train()

    avg_stats = defaultdict(float)
    for batch_i, (x, y) in enumerate(data_loader):
        x, y = to_variables(x, y, cuda=args.cuda)
        optimizer.zero_grad()

        y_ = model(x).view(-1, 2)
        loss = model.loss(y_, y)
        stats = model.stats(y_, y)

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
        stats = model.stats(y_, y)

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
    faces_data_ = torchfile.load(args.faces_data_path)
    faces_data = np.empty([len(faces_data_), 3, 12, 12], dtype='float32')
    for i, im in enumerate(faces_data_.values()):
        faces_data[i] = im
    bg_data = np.load(args.background_data_path)['bg_12']

    n_train_faces = int(len(faces_data) * 0.9)
    n_train_bg = int(len(bg_data) * 0.9)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        Net12Dataset(faces_data[:n_train_faces], bg_data[:n_train_bg]),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        Net12Dataset(faces_data[n_train_faces:], bg_data[n_train_bg:]),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    train_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'train'))
    test_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'test'))

    model = Net12()

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
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--no-cuda', action='store_true')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    main()
