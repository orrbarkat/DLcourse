from __future__ import print_function

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


class FacialKeypointDataset(Dataset):
    def __init__(self, images, keypoints):
        self.images = images
        self.keypoints = keypoints

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.keypoints[idx]


def load_data():
    df = pd.read_csv(args.data_path)

    # read keypoints
    keypoints = df.iloc[:, :-1].values
    all_keypoints_not_nan = np.all(~np.isnan(keypoints), axis=1)
    keypoints = keypoints[all_keypoints_not_nan].astype(np.float32)
    keypoints = (keypoints / 48) - 1

    # read images
    images = df.iloc[:, -1].tolist()
    images = np.stack([np.fromstring(im, dtype=np.float32, sep=' ') for im in images])
    images = images[all_keypoints_not_nan]
    images = images / 255

    n_train = int(len(images) * 0.9)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = DataLoader(FacialKeypointDataset(images[:n_train], keypoints[:n_train]),
                              batch_size=args.batch_size,
                              shuffle=True,
                              **kwargs)
    test_loader = DataLoader(FacialKeypointDataset(images[n_train:], keypoints[n_train:]),
                             batch_size=args.batch_size,
                             **kwargs)

    return train_loader, test_loader


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 11 * 11, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 30)
        )

    def forward(self, x):
        x = x.view(-1, 1, 96, 96)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(96 * 96, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 30)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


def train(model, optimizer, data_loader):
    model.train()

    avg_loss = 0
    for data, target in data_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target, size_average=False)
        loss.backward()
        optimizer.step()
        avg_loss += loss.data[0]

    avg_loss /= len(data_loader.dataset)
    return avg_loss


def test(model, data_loader):
    model.eval()
    avg_loss = 0
    for data, target in data_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        avg_loss += F.mse_loss(output, target, size_average=False).data[0]

    avg_loss /= len(data_loader.dataset)
    return avg_loss


def main():
    train_loader, test_loader = load_data()

    if args.model == 'fc':
        model = FCNet()
    else:
        model = ConvNet()

    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_losses = []
    test_losses = []
    for epoch in range(1, args.epochs + 1):
        train_losses.append(train(model, optimizer, train_loader))
        test_losses.append(test(model, test_loader))

        print('[Epoch {}/{}] Train: {}\tTest: {}'.format(
            epoch, args.epochs, train_losses[-1], test_losses[-1]))

    plt.plot(range(len(train_losses)), train_losses)
    plt.plot(range(len(test_losses)), test_losses)
    plt.legend(['Train loss', 'Test loss'])
    plt.xlabel('# Epochs')
    plt.ylabel('MSE')
    plt.title(args.model)
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--model', choices=['fc', 'conv'], help='Model to run', required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--no-cuda', action='store_true')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    main()
