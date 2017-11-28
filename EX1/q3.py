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


def train(model, criterion, optimizer, data_loader):
    model.train()

    avg_loss = 0
    for data, target in data_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        avg_loss += loss.data[0]

    avg_loss /= len(data_loader.dataset)
    return avg_loss


def test(model, criterion, data_loader):
    model.eval()
    avg_loss = 0
    for data, target in data_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target, volatile=True)
        output = model(data)
        avg_loss += criterion(output, target).data[0]

    avg_loss /= len(data_loader.dataset)
    return avg_loss


def plot_pred(model, image, keypoints):
    model.eval()
    input = Variable(torch.FloatTensor(image), volatile=True)
    if args.cuda:
        input = input.cuda()
    output = model(input).data.cpu().numpy()
    output = output.reshape(-1, 2)
    output = (output + 1) * 48

    plt.figure()
    plt.imshow(image.reshape(96, 96), cmap='gray')
    plt.scatter(output[:, 0], output[:, 1], c='r', marker='x')

    keypoints = (keypoints.reshape(-1, 2) + 1) * 48
    plt.scatter(keypoints[:, 0], keypoints[:, 1], c='b', marker='x')
    plt.legend(['pred', 'gt'])


def train_test(model_name, train_loader, test_loader):
    if model_name == 'fc':
        model = FCNet()
    else:
        model = ConvNet()

    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss(size_average=False)

    train_losses = []
    test_losses = []
    print('Training {} model...'.format(model_name))
    for epoch in range(1, args.epochs + 1):
        train_losses.append(train(model, criterion, optimizer, train_loader))
        test_losses.append(test(model, criterion, test_loader))

        print('[{}] Epoch {}/{}  Train: {:.6f}  Test: {:.6f}'.format(
            model_name, epoch, args.epochs, train_losses[-1], test_losses[-1]))

    return train_losses, test_losses


def main():
    train_loader, test_loader = load_data()

    fc_train_losses, fc_test_losses = train_test('fc', train_loader, test_loader)
    conv_train_losses, conv_test_losses = train_test('conv', train_loader, test_loader)

    plt.figure()
    plt.plot(range(len(fc_train_losses)), fc_train_losses,
             range(len(fc_test_losses)), fc_test_losses,
             range(len(conv_train_losses)), conv_train_losses,
             range(len(conv_test_losses)), conv_test_losses)
    plt.legend(['FC train loss', 'FC test loss',
                'Conv train loss', 'Conv test loss'])
    plt.xlabel('# Epochs')
    plt.ylabel('MSE')
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--no-cuda', action='store_true')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    main()
