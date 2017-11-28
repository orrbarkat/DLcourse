from __future__ import print_function

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


class IrisDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        label_names = self.df.iloc[:, 4].tolist()
        self.index_to_label = list(set(label_names))
        self.label_to_index = {n: i for i, n in enumerate(self.index_to_label)}

        self.features = self.df.iloc[:, :4].values.astype(np.float32)
        self.labels = [self.label_to_index[l] for l in label_names]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class ReQU(nn.Module):
    def __init__(self):
        super(ReQU, self).__init__()

    def forward(self, x):
        x = F.relu(x) ** 2
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            ReQU(),
            nn.Linear(64, 3),
            nn.LogSoftmax()
        )

    def forward(self, x):
        x = self.net(x)
        return x


def train(model, optimizer, data_loader):
    model.train()

    avg_loss = 0
    avg_acc = 0
    for data, target in data_loader:
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target, size_average=False)
        acc = target.eq(output.max(-1)[1]).sum()
        loss.backward()
        optimizer.step()

        avg_loss += loss.data[0]
        avg_acc += acc.data[0]

    avg_loss /= len(data_loader.dataset)
    avg_acc /= len(data_loader.dataset)
    return avg_loss, avg_acc


def main():
    data_loader = DataLoader(IrisDataset(args.data_path), batch_size=args.batch_size, shuffle=True)

    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    train_losses = []
    train_accs = []
    for epoch in range(1, args.epochs + 1):
        loss, acc = train(model, optimizer, data_loader)
        train_losses.append(loss)
        train_accs.append(acc)

        if epoch % 10 == 0:
            print('[Epoch {}/{}] Loss: {}  Accuracy: {}'.format(
                epoch, args.epochs, train_losses[-1], train_accs[-1]))

    plt.plot(range(len(train_losses)), train_losses)
    plt.xlabel('# Epochs')
    plt.ylabel('Cross Entropy')
    plt.title('Q2 Train Loss')

    plt.figure()
    plt.plot(range(len(train_accs)), train_accs)
    plt.xlabel('# Epochs')
    plt.ylabel('Cross Entropy')
    plt.title('Q2 Train Accuracy')
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    main()
