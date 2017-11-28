from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class XorNet(nn.Module):
    def __init__(self):
        super(XorNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 3),
            nn.Tanh(),
            nn.Linear(3, 1)
        )

    def forward(self, x):
        x = self.net(x)
        return x


def main():
    data = torch.FloatTensor([[0, 0, 0],
                              [0, 1, 1],
                              [1, 0, 1],
                              [1, 1, 0]])
    x = Variable(data[:, :2])
    y = Variable(data[:, 2])

    model = XorNet()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    model.train()
    for i in range(1, args.epochs + 1):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print('[Epoch {}]: Loss = {}'.format(i, loss.data[0]))

    print('\nDone training. Network predictions:')
    model.eval()
    output = model(x)
    for (a, b), o in zip(x.data, output.data[:, 0]):
        print('\t{} XOR {} = {}'.format(a, b, o))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.07)
    args = parser.parse_args()

    main()
