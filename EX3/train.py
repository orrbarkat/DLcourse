import os
import torch
import torchfile
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
import torch.optim as optim
from tensorboardX import SummaryWriter
from collections import defaultdict


class WordsDataset(Dataset):
    def __init__(self, data, vocabulary, seq_size):
        self.data = data
        self.seq_size = seq_size
        self.char2idx = vocabulary
        self.idx2char = {v: k for k, v in self.char2idx.items()}

    def __len__(self):
        return self.data.shape[0] - self.seq_size

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx:idx + self.seq_size]).long(), \
               torch.from_numpy(self.data[idx+1:idx+1 + self.seq_size]).long()


class LanguageModel(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, rnn_layers=1, batch_first=False):
        super(LanguageModel, self).__init__()
        self.encoder = nn.Embedding(num_classes, input_size)
        self.rnn = nn.LSTM(input_size, hidden_size, rnn_layers, batch_first=batch_first)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.hidden_size = hidden_size

    def forward(self, x, state):
        embedded = self.encoder(x)
        output, hidden = self.rnn(embedded, state)
        decoded = self.fc(output)
        return decoded, hidden

    def init_hidden(self):
        return (Variable(torch.zeros(1, 1, self.hidden_size)),
                Variable(torch.zeros(1, 1, self.hidden_size)))


def train(model, criterion, optimizer, data_loader, summary_writer, step, total_steps):
    model.train()

    steps = 100
    for batch_i, (x, y) in enumerate(data_loader):
        avg_stats = defaultdict(float)
        x, y = Variable(x), Variable(y)
        optimizer.zero_grad()
        state = model.init_hidden()
        y_, state = model(x, state)
        loss = 0
        for i in range(100):
            loss += criterion(y_[:, i, :], y[:, i])

        loss.backward()
        optimizer.step()

        avg_stats['loss'] += loss.data[0]

        if batch_i == steps:
            break

    str_out = '[train] {}/{} '.format(step, total_steps)
    for k, v in avg_stats.items():
        avg = v / args.batch_size
        summary_writer.add_scalar(k, avg, step)
        str_out += '{}: {:.6f}  '.format(k, avg)
    print(str_out)


def test(model, criterion, data_loader, summary_writer, step, total_steps):
    model.eval()

    for batch_i, (x, y) in enumerate(data_loader):
        avg_stats = defaultdict(float)
        x, y = Variable(x), Variable(y)
        state = model.init_hidden()
        y_, state = model(x, state)
        loss = 0
        for i in range(100):
            loss += criterion(y_[:, i, :], y[:, i])

        avg_stats['loss'] += loss.data[0]
        str_out = '[test] {}/{} '.format(step, total_steps)
        for k, v in avg_stats.items():
            avg = v / args.batch_size
            summary_writer.add_scalar(k, avg, step)
            str_out += '{}: {:.6f}  '.format(k, avg)
        print(str_out)
        break


def main():
    torch.manual_seed(args.seed)
    data = torchfile.load(os.path.join(args.data_path, 'train.t7'))
    vocab = torchfile.load(os.path.join(args.data_path, 'vocab.t7'))

    split = int(0.9* len(data))
    train_loader = torch.utils.data.DataLoader(WordsDataset(data[:split], vocab, args.seq),
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(WordsDataset(data[split:], vocab, args.seq),
                                              batch_size=args.batch_size, shuffle=True)
    num_classes = max(vocab.values()) + 1

    model = LanguageModel(num_classes, args.emsize, 256, batch_first=True)
    if args.retrain:
        model = torch.load('{}.checkpoint'.format(args.save))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    train_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'train'))
    test_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'test'))

    steps = 100
    total_steps = len(train_loader)//steps
    for epoch in range(1, args.epochs + 1):
        for step in range(total_steps):
            train(model, criterion, optimizer, train_loader, train_writer, step, total_steps)
            test(model, criterion, test_loader, test_writer, step, total_steps)
            torch.save(model, '{}.checkpoint'.format(args.save))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--log-dir', required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--retrain', action='store_true')
    parser.add_argument('--save', type=str, required=True, help='path to save the final model')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--seq', type=int, default=100, help='sequence length')
    parser.add_argument('--emsize', type=int, default=30, help='size of word embeddings')
    args = parser.parse_args()

    main()
