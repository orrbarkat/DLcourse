import os
import numpy as np
import torch
import torchfile
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from collections import defaultdict
from tqdm import tqdm


class WordsDataset(Dataset):
    def __init__(self, data, vocabulary, seq_size):
        # num_classes = max(vocabulary.values())+1
        # one_hot = np.zeros((data.shape[0], num_classes)) # torch.zeros((data.shape[0], num_classes)).int()
        # one_hot[np.arange(data.shape[0]), data] = 1
        # one_hot.scatter_(1, data, 1)
        self.data = data #torch.from_numpy(one_hot)
        self.seq_size = seq_size
        self.char2idx = vocabulary
        self.idx2char = {v: k for k, v in self.char2idx.items()}

    def __len__(self):
        return self.data.shape[0] - self.seq_size

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx:idx + self.seq_size]).long(), int(self.data[idx+self.seq_size])


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
        decoded = self.fc(output)  # .view(output.size(0) * output.size(1), output.size(2)))
        return decoded, hidden  # .view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self):
        return (Variable(torch.zeros(1, 1, self.hidden_size)),
                Variable(torch.zeros(1, 1, self.hidden_size)))



def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def train(model, optimizer, data_loader, summary_writer, epoch):
    model.train()

    avg_stats = defaultdict(float)
    state = model.init_hidden()
    for batch_i, (x, y) in tqdm(enumerate(data_loader), desc='Batch'):
        x, y = Variable(x), Variable(y)
        optimizer.zero_grad()
        state = model.init_hidden()
        y_, state = model(x, state)
        y_ = y_[:,-1,:]
        loss = F.cross_entropy(y_, y)
        stats = calc_stats(y_, y)

        loss.backward()
        optimizer.step()

        avg_stats['loss'] += loss.data[0]
        for k, v in stats.items():
            avg_stats[k] += v.data[0]

        if batch_i % 30 == 0:
            torch.save(model, os.path.join(args.log_dir, 'model_test.checkpoint'))

    str_out = '[train] {}/{} '.format(epoch, args.epochs)
    for k, v in avg_stats.items():
        avg = v / len(data_loader)
        summary_writer.add_scalar(k, avg, epoch)
        str_out += '{}: {:.6f}  '.format(k, avg)

    print(str_out)


def main():
    torch.manual_seed(args.seed)
    data = torchfile.load(os.path.join(args.data_path, 'train.t7'))
    vocab = torchfile.load(os.path.join(args.data_path, 'vocab.t7'))

    loader = torch.utils.data.DataLoader(WordsDataset(data, vocab, args.seq),
                                         batch_size=args.batch_size, shuffle=False)
    num_classes = max(vocab.values()) + 1
    model = LanguageModel(num_classes, args.emsize, 100, batch_first=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'train'))

    for epoch in range(1, args.epochs + 1):
        train(model, optimizer, loader, train_writer, epoch)
        torch.save(model, os.path.join(args.log_dir, 'model_t.checkpoint'))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--log-dir', required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--save', type=str, default='model.pt', help='path to save the final model')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--seq', type=int, default=100, help='sequence length')
    parser.add_argument('--emsize', type=int, default=30, help='size of word embeddings')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    main()
