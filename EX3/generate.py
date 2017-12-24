import os
from torch.autograd import Variable
import torchfile
import torch
from train import LanguageModel

def main():
    char2idx = torchfile.load(os.path.join(args.vocabulary, 'vocab.t7'))
    null_char = '<null>'.encode('utf8')
    char2idx[null_char] = 0
    idx2char = {v: k for k, v in char2idx.items()}
    model = torch.load(args.model)
    state = model.init_hidden()

    model.eval()
    data = torch.rand(1, 2).mul(max(idx2char.keys())+1).long()
    output = []
    for i in range(args.length):
        x = Variable(data, volatile=True)
        y_, state = model(x, state)
        # char_weights = y_.squeeze().data.div(args.temperature).exp().cpu()
        label = y_[:,-1,:].squeeze().max(0)[1].data[0]
        data[0,:-1] = data[0,1:]
        data[0,-1] = label
        last_char = idx2char[label].decode('utf8')
        # print(last_char)
        output.append(last_char)
    output = ''.join(output)
    with open(args.output, 'wb') as f:
        f.write(output.encode('utf8'))
    print(output)




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocabulary', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--length', type=int, default=500)
    args = parser.parse_args()

    main()
