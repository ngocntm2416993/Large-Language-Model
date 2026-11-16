import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from train import CharRNN


@torch.no_grad()
def predict(net, char, h, device, top_k=None, temperature=0.8):
    x = torch.tensor([[net.char2int[char]]], device=device)
    out, h = net(x, h)
    p = F.softmax(out / temperature, dim=1).squeeze(0).cpu()
    if top_k is None:
        top_ch = np.arange(net.n_chars)
    else:
        p, top_ch = p.topk(top_k)
        top_ch = top_ch.numpy()
    p = p.numpy()
    return net.int2char[int(np.random.choice(top_ch, p=p / p.sum()))], h


def sample(net, device, max_len=1000, prime='Ngày', top_k=None, temperature=0.8,
           end_chars='.!?', min_len=20):
    net.eval()
    prime = ''.join(ch for ch in prime if ch in net.char2int) or net.chars[0]
    chars = list(prime)
    h = net.init_hidden(1, device)
    for ch in prime:
        char, h = predict(net, ch, h, device, top_k, temperature)
    chars.append(char)

    n_new = 1
    while n_new < max_len:
        char, h = predict(net, chars[-1], h, device, top_k, temperature)
        chars.append(char)
        n_new += 1
        if n_new < min_len:
            continue
        if char == '\n' and chars[-2] == '\n':
            break
        if chars[-2] in end_chars and char in ' \n':
            break
        if chars[-3] in end_chars and chars[-2] in '"\'' and char in ' \n':
            break
    return ''.join(chars).rstrip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  'char_rnn_best.pt'))
    p.add_argument('--prime', default='Ngày')
    p.add_argument('--max-len', type=int, default=1000)
    p.add_argument('--top-k', type=int, default=5)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--n', type=int, default=3, help='number of samples to generate')
    args = p.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print('Device:', device)
    ckpt = torch.load(args.ckpt, map_location=device)
    net = CharRNN(ckpt['tokens'], ckpt['n_hidden'], ckpt['n_layers'], ckpt['drop_prob']).to(device)
    net.load_state_dict(ckpt['state_dict'])
    print(f"Checkpoint: epoch {ckpt['epoch'] + 1}, step {ckpt['step']}, val {ckpt['best_val']:.4f}\n")

    for i in range(args.n):
        print(f'--- Mẫu {i + 1} ---')
        print(sample(net, device, args.max_len, args.prime, args.top_k, args.temperature), '\n')


if __name__ == '__main__':
    main()
