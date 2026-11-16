import argparse
import os
import re
import time
import unicodedata

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# ----------------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------------
def join_pdf_lines(text):
    text = re.sub(r'[ \t]+\n', '\n', text)                      # delete blank lines
    text = re.sub(r'(?<!\n)\n(?!\n|\s*[-–—•])', ' ', text)     # \n -> space
    text = re.sub(r'[ \t]{2,}', ' ', text)                      # merge blank lines
    return text


def pdf_to_text(pdf_path, txt_path):
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    text = '\n'.join(p.extract_text() or '' for p in reader.pages)
    text = unicodedata.normalize('NFC', text)               # translate to unicode
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.M)     # delete page numbers
    text = re.sub(r'-\n(\w)', r'\1', text)                  # continue words at line breaks
    text = re.sub(r'\n{3,}', '\n\n', text)                  # merge extra blank lines
    text = join_pdf_lines(text) 
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f'Converted {pdf_path} -> {txt_path} ({len(text)} characters)')


def get_batches(arr, batch_size, seq_length):
    batch_size_total = batch_size * seq_length
    n_batches = (len(arr) - 1) // batch_size_total
    x_all = arr[:n_batches * batch_size_total].reshape(batch_size, -1)
    y_all = arr[1:n_batches * batch_size_total + 1].reshape(batch_size, -1)
    for n in range(0, x_all.shape[1], seq_length):
        yield x_all[:, n:n + seq_length], y_all[:, n:n + seq_length]


# ----------------------------------------------------------------------------
# 2. Model 
# ----------------------------------------------------------------------------
class CharRNN(nn.Module):
    def __init__(self, tokens, n_hidden=512, n_layers=2, drop_prob=0.5):
        super().__init__()
        self.chars = tuple(tokens)
        self.n_chars = len(self.chars)
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.drop_prob = drop_prob
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: i for i, ch in self.int2char.items()}

        self.lstm = nn.LSTM(self.n_chars, n_hidden, n_layers,
                            dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden, self.n_chars)

    def forward(self, x, hidden):
        x = F.one_hot(x, self.n_chars).float()
        r_output, hidden = self.lstm(x, hidden)
        out = self.dropout(r_output).reshape(-1, self.n_hidden)
        return self.fc(out), hidden

    def init_hidden(self, batch_size, device):
        z = torch.zeros(self.n_layers, batch_size, self.n_hidden, device=device)
        return (z, z.clone())


# ----------------------------------------------------------------------------
# 3. Checkpoint
# ----------------------------------------------------------------------------
def save_ckpt(path, net, opt, epoch, step, best_val):
    torch.save({
        'n_hidden': net.n_hidden,
        'n_layers': net.n_layers,
        'drop_prob': net.drop_prob,
        'tokens': net.chars,
        'state_dict': net.state_dict(),
        'optimizer': opt.state_dict(),
        'epoch': epoch,
        'step': step,
        'best_val': float(best_val),
    }, path + '.tmp')
    os.replace(path + '.tmp', path)   # Write to temp file first to avoid corruption if interrupted


@torch.no_grad()
def evaluate(net, data, batch_size, seq_length, device, criterion):
    net.eval()
    h = net.init_hidden(batch_size, device)
    losses = []
    for x, y in get_batches(data, batch_size, seq_length):
        x = torch.from_numpy(x).long().to(device)
        y = torch.from_numpy(y).long().to(device)
        out, h = net(x, h)
        losses.append(criterion(out, y.reshape(-1)).item())
    net.train()
    return float(np.mean(losses)) if losses else float('nan')


# ----------------------------------------------------------------------------
# 4. Train
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default=os.path.dirname(os.path.abspath(__file__)))
    p.add_argument('--pdf', nargs='+', default=None, help='một hoặc nhiều file PDF/TXT trong data-dir, sẽ được gộp thành sach.txt')
    p.add_argument('--txt', default='book.txt')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--seq-length', type=int, default=100)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--hidden', type=int, default=512)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--clip', type=float, default=5.0)
    p.add_argument('--val-frac', type=float, default=0.1)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print('Device:', device)

    txt_path = os.path.join(args.data_dir, args.txt)
    last_path = os.path.join(args.data_dir, 'char_rnn_last.pt')
    best_path = os.path.join(args.data_dir, 'char_rnn_best.pt')

        if args.pdf:
        parts = []
        for name in args.pdf:
            src = os.path.join(args.data_dir, name)
            if name.lower().endswith('.pdf'):
                tmp = src + '.txt'
                pdf_to_text(src, tmp)
            else:
                tmp = src                      # file .txt 
            with open(tmp, encoding='utf-8') as f:
                parts.append(unicodedata.normalize('NFC', f.read()))
            print(f'  + {name}: {len(parts[-1])} character ')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(parts))
        print(f'Merge {len(parts)} docs -> {txt_path}')
    with open(txt_path, encoding='utf-8') as f:
        text = unicodedata.normalize('NFC', f.read())

    ckpt = None
    if args.resume and os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=device)
        chars = ckpt['tokens']           # use the same dictionary as previous training
    else:
        chars = tuple(sorted(set(text)))
    char2int = {ch: i for i, ch in enumerate(chars)}
    unknown = set(text) - set(chars)
    if unknown:
        print(f'Characters in text but not in dictionary: {unknown}')
    encoded = np.array([char2int[ch] for ch in text if ch in char2int], dtype=np.int64)
    print(f'Length of encoded text: {len(encoded)} | Length of dictionary: {len(chars)}')

    val_idx = int(len(encoded) * (1 - args.val_frac))
    train_data, val_data = encoded[:val_idx], encoded[val_idx:]
    if len(val_data) <= args.batch_size * args.seq_length:
        print('Warning: validation data is too small for the given batch_size and seq_length.')
        print('Decrease --batch-size or increase --val-frac.')

    if ckpt:
        net = CharRNN(chars, ckpt['n_hidden'], ckpt['n_layers'], ckpt['drop_prob']).to(device)
        net.load_state_dict(ckpt['state_dict'])
    else:
        net = CharRNN(chars, args.hidden, args.layers, args.dropout).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    start_epoch, step, best_val = 0, 0, float('inf')
    if ckpt:
        opt.load_state_dict(ckpt['optimizer'])
        start_epoch, step, best_val = ckpt['epoch'] + 1, ckpt['step'], ckpt['best_val']
        print(f'Resume from epoch {start_epoch}, step {step}, best val {best_val:.4f}')
    print(net)

    net.train()
    t0 = time.time()
    for e in range(start_epoch, args.epochs):
        h = net.init_hidden(args.batch_size, device)
        for x, y in get_batches(train_data, args.batch_size, args.seq_length):
            step += 1
            x = torch.from_numpy(x).long().to(device)
            y = torch.from_numpy(y).long().to(device)
            h = tuple(each.detach() for each in h)       # detach hidden state to prevent backpropagating through entire training history

            opt.zero_grad()
            out, h = net(x, h)
            loss = criterion(out, y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            opt.step()

            if step % args.eval_every == 0:
                val_loss = evaluate(net, val_data, args.batch_size, args.seq_length, device, criterion)
                mark = ''
                if val_loss < best_val:
                    best_val = val_loss
                    save_ckpt(best_path, net, opt, e, step, best_val)
                    mark = '  <- best, saved'
                print(f'Epoch {e + 1}/{args.epochs} | Step {step} | Loss {loss.item():.4f} | '
                      f'Val {val_loss:.4f} | {time.time() - t0:.0f}s{mark}', flush=True)

        save_ckpt(last_path, net, opt, e, step, best_val)  # save checkpoint after each epoch
        print(f'== Finished epoch {e + 1}, saved {last_path}', flush=True)

    print('Best validation loss:', best_val)


if __name__ == '__main__':
    main()
