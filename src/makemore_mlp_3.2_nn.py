import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import os

# Load data
def load_words(filepath):
    with open(filepath, 'r') as file:
        return file.read().splitlines()

def build_vocabulary(words):
    chars = sorted(list(set(''.join(words))))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos, len(itos)

# Dataset building
def build_dataset(words, block_size, stoi):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)

# Model components
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out))
        self.bias = torch.zeros(fan_out) if bias else None
    def __call__(self, x):
        out = x @ self.weight
        if self.bias is not None:
            out += self.bias
        return out
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True)
        else:
            xmean, xvar = self.running_mean, self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        out = self.gamma * xhat + self.beta
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return out
    def parameters(self):
        return [self.gamma, self.beta]

class Tanh:
    def __call__(self, x):
        return torch.tanh(x)
    def parameters(self):
        return []

def main():
    words = load_words('data/names.txt')
    stoi, itos, vocab_size = build_vocabulary(words)
    n_embd, n_hidden, block_size = 10, 100, 3
    n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))

    Xtr, Ytr = build_dataset(words[:n1], block_size, stoi)
    Xdev, Ydev = build_dataset(words[n1:n2], block_size, stoi)
    Xte, Yte = build_dataset(words[n2:], block_size, stoi)

    g = torch.Generator().manual_seed(2147483647)
    C = torch.randn((vocab_size, n_embd), generator=g)
    layers = [
        Linear(n_embd * block_size, n_hidden), Tanh(),
        Linear(n_hidden, n_hidden), BatchNorm1d(n_hidden), Tanh(),
        Linear(n_hidden, vocab_size)
    ]
    parameters = [C] + [p for layer in layers for p in layer.parameters()]
    for p in parameters:
        p.requires_grad = True

    max_steps, batch_size, lossi = 20000, 32, []
    for i in range(max_steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
        Xb, Yb = Xtr[ix], Ytr[ix]
        emb = C[Xb].view(batch_size, -1)
        X = emb
        for layer in layers:
            X = layer(X)
        loss = F.cross_entropy(X, Yb)
        for p in parameters:
            p.grad = None
        loss.backward()
        lr = 0.1 if i < 100000 else 0.01
        for p in parameters:
            p.data += -lr * p.grad
        if i % 1000 == 0:
            print(f'Step {i}: Loss {loss.item()}')
        lossi.append(loss.item())

    plt.plot(lossi)
    plt.title('Training Loss')
    plt.show()

if __name__ == '__main__':
    main()
