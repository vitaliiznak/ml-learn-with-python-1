import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Configuration
BLOCK_SIZE = 8  # context length: how many characters do we take to predict the next one?
N_EMBD = 10     # the dimensionality of the character embedding vectors
N_HIDDEN = 68   # the number of neurons in the hidden layer of the MLP
BATCH_SIZE = 32
MAX_STEPS = 10000

class Linear:
    def __init__(self, fan_in: int, fan_out: int, bias: bool = True):
        self.weight = torch.randn((fan_in, fan_out), requires_grad=True)
        self.bias = torch.zeros(fan_out, requires_grad=True) if bias else None
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self) -> List[torch.Tensor]:
        return [self.weight] + ([] if self.bias is None else [self.bias])

class BatchNorm1d:
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # parameters (trained with backprop)
        self.gamma = torch.ones(dim, requires_grad=True)
        self.beta = torch.zeros(dim, requires_grad=True)
        # buffers (trained with a running 'momentum update')
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            if x.ndim == 2:
                dim = (0,)
            elif x.ndim == 3:
                dim = (0, 1)
            else:
                raise ValueError("only 2D or 3D tensors are supported")
            xmean = x.mean(dim, keepdim=True)
            xvar = x.var(dim, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
            
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out
    
    def parameters(self) -> List[torch.Tensor]:
        return [self.gamma, self.beta]

class Tanh:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self) -> List[torch.Tensor]:
        return []

class Embedding:
    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.weight = torch.randn((num_embeddings, embedding_dim))
    
    def __call__(self, ix: torch.Tensor) -> torch.Tensor:
        self.out = self.weight[ix]
        return self.out
    
    def parameters(self) -> List[torch.Tensor]:
        return [self.weight]

class FlattenConsecutive:
    def __init__(self, n: int):
        self.n = n
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x = x.view(B, T//self.n, C * self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out
    
    def parameters(self) -> List[torch.Tensor]:
        return []

class Sequential:
    def __init__(self, *layers):
        self.layers = layers
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self) -> List[torch.Tensor]:
        return [p for layer in self.layers for p in layer.parameters()]

def build_dataset(words: List[str], block_size: int, stoi: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build training dataset from words."""
    X, Y = [], []
    
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

def create_model(vocab_size: int) -> Sequential:
    """Create the WaveNet model."""
    return Sequential(
        Embedding(vocab_size, N_EMBD),
        FlattenConsecutive(2), Linear(N_EMBD * 2, N_HIDDEN, bias=False), BatchNorm1d(N_HIDDEN), Tanh(),
        FlattenConsecutive(2), Linear(N_HIDDEN * 2, N_HIDDEN, bias=False), BatchNorm1d(N_HIDDEN), Tanh(),
        FlattenConsecutive(2), Linear(N_HIDDEN * 2, N_HIDDEN, bias=False), BatchNorm1d(N_HIDDEN), Tanh(),
        Linear(N_HIDDEN, vocab_size)
    )

def train_model(model: Sequential, Xtr: torch.Tensor, Ytr: torch.Tensor) -> List[float]:
    """Train the model and return loss history."""
    parameters = model.parameters()
    lossi = []
    
    for i in range(MAX_STEPS):
        # minibatch construct
        ix = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,))
        Xb, Yb = Xtr[ix], Ytr[ix]
        
        # forward pass
        logits = model(Xb)
        loss = F.cross_entropy(logits, Yb)
        
        # backward pass
        for p in parameters:
            p.grad = None
        loss.backward()
        
        # update simple SGD
        lr = 0.1 if i < 150000 else 0.01
        for p in parameters:
            if p.grad is not None:
                p.data -= lr * p.grad
        
        if i % 1000 == 0:
            print(f'{i:7d}/{MAX_STEPS:7d}: {loss.item():.4f}')
        lossi.append(loss.log10().item())
    
    return lossi

def generate_names(model: Sequential, itos: Dict[int, str], num_samples: int = 20) -> None:
    """Generate sample names from the trained model."""
    for _ in range(num_samples):
        out = []
        context = [0] * BLOCK_SIZE
        
        while True:
            logits = model(torch.tensor([context]))
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, replacement=True).item()
            out.append(ix)
            context = context[1:] + [ix]
            if ix == 0:
                break
                
        print(''.join(itos[i] for i in out))

def main():
    # Get the script's directory and construct path to data file
    script_dir = Path(__file__).parent
    data_path = script_dir.parent / 'data' / 'names.txt'
    
    # Read data and create vocabulary
    words = open(data_path, 'r').read().splitlines()
    chars = sorted(list(set(''.join(words))))
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i:s for s,i in stoi.items()}
    vocab_size = len(itos)
    
    # Create train/dev/test splits
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))
    
    Xtr, Ytr = build_dataset(words[:n1], BLOCK_SIZE, stoi)
    Xdev, Ydev = build_dataset(words[n1:n2], BLOCK_SIZE, stoi)
    Xte, Yte = build_dataset(words[n2:], BLOCK_SIZE, stoi)
    
    # Create and train model
    model = create_model(vocab_size)
    
    # Initialize weights
    with torch.no_grad():
        for layer in model.layers[:-1]:
            if isinstance(layer, Linear):
                layer.weight *= 0.1
    
    # Train
    losses = train_model(model, Xtr, Ytr)
    
    # Generate samples
    for layer in model.layers:
        layer.training = False
    
    generate_names(model, itos)

if __name__ == "__main__":
    main()
