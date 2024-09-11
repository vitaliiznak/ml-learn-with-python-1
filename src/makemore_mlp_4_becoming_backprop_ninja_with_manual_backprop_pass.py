import torch
import torch.nn.functional as F

# Load and preprocess data
words = open('../data/names.txt', 'r').read().splitlines()

# Build vocabulary
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)

# Hyperparameters
n_embd = 10
n_hidden = 64
block_size = 3
batch_size = 32

# Split data
n1 = int(0.8*len(words))

def build_dataset(words):
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

Xtr, Ytr = build_dataset(words[:n1])

# Initialize parameters
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((vocab_size, n_embd), generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3)/((n_embd * block_size)**0.5)
b1 = torch.randn(n_hidden, generator=g) * 0.01
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.01
b2 = torch.randn(vocab_size, generator=g) * 0.01
bngain = torch.ones((1, n_hidden)) * 0.01 + 1
bnbias = torch.zeros((1, n_hidden)) * 0.01

parameters = [C, W1, b1, W2, b2, bngain, bnbias]

# Training loop
for _ in range(100):  # Number of iterations
    # Get a batch
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix]

    # Forward pass
    emb = C[Xb]
    embcat = emb.view(emb.shape[0], -1)
    hprebn = embcat @ W1 + b1
    bnmeani = 1/batch_size*hprebn.sum(0, keepdim=True)
    bndiff = hprebn - bnmeani
    bndiff2 = bndiff**2
    bnvar = 1/(batch_size-1)*(bndiff2).sum(0, keepdim=True)
    bnvar_inv = (bnvar + 1e-5)**-0.5
    bnraw = bndiff * bnvar_inv
    hpreact = bngain * bnraw + bnbias
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Yb)

    # Manual backward pass
    dlogits = F.softmax(logits, 1)
    dlogits[range(batch_size), Yb] -= 1
    dlogits /= batch_size

    dh = dlogits @ W2.T
    dW2 = h.T @ dlogits
    db2 = dlogits.sum(0)

    dhpreact = (1 - h**2) * dh
    dbngain = (bnraw * dhpreact).sum(0, keepdims=True)
    dbnraw = bngain * dhpreact
    dbnbias = dhpreact.sum(0, keepdim=True)
    dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)
    dbnvar = -0.5 * (bnvar + 1e-5)**(-1.5) * dbnvar_inv
    dbndiff = bnvar_inv * dbnraw
    dbndiff2 = 1.0 / (batch_size-1) * torch.ones_like(bndiff2) * dbnvar
    dbndiff += 2.0 * bndiff * dbndiff2

    dbnmeani = -(dbndiff).sum(0)
    dhprebn = 1.0 * dbndiff
    dhprebn += torch.ones_like(hprebn) * 1.0/batch_size * dbnmeani

    dembcat = dhprebn @ W1.T
    dW1 = embcat.T @ dhprebn
    db1 = dhprebn.sum(0)

    demb = dembcat.view(emb.shape)

    dC = torch.zeros_like(C)
    for k in range(Xb.shape[0]):
        for j in range(Xb.shape[1]):
            ix = Xb[k,j]
            dC[ix] += demb[k, j]

    # Update parameters
    lr = 0.1
    C -= lr * dC
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    bngain -= lr * dbngain
    bnbias -= lr * dbnbias

    print(f'Loss: {loss.item():.4f}')

# Generate from the model
for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print(''.join(itos[i] for i in out))