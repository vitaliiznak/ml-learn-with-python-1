import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Load and preprocess data
words = open('../names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {ch: i+1 for i, ch in enumerate(chars)}
stoi['.'] = 0
itos = {i: ch for ch, i in stoi.items()}

# Build the dataset
block_size = 3  # context length: how many characters to look at to predict the next one
X, Y = [], []

for w in words:  # Using first 5 words for demonstration
    context = [stoi['.']] * block_size
    for ch in w + '.':
        xi = stoi[ch]
        X.append(context)
        Y.append(xi)
        context = context[1:] + [stoi[ch]]

X = torch.tensor(X)
Y = torch.tensor(Y)

# Initialize parameters
C = torch.rand(27, 2)
W1 = torch.rand(6, 100)
b1 = torch.rand(100)
W2 = torch.rand(100, 27)
b2 = torch.rand(27)
parameters = [C, W1, b1, W2, b2]

for p in parameters:
    p.requires_grad_()

steps = 50000
lrs = torch.linspace(-3, 0, steps)
lrs = 10**lrs

for i in range(steps):
    # minibatch construct
    ix = torch.randint(0, X.shape[0], (32,))
    # forward pass
    # shape: (batch_size, block_size, emb_dim) basically 32 samples, each with 3 characters, each character has 2 dimensions
    emb = C[X[ix]]
    h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y[ix])
    print(loss.item())

    # backward pass
    learning_rate = 0.1
    for p in parameters:
        p.grad = None
    loss.backward()
    # update the weights
    lr = lrs[i]
    for p in parameters:
        p.data.sub_(p.grad.data * lr)

    # track the loss
    # lri.append(lr)
    # lossi.append(loss.item())
