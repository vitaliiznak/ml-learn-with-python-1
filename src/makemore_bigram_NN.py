import torch
import torch.nn.functional as F

# Initialize the stoi dictionary (character to index mapping)
# Example word list for testing
# Ensure the file path is correct
words = open('../data/names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {ch: i + 1 for i, ch in enumerate(chars)}
stoi['.'] = 0
itos = {i: ch for ch, i in stoi.items()}

# Create the training set of bigrams and targets (x, y)
xs, ys = [], []

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])

xsT = torch.tensor(xs)
ysT = torch.tensor(ys)

# Forward pass
g = torch.Generator().manual_seed(2147483647)
xenc = F.one_hot(xsT, num_classes=27).float()

# Initialize weights
W = torch.randn(27, 27, requires_grad=True)

# Define a learning rate
learning_rate = 8

# Training loop (for illustration, using 400 iterations)
for i in range(400):
    # Compute logits
    logits = xenc @ W

    # Compute probabilities
    counts = logits.exp()
    probs = counts / counts.sum(dim=1, keepdim=True)

    # Compute the loss
    loss = -probs[torch.arange(len(ysT)),
                  ysT].log().mean() + 0.01 * (W**2).mean()
    print(f'Iteration {i + 1}, loss={loss.item()}')

    # Backward pass
    W.grad = None  # zero the gradients
    loss.backward()

    # Update the weights
    with torch.no_grad():
        W -= learning_rate * W.grad

# Sample from the model
for i in range(5):
    out = []
    ix = 0
    while True:
        with torch.no_grad():
            xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
            logits = xenc @ W  # predict the log-counts
            counts = logits.exp()
            probs = counts / counts.sum(dim=1, keepdim=True)

            ix = torch.multinomial(probs, num_samples=1,
                                   replacement=True, generator=g).item()
            out.append(itos[ix])
            if ix == 0:
                break
    print(''.join(out))
