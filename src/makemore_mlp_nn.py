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
C = torch.rand(27, 3)
W1 = torch.rand(9, 180)
b1 = torch.rand(180)
W2 = torch.rand(180, 27)
b2 = torch.rand(27)
parameters = [C, W1, b1, W2, b2]

for p in parameters:
    p.requires_grad_()

steps = 58000
lrs = torch.linspace(-3, 0, steps)
lrs = 10**lrs

for i in range(steps):
    # minibatch construct
    ix = torch.randint(0, X.shape[0], (32,))
    # forward pass
    # shape: (batch_size, block_size, emb_dim) basically 32 samples, each with 3 characters, each character has 2 dimensions
    emb = C[X[ix]]
    h = torch.tanh(emb.view(-1, 9) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y[ix])
    # print(loss.item())

    # backward pass
    learning_rate = 0.01
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

print(loss.item())


# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):

    out = []
    context = [0] * block_size  # initialize with all ...
    while True:
        emb = C[torch.tensor([context])]  # (1,block_size,d)
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break

    print(''.join(itos[i] for i in out))


""" # Plot the 2D embedding
# visualize dimensions 0 and 1 of the embedding matrix C for all characters
plt.figure(figsize=(8, 8))
plt.scatter(C[:, 0].data, C[:, 1].data, s=200)
for i in range(C.shape[0]):
    plt.text(C[i, 0].item(), C[i, 1].item(), itos[i],
             ha="center", va="center", color='white')
plt.grid('minor')

plt.show()  # Add this line to display the plot
 """


# Plot the embedding in 3D
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(C[:, 0].data, C[:, 1].data, C[:, 2].data, s=200)
for i in range(C.shape[0]):
    ax.text(C[i, 0].item(), C[i, 1].item(), C[i, 2].item(), itos[i],
            ha="center", va="center")
plt.show()
