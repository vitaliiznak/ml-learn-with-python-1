import os
import torch
import torch.nn as nn
from torch.nn import functional as F
# ------------


torch.manual_seed(1337)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
learning_rate = 1e-2
eval_iters = 200
max_iters = 5000
# get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the data file
data_file_path = os.path.join(script_dir, '../data/input.txt')

with open(data_file_path, 'r') as file:
    text = file.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)


#create a mapping of characters to integers
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda x: ''.join([itos[i] for i in x])

# incode entire training text dataset
data = torch.tensor(encode(text), dtype=torch.long)

# Let's now split u[p the data into train and validation sets
n = int( 0.9 * len(data))
train_data, val_data = data[:n], data[n:]


batch_size = 32  # how many independent sequences will we process in parallel?
block_size = 8  # what is the maximum context length for predictions?
# data loader
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model: nn.Module):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim = vocab_size):
        super(BigramLanguageModel, self).__init__()
        self.embeddings_table = nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, idx, targets = None):
        # idx and targets are both (B, T) tensors of integers
        logits = self.embeddings_table(idx)  # (B, T, C)
        if targets is None:
          loss = None
        else:
          B, T, C = logits.shape
          logits = logits.view(B * T, C)
          targets = targets.view(B * T)
          loss = F.cross_entropy(logits, targets)
        return logits, loss
      
    def generate(self, x, max_new_tokens=100):
      for _ in range(max_new_tokens):
          logits, loss = self(x)
  
          logits = logits[:, -1, :]
          probs = F.softmax(logits, dim=-1)
          x = torch.cat((x, torch.multinomial(probs, num_samples=1)), dim=1)
      return x

model = BigramLanguageModel(vocab_size)
m = model.to(device)


# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# train the model
for iter in range(max_iters):
    if iter % eval_iters == 0:
        loss = estimate_loss(model)
        print(f"step {iter}: train loss {loss['train']:.4f}, val loss {loss['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))