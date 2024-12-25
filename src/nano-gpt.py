import os
import torch
import torch.nn as nn
from torch.nn import functional as F

def main():
    # Set the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the data file
    data_file_path = os.path.join(script_dir, '../data/input.txt')

    # Read the data file to inspect it
    with open(data_file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Get all unique characters in the text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(''.join(chars))
    print(vocab_size)

    # Create mappings for characters to integers and vice versa
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda x: ''.join([itos[i] for i in x])

    # Encode and decode an example string
    encoded_hello = encode('hello')
    print(encoded_hello)
    print(decode(encoded_hello))

    # Encode the entire dataset
    data = torch.tensor(encode(text), dtype=torch.long)

    # Split the data into training and validation sets
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    print(train_data.shape, val_data.shape)
    print('train_data[:100]')
    print(train_data[:100])

    # Define block size and create a training dataset
    block_size = 8
    train_dataset = data.unfold(0, block_size + 1, block_size)
    for train_data in train_dataset:
        for t in range(block_size):
            context = train_data[:t+1]
            target = train_data[t+1]
            print(f'when input is {context}, target is {target}')
        break

    # Set random seed for reproducibility
    torch.manual_seed(1337)

    # Define batch size and block size
    batch_size = 4  # Number of sequences processed in parallel

    def get_batch(split):
        # Generate a batch of data for inputs x and targets y
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x, y

    xb, yb = get_batch('train')
    print('inputs')
    print(xb.shape)
    print(xb)
    print('targets')
    print(yb.shape)
    print(yb)

    # Define the BigramLanguageModel class
    class BigramLanguageModel(nn.Module):
        def __init__(self, vocab_size, embedding_dim):
            super(BigramLanguageModel, self).__init__()
            self.embeddings_table = nn.Embedding(vocab_size, embedding_dim)

        def forward(self, idx, targets=None):
            # idx and targets are (B, T) tensors of integers
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
                logits, _ = self.forward(x)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                x = torch.cat((x, torch.multinomial(probs, num_samples=1)), dim=1)
            return x

    # Initialize the model
    m = BigramLanguageModel(vocab_size, vocab_size)
    logits, loss = m(xb, yb)
    print(loss)

    # Generate text using the model
    idx = torch.zeros(1, 1, dtype=torch.int)
    generate_encoded_text = m.generate(idx, max_new_tokens=100)[0].tolist()
    print(decode(generate_encoded_text))

if __name__ == "__main__":
    main()