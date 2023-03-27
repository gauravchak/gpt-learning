"""Using a bigram model to predict the next character in a sequence."""

import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ------------

torch.manual_seed(1337)

with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to indices
stoi = {c: i for i, c in enumerate(chars)}
# create a mapping from indices to characters
itos = {i: c for i, c in enumerate(chars)}
def encode(x):
    '''take a string, output a list of integers'''
    return [stoi[c] for c in x]

def decode(x):
    '''take a list of integers, output a string'''
    return ''.join([itos[i] for i in x])

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
test_data = data[n:]

# data loading
def get_batch(split):
    """get a small batch of data of inputs x and taregts y
    
    extern:
    train_data
    val_data
    batch_size
    block_size
    """
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, size=(batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i+1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model) -> dict:
    """compute the loss
    
    extern:
    eval_iters
    
    """
    out = {}
    model.eval() # set the model to evaluation mode
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # set the model back to training mode
    return out

# super simple Bigram model
class BigramLanguageModel(nn.Module):
    """Single char bigram model"""
    
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = embed_size
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size)
        
    def forward(self, idx, targets=None):
        """
        idx and targets are both (B,T) tensor of integers
        """
        logits = self.token_embedding_table(idx) # (B,T,E)

        if targets is None:
            loss = None
        else:
            B, T, E = logits.shape
            logits = logits.view(B*T, E)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

