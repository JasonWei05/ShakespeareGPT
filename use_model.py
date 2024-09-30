import torch
import torch.nn as nn
from torch.nn import functional as F

# Set parameters (ensure these match your training script)
block_size = 256
n_embd = 6*64
attention_heads = 6
attention_layers = 6
dropout = 0.25
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the text to rebuild vocabulary and mappings
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Create character mappings
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

# Define encode and decode functions
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Define the model classes (same as your training script)
class Head(nn.Module):
    """One head of self-attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        wei = (q @ k.transpose(-2, -1)) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential( 
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // attention_heads
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    """The language model"""
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, attention_heads) for _ in range(attention_layers)])
        self.layerNorm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)
        logits = self.lm_head(x)  # (B,T,vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -block_size:]  # Crop to block size
            logits, _ = self(idx_crop)
            logits = logits[:, -1, :]  # Focus on the last time step
            probs = F.softmax(logits, dim=-1)  # Get probabilities
            idx_next = torch.multinomial(probs, num_samples=1)  # Sample from distribution
            idx = torch.cat((idx, idx_next), dim=1)  # Append to sequence
        return idx

# Instantiate the model
model = BigramLanguageModel()
model.to(device)

# Load the saved model parameters
model.load_state_dict(torch.load('best_model4-0.pth', map_location=device))
model.eval()  # Set model to evaluation mode

# Generate text
context = torch.zeros((1, 1), dtype=torch.long, device=device)  # Starting token
generated_indices = model.generate(context, max_new_tokens=1000)[0].tolist()
generated_text = decode(generated_indices)

# Print the generated text
print(generated_text)