import torch 
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
batch_size = 64         
block_size = 256        # context length
learning_rate = 3e-4

n_embd = 384            # embedding dimension, aka "model dimension" or "d_model"
n_head = 6              # number of heads in the multi-head self-attention
n_blocks = 6            # number of transformer blocks
dropout = 0.2           # dropout rate

max_iters = 5000        # train epochs
eval_interval = 500     # evaluate the model every 500 iterations
eval_iters = 200        # eval epochs

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f: text = f.read()

# all the unique characters 
chars = sorted(list(set(text)))
vocab_size = len(chars)

# char to index and index to char
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [char_to_idx[c] for c in s]
decode = lambda li: ''.join([idx_to_char[i] for i in li])

# train-test split
data = torch.tensor(encode(text), dtype=torch.long, device=device)
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]

# data loader - gives a batch of block_size characters
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(high=len(data) - block_size, size=(batch_size,)) # random indices betweeen 0 to len(data)-batch_size
    x = torch.stack([data[i:i+block_size] for i in ix])                 # next block_size characters from each random integer
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])             # next block_size characters offset by 1
    x,y = x.to(device), y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    
    return out

class Head(nn.Module):
    """One head of self-attenttion"""

    def __init__(self, head_size):
        super().__init__()
        
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)).to(device)) # a convention in pytorch to store non-parameter tensors

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, C)
        q = self.query(x) # (B, T, C)

        # compute the attenstion scores, the "affinities" between each query and key
        w = q @ k.transpose(1,2) * C**-0.5 # (B, T, C) "scaled @" (B, C, T) -> (B, T, T)
        w = w.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # to mask out the scores of future tokens
        w = F.softmax(w, dim=-1) # (B, T, T)
        w = self.dropout(w) 

        # perform the weighted aggregation of the values
        v = self.value(x)   # (B, T, C)
        out = w @ v         # (B, T, T) @ (B, T, C) -> (B, T, C)

        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward (self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # concatenate the outputs of each head
        out = self.projection(out) # linear projection to n_embd
        out = self.dropout(out)    # apply dropout for regularization
        return out

class FeedForward(nn.Module):
    """ simple feed forward layer followed by a ReLU activation"""

    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(  
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), 
            nn.Dropout(dropout) # dropout for regularization
        )
    
    def forward(self, x): return self.net(x)
    
class Block(nn.Module):
    """ Transformer block: Communication folowed by computation"""
    def __init__(self, n_heads, n_embd):
        super().__init__()
        
        self.sa = MultiHeadAttention(n_heads, n_embd // n_heads) # n_heads heads of n_embd//n_heads-dimentional self-attention
        self.ffwd = FeedForward(n_embd)
        
        # normalizing the features (or n_embd-dimentional vectors) before and after the self-attention
        self.ln1 = nn.LayerNorm(n_embd) 
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  
        x = x + self.ffwd(self.ln2(x))
        return x

class AutoregressiveTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)       # returns a vector of dim=n_embd for each token
        self.positional_embedding_table = nn.Embedding(block_size, n_embd)  
        
        # single block is n_head heads of n_embd//n_head-dimentional self-attention followed by a feed forward layer
        self.blocks = nn.Sequential(*[Block(n_head, n_embd) for _ in range(n_blocks)]) # (B, T, n_embd)
        
        self.ln_final = nn.LayerNorm(n_embd)            # normalizing the features
        self.lm_head = nn.Linear(n_embd, vocab_size)    # language model head, up projection to the vocab_size

    def forward(self, idx, targets=None):
        # idx and targets are of shape (batch_size, block_size) or (B, T)
        B, T = idx.shape

        token_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        pos_emb = self.positional_embedding_table(torch.arange(T, device=device)) # (B, n_embd)
        
        x = token_emb + pos_emb            # (B, T, n_embd)
        x = self.blocks(x)                 # (B, T, n_embd)
        x = self.ln_final(x)               # (B, T, n_embd)
        
        # Up projection to vocab_size, to get probability distribution over all tokens
        logits = self.lm_head(x)           # (B, T, vocab_size) 
        
        if targets is None: loss = None # for generation
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context

        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens since the positional embedding is fixed to that length
            idx_cond = idx[:, -block_size:] # only consider the last block_size tokens
            
            # focus on the last time step, entire context that the model has processed up to this point.
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # becomes (B, C)
            
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1) append sampled index to the running sequence
        return idx
    
model = AutoregressiveTransformer().to(device)

def train():
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        # estimate loss on train and val datasets
        if iter % eval_interval == 0:
            losses = estimate_loss(model)
            print(f'{iter:5d}/{max_iters}: Train loss - {losses["train"]:.4f} - Val loss - {losses["val"]:.5f}')

        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    torch.save(model.state_dict(), 'saved_transformer.pth')
# train()                                                       # 4500/5000: Train loss - 1.1183 - Val loss - 1.47693

# generate some text
with torch.no_grad():    
    model.load_state_dict(torch.load('saved_transformer.pth', map_location=device))

    # context = torch.zeros(1,1, dtype=torch.long, device=device)
    # context = torch.tensor(encode('The meaning of life is')).reshape(1, -1).to(device)
    context = torch.tensor(encode('I say unto you, what he hath done famously, he did it to that end')).reshape(1, -1).to(device)
    
    generated = model.generate(context, 500)[0].tolist()
    print(f"\n[Generated Text]:\n {decode(generated)}")