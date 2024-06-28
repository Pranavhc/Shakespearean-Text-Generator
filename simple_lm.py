import torch 
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
batch_size = 32
block_size = 8 # context length
max_iters = 10000
eval_interval = 1000
learning_rate = 0.001
device = 'cude' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# --------------------------------------------

torch.manual_seed(1337)

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

# data loader
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(high=len(data) - batch_size, size=(batch_size,)) # random indices betweeen 0 to len(data)-batch_size
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

class SimpleLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) # representing each token as a vector of dim=65 (vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are of shape (batch_size, block_size) or (B, T)
        logits = self.token_embedding_table(idx) # shape: (batch_size, block_size, vocab_size) or (B, T, C)
        if targets is None: loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
model = SimpleLanguageModel(vocab_size).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # estimate loss on train and val datasets
    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        print(f'{iter:5d}/{max_iters}: Train loss - {losses["train"]:.4f} - Val loss - {losses["val"]:.5f}')

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate some text
context = torch.tensor(encode('The meaning of life is')).reshape(1, -1).to(device)
generated = model.generate(context, 500)[0].tolist()
print(f"\n[Generated Text]:\n {decode(generated)}")