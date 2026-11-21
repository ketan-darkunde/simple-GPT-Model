# The model. Both trainer.py and chat.py import it, so it exists only once.

import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    #Attention (characters talk to each other) + MLP (each thinks alone), with residual connections ('x + ...' = improve on what you had)
    def __init__(self, d):
        super().__init__()
        self.qkv = nn.Linear(d, 3 * d, bias=False) # query, key, value in one go
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.scale = d ** -0.5

    def forward(self, x, mask):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(mask == 0, float('-inf')) # can't peek at the future
        x = x + self.proj(F.softmax(scores, dim=-1) @ v) # mix of earlier chars
        return x + self.mlp(self.norm2(x))

class SimpleGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, n_layers=2, block_size=64):
        super().__init__()
        self.block_size = block_size
        self.mask = torch.tril(torch.ones(block_size, block_size))
        self.tok_emb = nn.Embedding(vocab_size, embed_dim) # char -> vector
        self.pos_emb = nn.Embedding(block_size, embed_dim) # position -> vector

        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            block = Block(embed_dim)
            self.blocks.append(block)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size) # vector -> next-char scores

    def forward(self, x, targets=None):
        T = x.shape[1]
        mask = self.mask[:T, :T].to(x.device)
        h = self.tok_emb(x) + self.pos_emb(torch.arange(T, device=x.device))
        for block in self.blocks:
            h = block(h, mask)
        logits = self.head(self.norm(h))

        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss