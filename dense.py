"""Small dense model for Tiny Shakespeare. The FFN is what we'll convert to MoE."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseModel(nn.Module):
    def __init__(self, vocab_size, n_embd=128, block_size=64, hidden=256):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        # this FFN will be replaced with MoE
        self.fc1 = nn.Linear(n_embd, hidden)
        self.fc2 = nn.Linear(hidden, n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        h = self.ln1(x)
        h = self.fc2(F.gelu(self.fc1(h)))
        x = x + h
        x = self.ln2(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
