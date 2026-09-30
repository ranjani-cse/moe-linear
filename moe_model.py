"""MoE model — same architecture as DenseModel but with an MoE FFN.

Includes a convert_from_dense() method that copies the trained dense model's
FFN weights into every expert and freezes the router near-uniform.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from moe import MoE


class MoEModel(nn.Module):
    def __init__(self, vocab_size, n_embd=128, block_size=64, hidden=256,
                 n_experts=4, top_k=2):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.moe = MoE(n_embd, hidden, n_experts=n_experts, top_k=top_k)
        self.ln2 = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        h = self.ln1(x)
        h = self.moe(h)
        x = x + h
        x = self.ln2(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def convert_from_dense(moe_model, dense_model):
    """Copy the dense FFN weights into every expert. Init router near-uniform."""
    # copy embeddings + norms + head from the dense model (they're identical)
    moe_model.tok_emb.load_state_dict(dense_model.tok_emb.state_dict())
    moe_model.pos_emb.load_state_dict(dense_model.pos_emb.state_dict())
    moe_model.ln1.load_state_dict(dense_model.ln1.state_dict())
    moe_model.ln2.load_state_dict(dense_model.ln2.state_dict())
    moe_model.head.load_state_dict(dense_model.head.state_dict())

    # copy the dense FFN weights into EVERY expert
    for expert in moe_model.moe.experts:
        expert.fc1.load_state_dict(dense_model.fc1.state_dict())
        expert.fc2.load_state_dict(dense_model.fc2.state_dict())

    # router: init to near-uniform (small random weights)
    nn.init.normal_(moe_model.moe.router.weight, mean=0.0, std=0.01)

    return moe_model
