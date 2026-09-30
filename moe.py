"""MoE layer with top-k routing and load balancing."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """One expert — same shape as the dense FFN."""
    def __init__(self, n_embd, hidden):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, hidden)
        self.fc2 = nn.Linear(hidden, n_embd)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class MoE(nn.Module):
    """Mixture of Experts with top-k routing and a load-balancing loss."""
    def __init__(self, n_embd, hidden, n_experts=4, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([Expert(n_embd, hidden) for _ in range(n_experts)])
        self.router = nn.Linear(n_embd, n_experts, bias=False)
        self.last_router_logits = None
        self.last_topk_idx = None

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C)

        router_logits = self.router(x_flat)
        self.last_router_logits = router_logits

        weights, indices = router_logits.topk(self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        self.last_topk_idx = indices

        out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_idx = indices[:, k]
            w = weights[:, k].unsqueeze(-1)
            for e in range(self.n_experts):
                mask = (expert_idx == e)
                if mask.any():
                    out[mask] += w[mask] * self.experts[e](x_flat[mask])

        return out.view(B, T, C)

    def load_balance_loss(self):
        """Standard Switch-Transformer auxiliary loss."""
        logits = self.last_router_logits
        idx = self.last_topk_idx
        n = self.n_experts

        one_hot = F.one_hot(idx, n).float()
        tokens_per_expert = one_hot.sum(dim=(0, 1))
        fraction = tokens_per_expert / tokens_per_expert.sum()

        probs = F.softmax(logits, dim=-1)
        mean_prob = probs.mean(dim=0)

        return n * (fraction * mean_prob).sum()
