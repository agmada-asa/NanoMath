import torch.nn as nn
from torch.nn import functional as F


class SwiGLU(nn.Module):
    """Parameter-efficient gated MLP used by the optimized model profiles."""

    def __init__(self, n_embd, dropout, bias=False, multiple_of=64):
        super().__init__()
        hidden = int(8 * n_embd / 3)
        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        self.gate_up = nn.Linear(n_embd, 2 * hidden, bias=bias)
        self.down = nn.Linear(hidden, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.dropout(self.down(F.silu(gate) * up))


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout, mlp_type="relu", bias=True):
        super().__init__()
        self.mlp_type = mlp_type
        if mlp_type == "swiglu":
            self.net = SwiGLU(n_embd, dropout, bias=bias)
        elif mlp_type == "relu":
            # The layout intentionally matches the original published checkpoints.
            self.net = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.ReLU(),
                nn.Linear(4 * n_embd, n_embd),
                nn.Dropout(dropout),
            )
        else:
            raise ValueError(f"Unknown mlp_type: {mlp_type}")

    def forward(self, x):
        return self.net(x)
