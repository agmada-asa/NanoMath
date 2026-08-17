import torch
import torch.nn as nn

from model_architecture.feed_forward import FeedForward
from model_architecture.multi_head_attention import MultiHeadAttention


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(dtype=x.dtype)


class Block(nn.Module):
    """Pre-normalized Transformer block."""

    def __init__(
        self,
        n_embd,
        n_head,
        dropout,
        block_size,
        n_kv_head=None,
        norm_type="layernorm",
        mlp_type="relu",
        position_encoding="learned",
        bias=True,
    ):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(
            num_heads=n_head,
            n_kv_head=n_kv_head,
            head_size=head_size,
            n_embd=n_embd,
            dropout=dropout,
            block_size=block_size,
            position_encoding=position_encoding,
            bias=bias,
        )
        self.ffwd = FeedForward(n_embd, dropout, mlp_type=mlp_type, bias=bias)
        norm = RMSNorm if norm_type == "rmsnorm" else nn.LayerNorm
        self.ln1 = norm(n_embd)
        self.ln2 = norm(n_embd)

    def forward(self, x, past_kv=None, use_cache=False, position_offset=0):
        attention, present = self.sa(
            self.ln1(x),
            past_kv=past_kv,
            use_cache=use_cache,
            position_offset=position_offset,
        )
        x = x + attention
        x = x + self.ffwd(self.ln2(x))
        return x, present
