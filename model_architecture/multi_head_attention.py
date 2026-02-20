import torch
import torch.nn as nn
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, dropout, block_size):
        super().__init__()
        self.n_head = num_heads
        self.head_size = head_size
        self.n_embd = n_embd

        # COMBINED PROJECTION:
        # Instead of 3 small Linear layers per head, we do 1 giant Linear layer
        # that calculates Query, Key, and Value for ALL heads at the same time.
        # Output size is 3 * n_embd because we need Q, K, and V.
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)

        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()  # Batch, Time, Channels (n_embd)

        # 1. Calculate Q, K, V for all heads in one go
        # shape: (B, T, n_embd) -> (B, T, 3 * n_embd)
        qkv = self.c_attn(x)

        # 2. Split the result into Query, Key, Value
        q, k, v = qkv.split(self.n_embd, dim=2)

        # 3. Reshape for Multi-Head Attention
        # Transform from (B, T, C) -> (B, n_head, T, head_size)
        # This allows us to treat "Heads" as a batch dimension for parallel processing
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # 4. Flash Attention (Optimized Kernel)
        # This is the single fastest way to do attention in PyTorch
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0,
            is_causal=True
        )

        # 5. Re-assemble heads
        # (B, n_head, T, head_size) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 6. Final projection
        y = self.resid_dropout(self.c_proj(y))

        return y