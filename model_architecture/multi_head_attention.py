import torch
import torch.nn as nn
from torch.nn import functional as F


_SDPA_HAS_GQA = "enable_gqa" in (F.scaled_dot_product_attention.__doc__ or "")


def _apply_rotary(x, cos, sin):
    """Apply RoPE without allocating a duplicated full-width frequency matrix."""
    x_even, x_odd = x[..., ::2], x[..., 1::2]
    rotated = torch.stack(
        (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1
    )
    return rotated.flatten(-2)


class MultiHeadAttention(nn.Module):
    """Fused QKV attention with SDPA, optional GQA/RoPE, and a compact KV cache."""

    def __init__(
        self,
        num_heads,
        head_size,
        n_embd,
        dropout,
        block_size,
        n_kv_head=None,
        position_encoding="learned",
        bias=True,
    ):
        super().__init__()
        del block_size  # SDPA creates its causal mask internally.
        self.n_head = num_heads
        self.n_kv_head = n_kv_head or num_heads
        if self.n_head % self.n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head")
        if head_size % 2 and position_encoding == "rope":
            raise ValueError("RoPE requires an even attention head size")

        self.head_size = head_size
        self.n_embd = n_embd
        self.position_encoding = position_encoding
        self.kv_dim = self.n_kv_head * head_size

        # Keeping this parameter name and layout preserves published checkpoint support
        # when n_kv_head == n_head.
        self.c_attn = nn.Linear(n_embd, n_embd + 2 * self.kv_dim, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        if position_encoding == "rope":
            inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, head_size, 2, dtype=torch.float32) / head_size)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rope(self, q, k, position_offset):
        positions = torch.arange(
            position_offset,
            position_offset + q.size(-2),
            device=q.device,
            dtype=self.inv_freq.dtype,
        )
        freqs = torch.outer(positions, self.inv_freq)
        cos = freqs.cos().to(dtype=q.dtype)[None, None, :, :]
        sin = freqs.sin().to(dtype=q.dtype)[None, None, :, :]
        return _apply_rotary(q, cos, sin), _apply_rotary(k, cos, sin)

    def forward(self, x, past_kv=None, use_cache=False, position_offset=0):
        batch, sequence, channels = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split((self.n_embd, self.kv_dim, self.kv_dim), dim=2)

        q = q.view(batch, sequence, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(batch, sequence, self.n_kv_head, self.head_size).transpose(1, 2)
        v = v.view(batch, sequence, self.n_kv_head, self.head_size).transpose(1, 2)

        if self.position_encoding == "rope":
            q, k = self._rope(q, k, position_offset)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)
        present = (k, v) if use_cache else None

        # Recent CUDA SDPA kernels accept grouped-query K/V directly. Older PyTorch
        # versions and non-CUDA backends use the compatible expanded view instead.
        native_gqa = (
            self.n_kv_head != self.n_head
            and q.device.type == "cuda"
            and _SDPA_HAS_GQA
            # Native grouped-query SDPA can fall back to the slow math kernel on
            # Turing/Pascal. Expanding K/V lets those GPUs use their best available
            # fused/memory-efficient kernel; Ampere and newer use native GQA.
            and torch.cuda.get_device_capability(q.device)[0] >= 8
        )
        if self.n_kv_head != self.n_head and not native_gqa:
            repeats = self.n_head // self.n_kv_head
            attention_k = k.repeat_interleave(repeats, dim=1)
            attention_v = v.repeat_interleave(repeats, dim=1)
        else:
            attention_k, attention_v = k, v

        # During one-token cached decoding every key is in the past, so no mask is
        # needed. A causal mask is required for full-sequence training/prefill.
        is_causal = past_kv is None and sequence > 1
        sdpa_args = {
            "attn_mask": None,
            "dropout_p": self.attn_dropout.p if self.training else 0.0,
            "is_causal": is_causal,
        }
        if native_gqa:
            sdpa_args["enable_gqa"] = True
        y = F.scaled_dot_product_attention(q, attention_k, attention_v, **sdpa_args)
        y = y.transpose(1, 2).contiguous().view(batch, sequence, channels)
        y = self.resid_dropout(self.c_proj(y))
        return y, present
