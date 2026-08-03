import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from model_architecture.block import Block, RMSNorm


class GPTLanguageModel(nn.Module):
    """NanoMath decoder with fast cached generation and legacy checkpoint support."""

    def __init__(
        self,
        vocab_size,
        n_embd,
        block_size,
        n_head,
        n_layer,
        device=None,
        dropout=0.0,
        n_kv_head=None,
        norm_type="layernorm",
        mlp_type="relu",
        position_encoding="learned",
        tie_embeddings=False,
        bias=True,
    ):
        super().__init__()
        del device  # Positions follow idx.device, so model.to(...) is always safe.
        if n_embd % n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if position_encoding not in {"learned", "rope"}:
            raise ValueError("position_encoding must be 'learned' or 'rope'")

        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_kv_head = n_kv_head or n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.position_encoding = position_encoding
        self.tie_embeddings = tie_embeddings

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = (
            nn.Embedding(block_size, n_embd) if position_encoding == "learned" else None
        )
        self.blocks = nn.ModuleList(
            [
                Block(
                    n_embd,
                    n_head=n_head,
                    n_kv_head=self.n_kv_head,
                    dropout=dropout,
                    block_size=block_size,
                    norm_type=norm_type,
                    mlp_type=mlp_type,
                    position_encoding=position_encoding,
                    bias=bias,
                )
                for _ in range(n_layer)
            ]
        )
        norm = RMSNorm if norm_type == "rmsnorm" else nn.LayerNorm
        self.ln_f = norm(n_embd)
        # Legacy checkpoints include this bias; optimized profiles disable it.
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=bias)

        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))
        if tie_embeddings:
            self.lm_head.weight = self.token_embedding_table.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_model_config(self):
        first_block = self.blocks[0]
        return {
            "vocab_size": self.vocab_size,
            "n_embd": self.n_embd,
            "block_size": self.block_size,
            "n_head": self.n_head,
            "n_kv_head": self.n_kv_head,
            "n_layer": self.n_layer,
            "dropout": first_block.ffwd.net.dropout.p
            if hasattr(first_block.ffwd.net, "dropout")
            else first_block.ffwd.net[-1].p,
            "norm_type": "rmsnorm" if isinstance(self.ln_f, RMSNorm) else "layernorm",
            "mlp_type": first_block.ffwd.mlp_type,
            "position_encoding": self.position_encoding,
            "tie_embeddings": self.tie_embeddings,
            "bias": first_block.sa.c_proj.bias is not None,
        }

    def forward(
        self,
        idx,
        targets=None,
        loss_weights=None,
        past_key_values=None,
        use_cache=False,
    ):
        batch, sequence = idx.shape
        del batch
        past_length = 0
        if past_key_values:
            past_length = past_key_values[0][0].size(-2)
        if sequence + past_length > self.block_size:
            raise ValueError(
                f"Sequence length {sequence + past_length} exceeds block_size {self.block_size}"
            )

        x = self.token_embedding_table(idx)
        if self.position_embedding_table is not None:
            positions = torch.arange(
                past_length, past_length + sequence, device=idx.device
            )
            x = x + self.position_embedding_table(positions)

        presents = [] if use_cache else None
        for layer_index, block in enumerate(self.blocks):
            past_kv = past_key_values[layer_index] if past_key_values else None
            x, present = block(
                x,
                past_kv=past_kv,
                use_cache=use_cache,
                position_offset=past_length,
            )
            if use_cache:
                presents.append(present)

        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            flat_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
                reduction="none",
            )
            if loss_weights is not None:
                weights = loss_weights.reshape(-1).to(flat_loss.dtype)
                valid = targets.reshape(-1).ne(-1)
                weights = weights * valid
                loss = (flat_loss * weights).sum() / weights.sum().clamp_min(1.0)
            else:
                loss = flat_loss[targets.reshape(-1).ne(-1)].mean()

        if use_cache:
            return logits, loss, presents
        return logits, loss

    @staticmethod
    def _sample(logits, temperature, top_k):
        if temperature <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        logits = logits / temperature
        if top_k:
            k = min(top_k, logits.size(-1))
            cutoff = torch.topk(logits, k).values[:, -1, None]
            logits = logits.masked_fill(logits < cutoff, float("-inf"))
        return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

    @torch.inference_mode()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=0.0,
        top_k=None,
        stop_token_ids=None,
        use_cache=True,
    ):
        """Generate with an O(n) KV-cache path and a checkpoint-compatible fallback."""
        if idx.ndim != 2:
            raise ValueError("idx must have shape (batch, sequence)")
        stop_ids = set()
        if stop_token_ids is not None:
            stop_ids = (
                {int(stop_token_ids)}
                if isinstance(stop_token_ids, int)
                else {int(token_id) for token_id in stop_token_ids}
            )

        output = idx
        past_key_values = None
        logits = None
        for _ in range(max_new_tokens):
            if use_cache and past_key_values is not None:
                model_input = output[:, -1:]
            else:
                model_input = output[:, -self.block_size :]

            if use_cache:
                # Learned positions reset after a sliding-window crop, so rebuild the
                # cache whenever it is full.
                if past_key_values and past_key_values[0][0].size(-2) >= self.block_size:
                    past_key_values = None
                    model_input = output[:, -self.block_size :]
                logits, _, past_key_values = self(
                    model_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            else:
                logits, _ = self(model_input)

            next_token = self._sample(logits[:, -1, :], temperature, top_k)
            output = torch.cat((output, next_token), dim=1)
            if stop_ids and all(token.item() in stop_ids for token in next_token[:, 0]):
                break
        return output
