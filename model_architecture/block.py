import torch.nn as nn

from model_architecture.feed_forward import FeedForward
from model_architecture.multi_head_attention import MultiHeadAttention

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    # this is just one round of running the attention heads and then running our feedforward

    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(num_heads=n_head, head_size=head_size, n_embd=n_embd, dropout=dropout, block_size=block_size)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
