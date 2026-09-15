
import math
import torch
from torch import nn
from .common import FeedForward, masked


class FeatureEncoder(nn.Module):
    def __init__(self, input_dim, cfg):
        super().__init__()
        dim = cfg.hidden_dim
        self.projection = nn.Linear(input_dim, dim)
        self.conv_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(cfg.encoder_conv_layers)])
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(dim, dim, cfg.encoder_kernel_size, padding=cfg.encoder_kernel_size // 2, groups=dim),
                          nn.Conv1d(dim, dim, 1), nn.ReLU())
            for _ in range(cfg.encoder_conv_layers)
        ])
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, cfg.encoder_heads, dropout=cfg.dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, cfg.dropout)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, mask):
        x = masked(self.projection(masked(x, mask)), mask)
        length, dim = x.shape[1:]
        positions = torch.arange(length, device=x.device, dtype=x.dtype)[:, None]
        scales = torch.exp(torch.arange(0, dim, 2, device=x.device, dtype=x.dtype) * (-math.log(10000) / dim))
        pe = x.new_zeros(length, dim)
        pe[:, 0::2] = (positions * scales).sin()
        pe[:, 1::2] = (positions * scales[:dim // 2]).cos()
        x = masked(x + pe, mask)
        for norm, conv in zip(self.conv_norms, self.convs):
            update = conv(masked(norm(x), mask).transpose(1, 2)).transpose(1, 2)
            x = masked(x + self.dropout(update), mask)
        normed = masked(self.attention_norm(x), mask)
        update = self.attention(normed, normed, normed, key_padding_mask=~mask, need_weights=False)[0]
        x = masked(x + self.dropout(update), mask)
        return masked(x + self.dropout(self.ffn(self.ffn_norm(x))), mask)
