




import torch
from torch import nn
from torch.nn import functional as F
from .common import masked, valid_softmax


class SelfGraphLayer(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.projection = nn.Linear(dim, dim, bias=False)
        self.source_score = nn.Linear(dim, 1, bias=False)
        self.target_score = nn.Linear(dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, return_attention=False):
        z = self.projection(masked(x, mask))
        scores = F.leaky_relu(self.source_score(z) + self.target_score(z).transpose(1, 2), negative_slope=0.2)
        weights = valid_softmax(scores, mask)
        out = masked(F.elu(self.dropout(weights) @ z), mask)
        return (out, weights) if return_attention else out


class PairGraphLayer(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.left_projection = nn.Linear(dim, dim, bias=False)
        self.right_projection = nn.Linear(dim, dim, bias=False)
        self.left_message = nn.Linear(dim, dim, bias=False)
        self.right_message = nn.Linear(dim, dim, bias=False)
        self.left_score = nn.Linear(dim, 1, bias=False)
        self.right_score = nn.Linear(dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, left, right, left_mask, right_mask):
        l = self.left_projection(masked(left, left_mask))
        r = self.right_projection(masked(right, right_mask))
        scores = F.leaky_relu(self.left_score(l) + self.right_score(r).transpose(1, 2), negative_slope=0.2)
        left_to_right = valid_softmax(scores, right_mask)
        right_to_left = valid_softmax(scores.transpose(1, 2), left_mask)
        left_out = masked(F.elu(self.dropout(left_to_right) @ self.right_message(masked(right, right_mask))), left_mask)
        right_out = masked(F.elu(self.dropout(right_to_left) @ self.left_message(masked(left, left_mask))), right_mask)
        return left_out, right_out, right_to_left


class SelfGraph(nn.Module):
    def __init__(self, dim, layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList([SelfGraphLayer(dim, dropout) for _ in range(layers)])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class PairGraph(nn.Module):
    def __init__(self, dim, layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList([PairGraphLayer(dim, dropout) for _ in range(layers)])

    def forward(self, left, right, left_mask, right_mask):
        for layer in self.layers:
            left, right, weights = layer(left, right, left_mask, right_mask)
        return left, right, weights
