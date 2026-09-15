
from torch import nn


def masked(x, mask):

    return x.masked_fill(~mask.unsqueeze(-1), 0)


def masked_mean(x, mask):
    return masked(x, mask).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)


def valid_softmax(scores, key_mask):
    return scores.masked_fill(~key_mask[:, None, :], float("-inf")).softmax(dim=-1)


class FeedForward(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, dim))

    def forward(self, x):
        return self.network(x)
