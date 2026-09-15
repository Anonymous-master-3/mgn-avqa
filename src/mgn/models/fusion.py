
from torch import nn


class CrossGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.left_gate = nn.Linear(dim, dim)
        self.right_gate = nn.Linear(dim, dim)

    def forward(self, left, right):
        return left * self.right_gate(right).sigmoid(), right * self.left_gate(left).sigmoid()


class BilinearFusion(nn.Module):
    def __init__(self, dim, rank):
        super().__init__()
        self.left = nn.Linear(dim, rank, bias=False)
        self.right = nn.Linear(dim, rank, bias=False)
        self.output = nn.Linear(rank, dim)

    def forward(self, left, right):
        return self.output(self.left(left).relu() * self.right(right).relu())
