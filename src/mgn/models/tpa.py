
import torch
from torch import nn
from .common import masked


class TPA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.activation = cfg.tpa_activation
        self.temporal = nn.ParameterDict()
        self.feature = nn.ParameterDict()
        for name, length in [("visual", cfg.max_visual_len), ("question", cfg.max_question_len), ("audio", cfg.max_audio_len)]:
            self.temporal[name] = nn.Parameter(torch.empty(cfg.tpa_rank, length))
            self.feature[name] = nn.Parameter(torch.empty(cfg.hidden_dim, cfg.tpa_feature_rank))
            nn.init.xavier_uniform_(self.temporal[name])
            nn.init.xavier_uniform_(self.feature[name])
        self.reduction = nn.Parameter(torch.full((cfg.tpa_rank,), 1.0 / cfg.tpa_rank))
        self.output = nn.Linear(cfg.tpa_feature_rank, cfg.hidden_dim, bias=False)

    def forward(self, features, masks):
        product = None
        for name in ("visual", "question", "audio"):
            x = masked(features[name], masks[name])
            if x.shape[1] > self.temporal[name].shape[1]:
                raise ValueError(f"{name} exceeds TPA configured capacity")
            projected = torch.einsum("ht,btd,dk->bhk", self.temporal[name][:, :x.shape[1]], x, self.feature[name])
            if self.activation == "relu":
                projected = projected.relu()
            product = projected if product is None else product * projected
        return self.output(torch.einsum("h,bhk->bk", self.reduction, product))
