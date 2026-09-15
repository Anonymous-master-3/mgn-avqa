
import torch
from torch import nn
from .common import masked
from .encoders import FeatureEncoder
from .ppa import PPA
from .tpa import TPA
from .decoder import AnswerDecoder


class MGN(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config
        if len({config.pad_id, config.bos_id, config.eos_id}) != 3 or min(config.pad_id, config.bos_id, config.eos_id) < 0:
            raise ValueError("PAD, BOS and EOS must have distinct nonnegative IDs")
        if vocab_size <= max(config.pad_id, config.bos_id, config.eos_id):
            raise ValueError("vocab_size must include all special token IDs")
        self.names = ["question"] + [name for name, code in [("visual", "v"), ("audio", "a")] if code in config.modalities]
        self.encoders = nn.ModuleDict({name: FeatureEncoder(getattr(config, f"{name}_dim"), config) for name in self.names})
        self.ppa = PPA(config) if config.variant in {"mgn", "ppa"} else None
        self.tpa = TPA(config) if config.variant in {"mgn", "tpa"} else None
        self.fusion = nn.Sequential(nn.Linear(2 * config.hidden_dim, config.hidden_dim), nn.ReLU(), nn.Dropout(config.dropout)) if config.variant == "mgn" else None
        self.decoder = AnswerDecoder(config, vocab_size)

    def encode(self, batch, compute_contrastive=True):
        features, masks = {}, {}
        batch_size = None
        for name in self.names:
            x, mask = batch[name], batch[f"{name}_mask"]
            if x.ndim != 3 or x.shape[-1] != getattr(self.config, f"{name}_dim"):
                raise ValueError(f"{name} must be [batch,length,{getattr(self.config, f'{name}_dim')}]")
            if x.shape[0] == 0 or x.shape[1] > getattr(self.config, f"max_{name}_len"):
                raise ValueError(f"invalid batch size or {name} exceeds configured capacity")
            if mask.dtype != torch.bool or mask.shape != x.shape[:2] or not bool(mask.any(dim=1).all()):
                raise ValueError(f"{name}_mask must be boolean with at least one valid position per sample")
            if batch_size is not None and x.shape[0] != batch_size:
                raise ValueError("all modalities must share batch size")
            batch_size = x.shape[0]
            masks[name] = mask
            features[name] = self.encoders[name](x, mask)
        local, contrastive = None, features["question"].sum() * 0
        if self.ppa is not None:
            local, contrastive = self.ppa(features, masks, batch.get("video_ids"), compute_contrastive)
        if self.tpa is not None:
            global_vector = self.tpa(features, masks)
            if local is None:
                return global_vector[:, None, :], torch.ones((batch_size, 1), device=global_vector.device, dtype=torch.bool), contrastive
            repeated = global_vector[:, None, :].expand(-1, local.shape[1], -1)
            local = masked(self.fusion(torch.cat([local, repeated], dim=-1)), masks["question"])
        return local, masks["question"], contrastive

    def forward(self, batch, compute_contrastive=True):
        memory, mask, contrastive = self.encode(batch, compute_contrastive)
        logits = self.decoder(memory, mask, batch["answer_in"])
        return {"logits": logits, "contrastive_loss": contrastive}

    @torch.no_grad()
    def generate(self, batch, max_length=None):
        memory, mask, _ = self.encode(batch, compute_contrastive=False)
        return self.decoder.generate(memory, mask, max_length)
