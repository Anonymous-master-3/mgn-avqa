
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    visual_dim: int = 2048
    question_dim: int = 768
    audio_dim: int = 512
    hidden_dim: int = 512
    max_visual_len: int = 20
    max_question_len: int = 25
    max_audio_len: int = 100
    max_answer_len: int = 15
    graph_layers: int = 2
    encoder_heads: int = 8
    encoder_conv_layers: int = 2
    encoder_kernel_size: int = 7
    dropout: float = 0.3
    tpa_rank: int = 64
    tpa_feature_rank: int = 64
    tpa_activation: str = "relu"
    bilinear_rank: int = 64
    variant: str = "mgn"
    graph_mode: str = "full"
    modalities: str = "va"
    contrastive: bool = True
    contrastive_temperature: float = 1.0
    contrastive_normalize: bool = False
    contrastive_chunk_size: int = 16
    contrastive_checkpoint: bool = True
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2

    def __post_init__(self):
        if (self.pad_id, self.bos_id, self.eos_id) != (0, 1, 2):
            raise ValueError("vocabulary contract requires PAD=0, BOS=1, EOS=2")
        for name in ("visual_dim", "question_dim", "audio_dim", "hidden_dim", "max_visual_len",
                     "max_question_len", "max_audio_len", "max_answer_len", "graph_layers",
                     "encoder_heads", "encoder_kernel_size", "tpa_rank", "tpa_feature_rank",
                     "bilinear_rank", "contrastive_chunk_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.hidden_dim % self.encoder_heads:
            raise ValueError("hidden_dim must be divisible by encoder_heads")
        if self.encoder_kernel_size % 2 != 1:
            raise ValueError("encoder_kernel_size must be odd")
        if self.encoder_conv_layers < 0 or not 0 <= self.dropout < 1:
            raise ValueError("invalid convolution count or dropout")
        if self.variant not in {"mgn", "ppa", "tpa"} or self.graph_mode not in {"full", "no_self", "no_pair", "none"}:
            raise ValueError("unknown model variant or graph mode")
        if self.modalities not in {"va", "v", "a"}:
            raise ValueError("modalities must be va, v or a")
        if self.modalities != "va" and self.variant != "ppa":
            raise ValueError("single-content-modality ablations require variant=ppa")
        if self.contrastive_temperature <= 0 or self.tpa_activation not in {"relu", "identity"}:
            raise ValueError("invalid contrastive temperature or TPA activation")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        return cls(**values)


def load_config(path):
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("configuration must be a mapping")
    unknown = set(config) - {"model", "training", "data", "evaluation"}
    if unknown:
        raise ValueError(f"unknown configuration sections: {sorted(unknown)}")
    model = ModelConfig.from_dict(config.get("model", {}))
    config["model"] = model.to_dict()
    return config
