
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import sys

import torch

from .common import sha256_path


def _source_module(name: str, repository: Path):
    module = importlib.import_module(name)
    origin = Path(module.__file__).resolve()
    if not origin.is_relative_to(repository):
        raise ImportError(f"{name} resolved outside ChineseBERT repository ({origin}); run extraction in a clean process")
    return module


class ChineseBERTExtractor:






    def __init__(self, repository: str | Path, checkpoint: str | Path, *, max_tokens: int = 25, device: str = "cpu"):
        self.repository = Path(repository).resolve(strict=True)
        self.checkpoint = Path(checkpoint).resolve(strict=True)
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        sys.path.insert(0, str(self.repository))
        try:
            model_class = _source_module("models.modeling_glycebert", self.repository).GlyceBertModel
            dataset_class = _source_module("datasets.bert_dataset", self.repository).BertDataset
        except (ImportError, AttributeError) as error:
            raise ImportError("Cannot import local ChineseBERT GlyceBertModel/BertDataset. Install that repository's compatible dependencies (including pypinyin/tokenizers); the original source uses transformers.modeling_bert.") from error
        finally:
            sys.path.remove(str(self.repository))
        self.device = torch.device(device)
        self.model = model_class.from_pretrained(str(self.checkpoint), local_files_only=True).to(self.device).eval()
        self.dataset = dataset_class(str(self.checkpoint), max_length=max_tokens + 2)


        self.dataset.tokenizer.enable_truncation(max_length=max_tokens + 2)
        self.max_tokens = max_tokens
        self.checkpoint_hash = sha256_path(self.checkpoint)
        self.source_hashes = {name: sha256_path(self.repository / name) for name in
                              ("models/modeling_glycebert.py", "models/fusion_embedding.py", "datasets/bert_dataset.py")}

    @torch.inference_mode()
    def __call__(self, question: str) -> torch.Tensor:
        if not question.strip():
            raise ValueError("Question must contain at least one content token")
        input_ids, pinyin_ids = self.dataset.tokenize_sentence(question)
        if len(input_ids) < 3:
            raise ValueError("Question tokenization produced no content tokens")
        ids = input_ids.unsqueeze(0).to(self.device)
        pinyin = pinyin_ids.reshape(1, len(input_ids), 8).to(self.device)
        outputs = self.model(input_ids=ids, pinyin_ids=pinyin, attention_mask=torch.ones_like(ids), return_dict=True)
        features = outputs.last_hidden_state[0, 1:-1]
        if features.shape != (len(input_ids) - 2, 768) or not torch.isfinite(features).all():
            raise ValueError("ChineseBERT-base must produce finite [content_tokens,768] features")
        return features.cpu()

    def extract(self, question: str):
        metadata = {
            "extractor": "ShannonAI/ChineseBERT/GlyceBertModel", "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_hash, "repository": str(self.repository),
            "source_hashes": self.source_hashes, "output_layer": "last_hidden_state, excluding CLS/SEP",
            "source_text_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
            "torch_version": torch.__version__,
            "preprocessing": {"tokenizer": "author BertDataset WordPiece + pinyin", "glyphs": "checkpoint config resources",
                              "max_content_tokens": self.max_tokens, "truncate": "right", "cache_special_tokens": False},
        }
        return self(question), metadata
