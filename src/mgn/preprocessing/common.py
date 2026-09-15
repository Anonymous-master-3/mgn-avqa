
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def sha256_path(path: str | Path) -> str:

    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"No files to fingerprint: {path}")
    for file in files:
        if path.is_dir():
            digest.update(str(file.relative_to(path)).encode() + b"\0")
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def pool_sequence(features: torch.Tensor, length: int = 100) -> torch.Tensor:

    if features.ndim != 2 or min(features.shape) == 0 or length < 1:
        raise ValueError("Expected a nonempty [time, dimension] tensor and positive length")
    return F.adaptive_avg_pool1d(features.T.unsqueeze(0), length).squeeze(0).T.contiguous()


def write_feature_cache(
    path: str | Path, features: np.ndarray | torch.Tensor, metadata: dict[str, Any]
) -> dict[str, Any]:

    path = Path(path).resolve()
    if path.suffix not in {".npy", ".npz"}:
        raise ValueError("Feature cache suffix must be .npy or .npz")
    for field in ("extractor", "checkpoint", "output_layer", "preprocessing"):
        if not metadata.get(field):
            raise ValueError(f"Missing cache provenance field: {field}")
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 2 or min(array.shape) == 0 or not np.isfinite(array).all():
        raise ValueError("Feature cache must be finite and nonempty with shape [T, D]")
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched = dict(metadata, schema_version=1, shape=list(array.shape), dtype="float32")
    reference = {"path": str(path), "metadata": enriched}
    if path.suffix == ".npz":
        reference["key"] = "features"

    json.dumps(reference, ensure_ascii=False, allow_nan=False)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as stream:
        temp_path = Path(stream.name)
        try:
            if path.suffix == ".npz":
                np.savez_compressed(stream, features=array)
            else:
                np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)
    enriched["cache_sha256"] = sha256_path(path)
    sidecar = path.with_suffix(path.suffix + ".json")
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", delete=False) as stream:
        temp_path = Path(stream.name)
        try:
            json.dump(reference, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temp_path, sidecar)
        finally:
            temp_path.unlink(missing_ok=True)
    return reference
