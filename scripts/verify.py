
import argparse
import gc
import json
import platform
import time
from dataclasses import replace
from pathlib import Path

import torch

from mgn.config import ModelConfig, load_config
from mgn.losses import compute_loss
from mgn.models import MGN


def make_batch(cfg, batch_size=2):
    batch = {"video_ids": [f"verification-video-{i}" for i in range(batch_size)]}
    for name in ("visual", "question", "audio"):
        length = getattr(cfg, f"max_{name}_len")
        batch[name] = torch.randn(batch_size, length, getattr(cfg, f"{name}_dim"))
        mask = torch.ones(batch_size, length, dtype=torch.bool)
        mask[-1, max(1, length // 2):] = False
        batch[f"{name}_mask"] = mask
    targets = torch.randint(4, 12, (batch_size, cfg.max_answer_len + 1))
    targets[:, -1] = cfg.eos_id
    batch["answer_target"] = targets
    batch["answer_in"] = torch.cat([torch.full((batch_size, 1), cfg.bos_id), targets[:, :-1]], dim=1)
    return batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mgn.yaml")
    parser.add_argument("--output", default="outputs/verification/model_dimensions.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = ModelConfig.from_dict(load_config(args.config)["model"])
    variants = {
        "mgn": {},
        "ppa": {"variant": "ppa"},
        "tpa": {"variant": "tpa", "contrastive": False},
        "ppa_no_self": {"variant": "ppa", "graph_mode": "no_self"},
        "ppa_no_pair": {"variant": "ppa", "graph_mode": "no_pair", "contrastive": False},
        "ppa_no_graph": {"variant": "ppa", "graph_mode": "none", "contrastive": False},
        "ppa_no_contrastive": {"variant": "ppa", "contrastive": False},
        "ppa_visual": {"variant": "ppa", "modalities": "v", "contrastive": False},
        "ppa_audio": {"variant": "ppa", "modalities": "a", "contrastive": False},
    }
    records = []
    for name, changes in variants.items():
        torch.manual_seed(123)
        current = replace(cfg, **changes)
        batch = make_batch(current)
        model = MGN(current, 12)
        start = time.perf_counter()
        outputs = model(batch)
        losses = compute_loss(outputs, batch["answer_target"])
        assert outputs["logits"].shape == (2, current.max_answer_len + 1, 12)
        assert torch.isfinite(losses["loss"])
        losses["loss"].backward()
        missing_grad = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
        assert not missing_grad, f"{name}: disconnected parameters {missing_grad}"
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(bool(g.abs().sum() > 0) for g in grads)
        model.eval()
        prediction = model.generate({k: v for k, v in batch.items() if not k.startswith("answer_")}, max_length=4)
        assert prediction.shape == (2, 4)
        record = dict(variant=name, passed=True, batch_size=2,
                      input_shapes={m: list(batch[m].shape) for m in ("visual", "question", "audio")},
                      parameters=sum(p.numel() for p in model.parameters()),
                      fp32_weight_bytes=sum(p.numel() * 4 for p in model.parameters()),
                      nll=float(losses["nll"].detach()), contrastive=float(losses["contrastive"].detach()),
                      seconds=time.perf_counter() - start)
        records.append(record)
        print(json.dumps(record), flush=True)
        del model, outputs, losses, grads
        gc.collect()
    report = dict(passed=True, python=platform.python_version(), torch=torch.__version__, device="cpu", variants=records)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
