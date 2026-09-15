
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Iterator

import numpy as np
import torch

from .common import sha256_path


def sample_clip_indices(
    frame_count: int, stride: int = 2, clip_size: int = 16, max_clips: int | None = 20
) -> list[list[int]]:

    if frame_count < 1 or stride < 1 or clip_size < 1 or (max_clips is not None and max_clips < 1):
        raise ValueError("Frame count, stride, clip size and clip limit must be positive")
    sampled = list(range(0, frame_count, stride))
    clips = [sampled[i : i + clip_size] for i in range(0, len(sampled), clip_size)]
    if max_clips is not None:
        clips = clips[:max_clips]
    for clip in clips:
        clip.extend([clip[-1]] * (clip_size - len(clip)))
    return clips


def decode_video_clips(
    path: str | Path, *, ffmpeg: str = "ffmpeg", max_clips: int = 20,
    stride: int = 2, clip_size: int = 16, crop_size: int = 224, resize_short: int = 256,
) -> Iterator[torch.Tensor]:

    path = Path(path).resolve(strict=True)
    if min(max_clips, stride, clip_size, crop_size, resize_short) < 1 or resize_short < crop_size:
        raise ValueError("Invalid video sampling or resize configuration")
    filters = (
        f"select=not(mod(n\\,{stride})),"
        f"scale={resize_short}:{resize_short}:force_original_aspect_ratio=increase,"
        f"crop={crop_size}:{crop_size}"
    )
    args = [ffmpeg, "-nostdin", "-v", "error", "-i", str(path), "-an", "-vf", filters,
            "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    frame_bytes = crop_size * crop_size * 3
    with tempfile.TemporaryFile() as error_stream:
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=error_stream)
        count = 0
        truncated = False
        try:
            assert process.stdout is not None
            for _ in range(max_clips):
                raw = process.stdout.read(frame_bytes * clip_size)
                if not raw:
                    break
                if len(raw) % frame_bytes:
                    raise RuntimeError("ffmpeg returned an incomplete RGB frame")
                frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, crop_size, crop_size, 3).copy()
                if len(frames) < clip_size:
                    frames = np.concatenate([frames, np.repeat(frames[-1:], clip_size - len(frames), axis=0)])
                count += 1
                yield torch.from_numpy(frames).permute(3, 0, 1, 2).float().div_(127.5).sub_(1)
            else:
                truncated = True
        finally:
            interrupted = sys.exc_info()[0] is not None
            if process.poll() is None and (truncated or interrupted):
                process.terminate()
            if process.stdout is not None:
                process.stdout.close()
            try:
                code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                if not interrupted:
                    raise RuntimeError("ffmpeg did not finish decoding")
                code = -1
            error_stream.seek(0)
            errors = error_stream.read().decode(errors="replace")
            if code and not truncated and not interrupted:
                raise RuntimeError(f"ffmpeg failed ({code}): {errors}")
        if not count:
            raise ValueError(f"Video contains no decodable frames: {path}")


class I3DExtractor:






    def __init__(self, checkpoint: str | Path, *, output_layer: str, device: str = "cpu"):
        self.checkpoint = Path(checkpoint).resolve(strict=True)
        if not output_layer.strip():
            raise ValueError("Specify the exported I3D feature output layer")
        self.device = torch.device(device)
        self.model = torch.jit.load(str(self.checkpoint), map_location=self.device).eval()
        self.output_layer = output_layer
        self.checkpoint_hash = sha256_path(self.checkpoint)

    @torch.inference_mode()
    def __call__(self, clips: torch.Tensor) -> torch.Tensor:
        if clips.ndim != 5 or clips.shape[1:] != (3, 16, 224, 224):
            raise ValueError("I3D input must have shape [B,3,16,224,224]")
        outputs = self.model(clips.to(self.device))
        if not isinstance(outputs, torch.Tensor) or outputs.shape != (clips.shape[0], 2048):
            raise ValueError("Exported I3D must return [B,2048]; verify output layer and RGB/flow provenance")
        if not torch.isfinite(outputs).all():
            raise ValueError("I3D produced nonfinite features")
        return outputs.cpu()

    def extract(self, path: str | Path, *, ffmpeg: str = "ffmpeg", max_clips: int = 20):
        features = [self(clip.unsqueeze(0)) for clip in decode_video_clips(path, ffmpeg=ffmpeg, max_clips=max_clips)]
        metadata = {
            "extractor": "I3D/local-torchscript", "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_hash, "output_layer": self.output_layer,
            "source_sha256": sha256_path(path), "torch_version": torch.__version__,
            "preprocessing": {"frame_stride": 2, "clip_frames": 16, "max_clips": max_clips,
                              "selection": "prefix", "tail_padding": "repeat-last", "rgb_range": [-1, 1],
                              "resize_short": 256, "center_crop": 224},
        }
        return torch.cat(features), metadata
