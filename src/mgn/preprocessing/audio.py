
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import torch

from .common import pool_sequence, sha256_path


def _run(args: list[str]) -> None:
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode:
        raise RuntimeError(f"{args[0]} failed ({result.returncode}): {result.stderr.decode(errors='replace')}")


def rnnoise_waveform(
    path: str | Path, executable: str | Path, *, ffmpeg: str = "ffmpeg"
) -> np.ndarray:





    path = Path(path).resolve(strict=True)
    located = shutil.which(str(executable))
    if not located:
        raise FileNotFoundError(f"RNNoise executable not found: {executable}")
    with tempfile.TemporaryDirectory(prefix="mgn-rnnoise-") as temp:
        root = Path(temp)
        raw, padded, denoised, trimmed, final = [root / name for name in ("raw.pcm", "padded.pcm", "denoised.pcm", "trimmed.pcm", "final.f32")]
        _run([ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(path), "-vn", "-ac", "1",
              "-ar", "48000", "-f", "s16le", str(raw)])
        pcm = np.fromfile(raw, dtype="<i2")
        if not len(pcm):
            raise ValueError("Input contains no audio samples")
        size = ((len(pcm) + 479) // 480 + 1) * 480
        np.pad(pcm, (0, size - len(pcm))).astype("<i2").tofile(padded)
        _run([located, str(padded), str(denoised)])
        clean = np.fromfile(denoised, dtype="<i2")
        if len(clean) != size - 480:
            raise ValueError("RNNoise CLI output does not match upstream rnnoise_demo framing; verify the executable")
        clean[:len(pcm)].tofile(trimmed)
        _run([ffmpeg, "-nostdin", "-v", "error", "-y", "-f", "s16le", "-ar", "48000",
              "-ac", "1", "-i", str(trimmed), "-ar", "16000", "-f", "f32le", str(final)])
        waveform = np.fromfile(final, dtype="<f4")
    if not len(waveform) or not np.isfinite(waveform).all():
        raise ValueError("Invalid denoised waveform")
    return waveform


class XLSRExtractor:


    def __init__(self, checkpoint: str | Path, *, device: str = "cpu", pooled_length: int = 100):
        self.checkpoint = Path(checkpoint).resolve(strict=True)
        if pooled_length < 1:
            raise ValueError("Pooled length must be positive")
        try:
            import transformers
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
        except ImportError as error:
            raise ImportError("XLSR extraction requires the optional transformers package") from error
        self.transformers_version = transformers.__version__
        self.device = torch.device(device)
        self.model = Wav2Vec2Model.from_pretrained(str(self.checkpoint), local_files_only=True).to(self.device).eval()
        if self.model.config.conv_dim[-1] != 512:
            raise ValueError("This adapter requires an XLSR convolutional output dimension of 512")
        self.processor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000,
                                                 padding_value=0.0, do_normalize=True,
                                                 return_attention_mask=True)
        self.pooled_length = pooled_length
        self.checkpoint_hash = sha256_path(self.checkpoint)

    @torch.inference_mode()
    def __call__(self, waveform: np.ndarray, sampling_rate: int = 16000) -> torch.Tensor:
        waveform = np.asarray(waveform, dtype=np.float32)
        if sampling_rate != 16000 or waveform.ndim != 1 or len(waveform) < 400 or not np.isfinite(waveform).all():
            raise ValueError("XLSR expects finite mono 16kHz audio with at least 400 samples")
        inputs = self.processor(waveform, sampling_rate=sampling_rate, return_tensors="pt")
        outputs = self.model(**{key: value.to(self.device) for key, value in inputs.items()}, return_dict=True)
        features = outputs.extract_features[0]
        if features.ndim != 2 or features.shape[1] != 512 or not torch.isfinite(features).all():
            raise ValueError("Unexpected XLSR extract_features shape or values")
        return pool_sequence(features, self.pooled_length).cpu()

    def extract(self, path: str | Path, *, rnnoise: str | Path, ffmpeg: str = "ffmpeg"):
        waveform = rnnoise_waveform(path, rnnoise, ffmpeg=ffmpeg)
        executable = shutil.which(str(rnnoise))
        metadata = {
            "extractor": "Wav2Vec2Model/XLSR", "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_hash,
            "output_layer": "extract_features: convolutional features after feature_projection layer normalization",
            "source_sha256": sha256_path(path), "torch_version": torch.__version__,
            "transformers_version": self.transformers_version,
            "preprocessing": {"denoise": "upstream rnnoise_demo", "rnnoise_sha256": sha256_path(executable),
                              "rnnoise_rate": 48000, "rnnoise_input": "mono signed-16 little-endian PCM",
                              "rnnoise_drain_samples": 480, "resample_rate": 16000,
                              "waveform_normalization": "zero-mean-unit-variance",
                              "pool": "adaptive_avg_pool1d", "pool_length": self.pooled_length},
        }
        return self(waveform), metadata
