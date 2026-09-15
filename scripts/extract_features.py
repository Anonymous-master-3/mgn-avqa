
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mgn.preprocessing import ChineseBERTExtractor, I3DExtractor, XLSRExtractor, write_feature_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="modality", required=True)
    for modality in ("visual", "question", "audio"):
        child = subparsers.add_parser(modality)
        child.add_argument("--checkpoint", type=Path, required=True, help="Existing local checkpoint; no downloads")
        child.add_argument("--output", type=Path, required=True, help=".npy or .npz cache path")
        child.add_argument("--device", default="cpu")
        child.add_argument("--reference-output", type=Path, help="Optional JSON file with this modality's feature reference")
        if modality == "question":
            child.add_argument("--repository", type=Path, required=True, help="Local ShannonAI ChineseBERT source checkout")
            child.add_argument("--text", required=True)
            child.add_argument("--max-tokens", type=int, default=25)
        else:
            child.add_argument("--input", type=Path, required=True)
            child.add_argument("--ffmpeg", default="ffmpeg")
        if modality == "visual":
            child.add_argument("--output-layer", required=True, help="Verified layer exposed by local TorchScript export")
            child.add_argument("--max-clips", type=int, default=20)
        if modality == "audio":
            child.add_argument("--rnnoise", required=True, help="Local upstream rnnoise_demo executable")
    args = parser.parse_args()
    if args.modality == "visual":
        extractor = I3DExtractor(args.checkpoint, output_layer=args.output_layer, device=args.device)
        features, metadata = extractor.extract(args.input, ffmpeg=args.ffmpeg, max_clips=args.max_clips)
    elif args.modality == "audio":
        extractor = XLSRExtractor(args.checkpoint, device=args.device)
        features, metadata = extractor.extract(args.input, rnnoise=args.rnnoise, ffmpeg=args.ffmpeg)
    else:
        extractor = ChineseBERTExtractor(args.repository, args.checkpoint, max_tokens=args.max_tokens, device=args.device)
        features, metadata = extractor.extract(args.text)
    reference = {"features": {args.modality: write_feature_cache(args.output, features, metadata)}}
    encoded = json.dumps(reference, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.reference_output:
        args.reference_output.parent.mkdir(parents=True, exist_ok=True)
        args.reference_output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
