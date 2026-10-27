"""Download the OpenCLIP ViT-B/16 checkpoint required by SPGFusion."""

from __future__ import annotations

import argparse
from pathlib import Path

from sota_runners.download_spgfusion_weights import parallel_download


CLIP_URL = (
    "https://hf-mirror.com/laion/"
    "CLIP-ViT-B-16-laion2B-s34B-b88K/resolve/main/"
    "open_clip_pytorch_model.bin"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=(
            "sota/SPGFusion/model/ViT-B-16-laion2B/"
            "open_clip_pytorch_model.bin"
        ),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    parallel_download(
        CLIP_URL,
        None,
        Path(args.output),
        workers=args.workers,
        attempts=args.attempts,
        force=args.force,
        label="clip-download",
    )


if __name__ == "__main__":
    main()
