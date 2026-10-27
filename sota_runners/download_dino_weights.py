"""Download the DINO ViT-B/16 checkpoint required by SPGFusion."""

from __future__ import annotations

import argparse
from pathlib import Path

from sota_runners.download_spgfusion_weights import parallel_download


DINO_URL = (
    "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/"
    "dino_vitbase16_pretrain.pth"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(
            Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
            / "dino_vitbase16_pretrain.pth"
        ),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    parallel_download(
        DINO_URL,
        None,
        Path(args.output),
        workers=args.workers,
        attempts=args.attempts,
        force=args.force,
        label="dino-download",
    )


if __name__ == "__main__":
    main()
