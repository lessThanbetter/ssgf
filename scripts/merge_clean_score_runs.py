"""Merge independently scored clean-split detectors into one result directory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    destination_images = output / "per_image"
    destination_images.mkdir(parents=True, exist_ok=True)

    scores: dict[str, dict] = {}
    multiplicity: dict[str, dict] = {}
    reference: tuple[np.ndarray, np.ndarray] | None = None
    copied = 0
    for score_root in map(Path, args.scores):
        score_file = score_root / "scores.json"
        multiplicity_file = score_root / "multiplicity.json"
        if not score_file.is_file() or not multiplicity_file.is_file():
            raise SystemExit(f"incomplete score directory: {score_root}")
        scores.update(read_json(score_file))
        multiplicity.update(read_json(multiplicity_file))

        files = sorted((score_root / "per_image").glob("*.npz"))
        if len(files) != 16:
            raise SystemExit(f"expected 16 per-image files in {score_root}, got {len(files)}")
        for source in files:
            with np.load(source) as current:
                signature = (
                    np.asarray(current["stems"]),
                    np.asarray(current["n_gt"]),
                )
            if reference is None:
                reference = signature
            elif not (
                np.array_equal(reference[0], signature[0])
                and np.array_equal(reference[1], signature[1])
            ):
                raise SystemExit(f"image or label order mismatch in {source}")
            destination = destination_images / source.name
            if destination.exists():
                raise SystemExit(f"duplicate per-image result: {destination.name}")
            shutil.copy2(source, destination)
            copied += 1

    (output / "scores.json").write_text(
        json.dumps(scores, indent=2),
        encoding="utf-8",
    )
    (output / "multiplicity.json").write_text(
        json.dumps(multiplicity, indent=2),
        encoding="utf-8",
    )
    print(
        f"detectors={sorted(scores)} per_image_files={copied} "
        f"n_images={len(reference[0]) if reference else 0} "
        f"n_gt={int(reference[1].sum()) if reference else 0}"
    )


if __name__ == "__main__":
    main()
