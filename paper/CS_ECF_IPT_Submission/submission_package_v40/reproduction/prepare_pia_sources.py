"""Assemble the fixed LLVIP PIA-confirmation source set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np


DATA_ROOT = Path(r"D:\Code\_server_handoff_v2\ssgf\data")
SEQUENCES = ("080", "090", "100")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_ROOT / "prospective_pia_sources",
    )
    args = parser.parse_args()

    all_stems = []
    sequence_stems = {}
    file_manifest = {}
    for sequence in SEQUENCES:
        source = args.data_root / f"prospective_pia_{sequence}"
        manifest = json.loads(
            (source / "manifest.json").read_text(encoding="utf-8")
        )
        stems = [str(stem) for stem in manifest["stems"]]
        if len(stems) != 300:
            raise RuntimeError(f"{sequence}: expected 300 stems")
        sequence_stems[sequence] = stems
        for stem in stems:
            all_stems.append(stem)
            for modality, suffix in (
                ("visible", ".jpg"),
                ("infrared", ".jpg"),
                ("labels", ".txt"),
            ):
                input_path = source / modality / f"{stem}{suffix}"
                if not input_path.is_file() or input_path.stat().st_size == 0:
                    raise FileNotFoundError(input_path)
                output_path = args.out / modality / input_path.name
                link_or_copy(input_path, output_path)
                if modality != "labels":
                    file_manifest[
                        output_path.relative_to(args.out).as_posix()
                    ] = sha256(input_path)

    if len(all_stems) != len(set(all_stems)):
        raise RuntimeError("duplicate stems across confirmation sequences")
    np.savez_compressed(
        args.out / "stems.npz",
        stems=np.asarray(all_stems),
    )
    payload = {
        "sequences": list(SEQUENCES),
        "limit_per_sequence": 300,
        "total_images": len(all_stems),
        "stem_order": all_stems,
        "sequence_stems": sequence_stems,
        "source_file_sha256": file_manifest,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"WROTE {args.out}: {len(all_stems)} pairs, "
        f"{len(file_manifest)} image hashes"
    )


if __name__ == "__main__":
    main()
