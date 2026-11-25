"""Assemble provenance and source tables for PIA confirmation evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
PACKAGE = SCRIPT_PATH.parents[1]
REPO_ROOT = next(
    path
    for path in (SCRIPT_PATH.parent, *SCRIPT_PATH.parents)
    if (path / "data").is_dir()
)
SEQUENCES = ("080", "090", "100")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=REPO_ROOT / "data" / "prospective_pia_sources" / "manifest.json",
    )
    parser.add_argument(
        "--stems",
        type=Path,
        default=REPO_ROOT / "data" / "prospective_pia_sources" / "stems.npz",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=(
            REPO_ROOT
            / "results"
            / "prospective_pia"
            / "candidates"
            / "piafusion"
        ),
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=PACKAGE / "evidence" / "prospective_pia_confirmation",
    )
    args = parser.parse_args()

    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    if source.get("total_images") != 900:
        raise RuntimeError("source manifest does not contain 900 images")
    source_target = args.evidence / "source_manifest.json"
    source_target.write_text(
        json.dumps(source, indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(args.stems, args.evidence / "stems.npz")

    files = {}
    total_bytes = 0
    for sequence in SEQUENCES:
        stems = source["sequence_stems"][sequence]
        for stem in stems:
            path = args.candidates / f"{stem}.png"
            if not path.is_file():
                raise FileNotFoundError(path)
            total_bytes += path.stat().st_size
            files[path.name] = {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
    (args.evidence / "candidate_manifest.json").write_text(
        json.dumps(
            {
                "candidate": "PIAFusion",
                "count": len(files),
                "total_bytes": total_bytes,
                "source_manifest_sha256": sha256(source_target),
                "files": files,
                "note": (
                    "Candidate images are not redistributed. Their per-file "
                    "hashes are retained here."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    analysis = json.loads(
        (args.evidence / "analysis.json").read_text(encoding="utf-8")
    )
    rows = []
    for sequence, value in analysis["primary"]["per_sequence"].items():
        rows.append(
            {
                "scope": "sequence",
                "model": "equal-checkpoint",
                "sequence": sequence,
                "ap50_gain": value,
                "ap50_95_gain": "",
                "switch_fraction": "",
            }
        )
    for model, value in analysis["primary"]["per_checkpoint"].items():
        rows.append(
            {
                "scope": "checkpoint",
                "model": model,
                "sequence": "equal-sequence",
                "ap50_gain": value,
                "ap50_95_gain": "",
                "switch_fraction": "",
            }
        )
    for model, sequence_rows in analysis["runs"].items():
        for sequence, value in sequence_rows.items():
            rows.append(
                {
                    "scope": "cell",
                    "model": model,
                    "sequence": sequence,
                    "ap50_gain": value["point50"],
                    "ap50_95_gain": value["point95"],
                    "switch_fraction": value["switch_fraction"],
                }
            )
    path = PACKAGE / "source_data" / "TablePIAConfirmation_source.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {args.evidence}")
    print(f"WROTE {path}")


if __name__ == "__main__":
    main()
