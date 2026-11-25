"""Package the released per-image external-evaluator score arrays."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
SOURCE_ROOT = (
    PACKAGE.parents[2]
    / "results"
    / "independent_evaluators"
)
OUTPUT_ROOT = PACKAGE / "evidence" / "external_evaluator_scores"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    source_groups = {
        "yolov9t_coco_and_yolov10s_coco": SOURCE_ROOT / "per_image",
        "yolov9t_llvip_trained": (
            SOURCE_ROOT / "domain_yolov9t_v3" / "per_image"
        ),
    }
    output_per_image = OUTPUT_ROOT / "per_image"
    output_per_image.mkdir(parents=True, exist_ok=True)

    manifest = {
        "purpose": (
            "Per-image score arrays used for external-evaluator switching, "
            "calibration, and equal-scene analyses."
        ),
        "matching_boundary": (
            "These arrays retain the study's released TP flags. Raw detection "
            "geometry is not present in every array, so standard-matching "
            "rescoring requires detector inference."
        ),
        "groups": {},
    }
    for group_name, source_dir in source_groups.items():
        if not source_dir.is_dir():
            raise FileNotFoundError(source_dir)
        entries = []
        for source in sorted(source_dir.glob("*.npz")):
            target = output_per_image / source.name
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
            entries.append(
                {
                    "file": f"per_image/{source.name}",
                    "bytes": target.stat().st_size,
                    "sha256": sha256(target),
                }
            )
        manifest["groups"][group_name] = entries

    manifest_path = PACKAGE / "evidence" / "external_evaluator_scores.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "manifest": str(manifest_path),
        "files": sum(len(entries) for entries in manifest["groups"].values()),
    }, indent=2))


if __name__ == "__main__":
    main()
