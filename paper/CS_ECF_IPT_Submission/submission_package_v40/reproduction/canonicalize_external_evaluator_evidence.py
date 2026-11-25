"""Label pooled diagnostics distinctly and add canonical equal-scene contrasts."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
EVIDENCE = PACKAGE / "evidence"
DIAGNOSTICS = EVIDENCE / "major_revision" / "external_evaluator_diagnostics.json"
TARGETS = (
    EVIDENCE / "external_evaluator_policy_summary.json",
    EVIDENCE / "domain_yolov9t_policy_summary.json",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    diagnostics = load(DIAGNOSTICS)["models"]
    for path in TARGETS:
        payload = load(path)
        payload["estimand_note"] = (
            "Pooled-image diagnostics are retained under pooled_ keys. "
            "Scene-equal-weight contrasts are the estimates used in the "
            "manuscript."
        )
        for model_name, row in payload["models"].items():
            canonical = diagnostics[model_name]["scene_equal_weight"]
            if "delta" in row:
                row["pooled_image_delta"] = row.pop("delta")
            if "image_ci95" in row:
                row["pooled_image_ci95"] = row.pop("image_ci95")
            if "group_ci95" in row:
                row["preliminary_group_ci95_for_pooled_point"] = row.pop(
                    "group_ci95"
                )
            row["equal_scene_delta"] = canonical
        write(path, payload)
        print("UPDATED", path)


if __name__ == "__main__":
    main()
