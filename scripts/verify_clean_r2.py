"""Validate the independent-test artifacts for Reviewer 2, comment 2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FUSED = {
    "mean",
    "activity_only",
    "tophat",
    "probe",
    "tophat_permuted",
    "probe_permuted",
    "tophat_wshuffle",
    "probe_wshuffle",
    "ensemble",
    "ifcnn",
    "densefuse",
    "csf",
    "piafusion",
    "spgfusion",
}
EVALUATORS = {
    "yolo11n_visible_clean",
    "yolo26n_visible_clean",
    "rtdetr_l_visible_clean",
    "yolo11n_modality_agnostic_clean",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    result_root = root / "results/clean_r2"
    deposit_data = (
        root
        / "_deposit/ssgf-llvip-fusion-detection-v1/data"
    )
    split_manifest = read_json(
        root
        / "_deposit/ssgf-llvip-fusion-detection-v1/data/splits/clean_split_manifest.json"
    )
    if split_manifest["counts"] != {
        "selected": 1500,
        "train": 1350,
        "validation": 150,
        "test": 300,
    }:
        raise SystemExit("unexpected stem-manifest counts")
    if any(split_manifest["overlap"].values()):
        raise SystemExit(f"non-independent split: {split_manifest['overlap']}")
    if split_manifest["label_counts"] != {
        "train": 4479,
        "validation": 419,
        "test": 744,
    }:
        raise SystemExit("unexpected stem-manifest label counts")

    manifest = read_json(result_root / "clean_r2_full.json")
    counts = manifest["protocol"]["split"]
    if counts != {"train": 1350, "validation": 150, "test": 300}:
        raise SystemExit(f"unexpected split counts: {counts}")
    if manifest["protocol"]["test_labels"] != 744:
        raise SystemExit("unexpected test-label count")
    if set(manifest["evaluators"]) != EVALUATORS:
        raise SystemExit("missing clean evaluator")
    for evaluator in EVALUATORS:
        names = set(manifest["evaluators"][evaluator]["inputs"])
        if names != FUSED:
            raise SystemExit(f"incomplete fused inputs for {evaluator}")
    if len(manifest["rank_associations"]) != 6:
        raise SystemExit("expected six pairwise clean-evaluator rank associations")

    for detector in ("yolo26n_clean", "rtdetr_l_clean"):
        score_root = result_root / detector
        scores = read_json(score_root / "scores.json")
        multiplicity = read_json(score_root / "multiplicity.json")
        if set(scores) != {detector}:
            raise SystemExit(f"unexpected score payload for {detector}")
        if set(scores[detector]) != FUSED | {"vis", "ir"}:
            raise SystemExit(f"incomplete score inputs for {detector}")
        if multiplicity[detector]["n_tests"] != 14:
            raise SystemExit(f"unexpected multiplicity family for {detector}")
        if len(list((score_root / "per_image").glob("*.npz"))) != 16:
            raise SystemExit(f"incomplete per-image output for {detector}")
        for source in (score_root / "per_image").glob("*.npz"):
            archived = deposit_data / "per_image" / source.name
            if not archived.is_file() or sha256(source) != sha256(archived):
                raise SystemExit(f"archived per-image mismatch: {source.name}")

    for name in (
        "yolo11n_clean",
        "yolo26n_clean",
        "rtdetr_l_clean",
    ):
        regime = read_json(result_root / f"regime_bh_{name}.json")
        if len(regime["tests"]) != 84:
            raise SystemExit(f"unexpected regime family for {name}")

    for name in ("yolo26n_clean", "rtdetr_l_clean"):
        training = read_json(result_root / f"{name}_training.json")
        if training["epochs_completed"] != 40:
            raise SystemExit(f"incomplete training for {name}")
        if not training["checkpoints"]["best.pt"]["sha256"]:
            raise SystemExit(f"missing checkpoint hash for {name}")

    for source_name, archived_name in (
        ("clean_r2_full.json", "clean_r2_full.json"),
        ("full_16/multiplicity.json", "clean_r2_multiplicity.json"),
        ("clean_r2_correlations.json", "clean_r2_correlations.json"),
        ("yolo26n_clean_training.json", "clean_r2_yolo26n_training.json"),
        ("rtdetr_l_clean_training.json", "clean_r2_rtdetr_l_training.json"),
    ):
        source = result_root / source_name
        archived = deposit_data / "aggregate" / archived_name
        if not archived.is_file() or sha256(source) != sha256(archived):
            raise SystemExit(f"archived aggregate mismatch: {source_name}")

    print("CLEAN_R2_OK")


if __name__ == "__main__":
    main()
