"""Assemble the canonical VEDAI external-domain evidence record."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.png")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(r"D:\Code\_server_handoff_v2\ssgf\results\vedai_20261029"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(r"D:\Code\_ssgf_external\vedai_fold01"),
    )
    args = parser.parse_args()
    results = args.results.resolve()
    dataset = args.dataset.resolve()

    analysis = load(results / "analysis" / "vedai_native_analysis.json")
    dataset_manifest = load(dataset / "dataset_manifest.json")
    run_manifests = {}
    score_files = {}
    for run in ("vedai_run1", "vedai_run2"):
        manifest = load(results / f"{run}_manifest.json")
        manifest["last_sha256"] = sha256(Path(manifest["last_weights"]))
        manifest["packaged_weights"] = (
            f"weights/{run}/last.pt"
        )
        run_manifests[run] = manifest
        score_files[run] = {
            "summary": load(results / "test_scores" / f"{run}_scores.json"),
            "score_sha256": {
                path.name: sha256(path)
                for path in sorted(
                    (results / "test_scores").glob(f"{run}__*.npz")
                )
            },
        }

    payload = {
        "schema_version": "1.0",
        "recorded": "2026-10-29",
        "protocol": {
            "path": "VEDAI_CONFIRMATION.md",
            "sha256": sha256(results / "protocol" / "VEDAI_CONFIRMATION.md"),
            "externally_registered": False,
        },
        "dataset": {
            "name": "VEDAI 512",
            "provider": "https://downloads.greyc.fr/vedai/",
            "terms": (
                "https://downloads.greyc.fr/vedai/"
                "TermsandConditionsofUseVeDAI2014.pdf"
            ),
            "use_boundary": "research and non-commercial use only",
            "manifest": dataset_manifest,
        },
        "models": run_manifests,
        "candidate_generation": {
            "source_tree_sha256": {
                "visible": tree_sha256(dataset / "sources" / "test" / "visible"),
                "infrared": tree_sha256(dataset / "sources" / "test" / "infrared"),
            },
            "candidate_tree_sha256": {
                "cdf": tree_sha256(
                    results
                    / "test_candidates"
                    / "erf"
                    / "consensus_intersection_pixel_mean_ifcnn_q095_a020_s05"
                ),
                "piafusion": tree_sha256(
                    results / "test_candidates" / "piafusion"
                ),
                "spgfusion": tree_sha256(
                    results / "test_candidates" / "spgfusion"
                ),
            },
            "source_geometry": [512, 512],
            "source_count": 121,
            "component_sha256": {
                "ifcnn": sha256(
                    Path(r"D:\Code\_server_handoff_v2\ssgf\data\btl\ifcnn.pt")
                ),
                "piafusion": sha256(
                    Path(
                        r"D:\Code\_ssgf_external\PIAFusion_pytorch"
                        r"\pretrained\fusion_model_epoch_29.pth"
                    )
                ),
                "piafusion_illumination": sha256(
                    Path(
                        r"D:\Code\_ssgf_external\PIAFusion_pytorch"
                        r"\pretrained\best_cls.pth"
                    )
                ),
                "spgfusion": sha256(
                    Path(
                        r"D:\Code\_ssgf_external\SPGFusion"
                        r"\CHECKPOINT\best_model_1.pth"
                    )
                ),
                "spgfusion_clip": sha256(
                    Path(
                        r"D:\Code\_ssgf_external\SPGFusion"
                        r"\model\ViT-B-16-laion2B"
                        r"\open_clip_pytorch_model.bin"
                    )
                ),
                "detector_initialization": sha256(
                    Path(r"D:\Code\_server_handoff_v2\ssgf\yolo11n-obb.pt")
                ),
            },
        },
        "scores": score_files,
        "analysis": analysis,
    }
    out = results / "evidence" / "vedai_evidence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
