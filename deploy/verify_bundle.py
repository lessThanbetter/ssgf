"""Check that the new-server handoff contains the files needed to continue."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path


REQUIRED = [
    "analysis/vif_reference_rescore.py",
    "analysis/run_reference_vif_parallel.py",
    "analysis/extract_llvip_sources.py",
    "analysis/estimate_alignment.py",
    "analysis/alignment_sensitivity.py",
    "analysis/convention_and_multiplicity.py",
    "analysis/regime_bh.py",
    "scripts/verify_clean_r2.py",
    "scripts/score_clean_evaluators.py",
    "scripts/prepare_clean_evaluator_dataset.py",
    "scripts/merge_clean_score_runs.py",
    "scripts/summarize_clean_r2.py",
    "scripts/compute_clean_correlations.py",
    "scripts/framework_figure_utils.py",
    "scripts/make_fig11_decision_v2.py",
    "scripts/make_framework_v7.py",
    "scripts/make_framework_v7_maps.py",
    "scripts/make_figs2_10_v2.py",
    "results/reference_vif/llvip_14_full.json",
    "results/reference_vif/tno_8_full.json",
    "results/reference_vif/msrs_8_full.json",
    "results/reference_vif/roadscene_8_full.json",
    "results/alignment/alignment.json",
    "results/alignment/alignment_sensitivity_clean_ranges.json",
    "sota_runners/generate_fusion_outputs.py",
    "sota_runners/run_spgfusion.py",
    "deploy/EXTERNAL_ASSETS.md",
    "requirements-experiments.txt",
    "data/msrs_ae.pt",
    "data/msrs_probe.pt",
    "data/msrs_csf.pt",
    "data/det_runs/llvip/weights/best.pt",
    "data/det_agnostic/agn/weights/best.pt",
    "data/det_clean/yolo26n/weights/best.pt",
    "data/det_clean/rtdetr_l/weights/best.pt",
    "results/clean_r2/README.md",
    "results/clean_r2/clean_r2_full.json",
    "results/clean_r2/clean_r2_correlations.json",
    "results/clean_r2/full_16/scores.json",
    "results/clean_r2/full_16/multiplicity.json",
    "results/clean_r2/regime_bh_yolo11n_clean.json",
    "results/clean_r2/regime_bh_yolo26n_clean.json",
    "results/clean_r2/regime_bh_rtdetr_l_clean.json",
    "_deposit/ssgf-llvip-fusion-detection-v1/data/splits/clean_split_manifest.json",
    "_deposit/ssgf-llvip-fusion-detection-v1/data/aggregate/clean_r2_full.json",
    "_deposit/ssgf-llvip-fusion-detection-v1/data/aggregate/clean_r2_multiplicity.json",
    "_deposit/ssgf-llvip-fusion-detection-v1/data/aggregate/clean_r2_correlations.json",
    "_deposit/ssgf-llvip-fusion-detection-v1/data/aggregate/clean_r2_yolo26n_training.json",
    "_deposit/ssgf-llvip-fusion-detection-v1/data/aggregate/clean_r2_rtdetr_l_training.json",
    "_deposit/ssgf-llvip-fusion-detection-v1/data/aggregate/regime_bh_yolo11n_clean.json",
    "_deposit/ssgf-llvip-fusion-detection-v1/data/aggregate/regime_bh_yolo26n_clean.json",
    "_deposit/ssgf-llvip-fusion-detection-v1/data/aggregate/regime_bh_rtdetr_l_clean.json",
    "_deposit/ssgf-llvip-fusion-detection-v1/data/aggregate/clean_split_seeds.json",
    "_deposit/ssgf-llvip-fusion-detection-v1/data/aggregate/clean_split_seed_ranges.json",
    "_deposit/ssgf-llvip-fusion-detection-v1/data/aggregate/convention_multiplicity.json",
    "_deposit/ssgf-llvip-fusion-detection-v1/data/aggregate/regime_bh_yolo11n.json",
    "_deposit/ssgf-llvip-fusion-detection-v1/data/aggregate/regime_bh_yolo26n.json",
    "_deposit/ssgf-llvip-fusion-detection-v1/data/aggregate/regime_bh_rtdetr_l.json",
    "_deposit/ssgf-llvip-fusion-detection-v1/data/aggregate/inference_cost.json",
    "_clean_split_seed_ranges.py",
    "_r5_seeds.py",
    "paper/PAPER_IPT_v3.pdf",
    "paper/PAPER_MEASUREMENT_v4.pdf",
    "paper/Supplementary_Material_IPT.pdf",
    "paper/Supplementary_Material_MEASUREMENT.pdf",
    "paper/fig_decision_v2.pdf",
    "paper/fig_decision_v2.svg",
    "paper/fig_decision_v2.png",
    "paper/fig_decision_v2_QA.md",
    "paper/figs2_10_v2_QA.md",
    "paper/LAYOUT_AUDIT.md",
    "paper/framework_v7.pdf",
    "paper/framework_v7.svg",
    "paper/framework_v7.png",
    "paper/framework_v7_QA.md",
    "paper/framework_v7_maps.npz",
    "paper/fig_claim_v2.pdf",
    "paper/fig_claim_v2.svg",
    "paper/fig_regime_v2.pdf",
    "paper/fig_regime_v2.svg",
    "paper/fig_robust_v2.pdf",
    "paper/fig_robust_v2.svg",
    "paper/fig_corr_v2.pdf",
    "paper/fig_corr_v2.svg",
    "paper/fig_criterion_v2.pdf",
    "paper/fig_criterion_v2.svg",
    "paper/fig_operating_v2.pdf",
    "paper/fig_operating_v2.svg",
    "paper/fig_tau_v2.pdf",
    "paper/fig_tau_v2.svg",
    "paper/fig_scale_v2.pdf",
    "paper/fig_scale_v2.svg",
    "paper/fig_corrupt_v2.pdf",
    "paper/fig_corrupt_v2.svg",
    "paper/Measurement_Figures_SVG.zip",
    "paper/HIGHLIGHTS_MEASUREMENT.docx",
    "paper/HIGHLIGHTS_MEASUREMENT.txt",
    "paper/COVER_LETTER_MEASUREMENT.pdf",
    "paper/MEASUREMENT_SUBMISSION_CHECKLIST.md",
    "paper/measurement_submission/README.md",
    "paper/measurement_submission/Manuscript.pdf",
    "paper/measurement_submission/Supplementary_Material.pdf",
    "paper/measurement_submission/Highlights.docx",
    "paper/measurement_submission/Highlights.txt",
    "paper/measurement_submission/Cover_Letter.pdf",
    "paper/measurement_submission/Submission_Checklist.md",
    "paper/measurement_submission/Figures_SVG.zip",
    "paper/Measurement_Submission.zip",
]

SKIP_SUFFIXES = {
    ".aux",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
    ".spl",
    ".synctex",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_manifest(
    root: Path,
    destination: Path,
    skip_names: set[str],
    skip_prefixes: tuple[str, ...] = (),
) -> int:
    paths = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        relative_text = relative.as_posix()
        if (
            path.name in skip_names
            or "__pycache__" in path.parts
            or path.suffix == ".pyc"
            or path.suffix in SKIP_SUFFIXES
            or path.name.endswith(".synctex.gz")
            or any(relative_text.startswith(prefix) for prefix in skip_prefixes)
            or "/.git/" in f"/{relative_text}/"
        ):
            continue
        paths.append((relative, path))

    with destination.open("w", encoding="utf-8", newline="\n") as out:
        for relative, path in sorted(paths, key=lambda item: item[0].as_posix()):
            out.write(f"{sha256(path)}  {relative.as_posix()}\n")
    return len(paths)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--write-deposit-manifest", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()

    missing = [p for p in REQUIRED if not (root / p).is_file()]
    if missing:
        raise SystemExit("missing files:\n" + "\n".join(missing))

    aggregate = root / "_deposit/ssgf-llvip-fusion-detection-v1/data/aggregate"
    per_image = root / "_deposit/ssgf-llvip-fusion-detection-v1/data/per_image"
    payload = {
        "aggregate_json": len(list(aggregate.glob("*.json"))),
        "per_image_npz": len(list(per_image.glob("*.npz"))),
        "required_files": len(REQUIRED),
    }
    print(json.dumps(payload, indent=2))
    if payload["per_image_npz"] < 300:
        raise SystemExit("per-image archive is incomplete")

    for name in ("numpy", "scipy", "PIL", "torch"):
        module = importlib.import_module(name)
        print("%-8s %s" % (name, getattr(module, "__version__", "ok")))

    # The deposit manifest is part of the top-level manifest. Regenerate it first
    # whenever both manifests are requested, otherwise the top-level checksum records
    # the pre-update deposit manifest.
    if args.write_deposit_manifest:
        deposit = root / "_deposit/ssgf-llvip-fusion-detection-v1"
        deposit_manifest = deposit / "MANIFEST.sha256"
        count = write_manifest(
            deposit,
            deposit_manifest,
            {"MANIFEST.sha256"},
            ("data/fused_images/",),
        )
        print("deposit_manifest", deposit_manifest, count)

    manifest = root / "SHA256SUMS"
    if args.write_manifest:
        count = write_manifest(
            root,
            manifest,
            {"SHA256SUMS"},
            (
                "results/clean_r2/smoke/",
                "results/clean_r2/smoke_fast/",
                "paper/_generated_clean_r2/",
            ),
        )
        print("manifest", manifest, count)
    elif not args.write_deposit_manifest and manifest.is_file():
        checked = 0
        problems = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, relative = line.split("  ", 1)
            path = root / relative
            if not path.is_file():
                problems.append(f"missing {relative}")
            elif sha256(path) != expected:
                problems.append(f"checksum {relative}")
            checked += 1
        if problems:
            raise SystemExit("manifest problems:\n" + "\n".join(problems))
        print("manifest_checked", checked)

    print("BUNDLE_OK", root)


if __name__ == "__main__":
    main()
