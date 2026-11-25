"""Evaluate the archived LLVIP checkpoints on unseen LLVIP sequences."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prospective_llvip_analysis as analysis  # noqa: E402


MODELS = (
    "orig_yolo11",
    "orig_agnostic",
    "orig_yolo26_s0",
    "orig_yolo26_s1",
    "orig_yolo26_s2",
    "orig_yolo8_s0",
    "orig_yolo8_s1",
    "orig_rtdetr_l",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            os.environ.get(
                "SSGF_DATA_ROOT",
                r"D:\Code\_server_handoff_v2\ssgf\data",
            )
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    data_root = args.data_root.resolve()

    analysis.MODEL_NAMES = MODELS
    rows = {}
    for sequence in analysis.CONFIRM_SEQUENCES:
        source = (
            data_root
            / f"erf_holdout_llvip{sequence}"
        )
        stems = sorted(
            path.stem for path in (source / "visible").glob("*.jpg")
        )
        score_dir = root / f"llvip{sequence}" / "scores_original8"
        rows[sequence] = {}
        for model in MODELS:
            states = analysis._load_model(score_dir, model, stems)
            rows[sequence][model] = analysis._selection_record(
                states,
                analysis._blocks(stems),
                analysis._decisions(states),
            )

    by_model = {
        model: {
            sequence: rows[sequence][model]
            for sequence in analysis.CONFIRM_SEQUENCES
        }
        for model in MODELS
    }
    summary = analysis._summarize(
        by_model,
        analysis.CONFIRM_SEQUENCES,
        MODELS,
    )
    interval = analysis._bootstrap_interval(by_model, MODELS)
    comparison_states = {
        sequence: {
            model: rows[sequence][model]["states"]
            for model in MODELS
        }
        for sequence in analysis.CONFIRM_SEQUENCES
    }
    comparison_blocks = {
        sequence: {
            model: analysis._blocks([
                stem
                for stem in rows[sequence][model]["states"]["vis"]["stems"]
            ])
            for model in MODELS
        }
        for sequence in analysis.CONFIRM_SEQUENCES
    }
    comparisons = analysis._comparison_payload(
        comparison_states,
        comparison_blocks,
        {
            "csecf_minus_visible": ("csecf", "visible"),
            "cdf_minus_visible": ("cdf", "visible"),
            "no_shield_minus_visible": ("no_shield", "visible"),
            "csecf_minus_cdf": ("csecf", "cdf"),
            "csecf_minus_no_shield": ("csecf", "no_shield"),
        },
        MODELS,
    )
    payload = {
        "checkpoints": list(MODELS),
        "confirmation": {
            "frozen_policy": summary,
            "conditional_sequence_by_run_interval": interval,
            "comparisons": comparisons,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
