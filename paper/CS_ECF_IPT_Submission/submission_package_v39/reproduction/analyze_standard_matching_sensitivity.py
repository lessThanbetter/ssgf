"""Analyze the standard-matching LLVIP confirmation scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
sys.path.insert(0, str(HERE))

import prospective_llvip_analysis as prospective  # noqa: E402
from local_ap_reference import pooled  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--score-root",
        type=Path,
        default=PACKAGE / "evidence" / "prospective_standard_matching",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"D:\Code\_server_handoff_v2\ssgf\data"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PACKAGE / "evidence"
        / "prospective_standard_matching"
        / "analysis.json",
    )
    args = parser.parse_args()

    sequences = ("050", "060", "070")
    models = tuple(prospective.MODEL_NAMES)
    selection_rows = {}
    blocks_by_model = {}
    for sequence in sequences:
        source_dir = args.data_root / f"erf_holdout_llvip{sequence}"
        stems = sorted(
            path.stem for path in (source_dir / "visible").glob("*.jpg")
        )
        score_dir = args.score_root / f"llvip{sequence}" / "scores"
        selection_rows[sequence] = {}
        blocks_by_model[sequence] = {}
        for model in models:
            states = prospective._load_model(score_dir, model, stems)
            blocks = prospective._blocks(stems)
            selection_rows[sequence][model] = prospective._checkpoint_rows(
                states,
                blocks,
            )
            blocks_by_model[sequence][model] = blocks

    confirmation_rows = {
        model: {
            sequence: selection_rows[sequence][model]
            for sequence in sequences
        }
        for model in models
    }
    frozen_summary = prospective._summarize(
        confirmation_rows,
        sequences,
    )
    interval = prospective._bootstrap_interval(confirmation_rows)

    comparison_states = {
        sequence: {
            model: selection_rows[sequence][model]["states"]
            for model in models
        }
        for sequence in sequences
    }
    comparison_blocks = {
        sequence: {
            model: blocks_by_model[sequence][model]
            for model in models
        }
        for sequence in sequences
    }
    comparisons = prospective._comparison_payload(
        comparison_states,
        comparison_blocks,
        {
            "csecf_minus_visible": ("csecf", "visible"),
            "cdf_minus_visible": ("cdf", "visible"),
            "csecf_minus_cdf": ("csecf", "cdf"),
            "no_shield_minus_visible": ("no_shield", "visible"),
            "csecf_minus_no_shield": ("csecf", "no_shield"),
        },
        models,
    )

    ap50_95 = {}
    policy_block_ap95 = {}
    for sequence in sequences:
        for model in models:
            states = selection_rows[sequence][model]["states"]
            n_images = states["vis"]["n"]
            choices = {
                "visible": ["vis"] * n_images,
                "always_cdf": ["cdf"] * n_images,
                "no_shield": prospective._decisions(states, margin=0.0),
                "csecf": prospective._decisions(states),
            }
            for name, selected in choices.items():
                stats = prospective._policy_stats(states, selected)
                policy_block_ap95[(sequence, model, name)] = np.asarray([
                    pooled(stats, block)[1]
                    for block in blocks_by_model[sequence][model]
                ])

    for left, right in (
        ("csecf", "visible"),
        ("always_cdf", "visible"),
        ("csecf", "always_cdf"),
        ("no_shield", "visible"),
        ("csecf", "no_shield"),
    ):
        rows = {}
        per_checkpoint = {}
        for model in models:
            rows[model] = {}
            for sequence in sequences:
                block_gains = (
                    policy_block_ap95[(sequence, model, left)]
                    - policy_block_ap95[(sequence, model, right)]
                )
                rows[model][sequence] = {
                    "point": float(block_gains.mean()),
                    "block_gains": block_gains,
                }
            per_checkpoint[model] = float(np.mean([
                rows[model][sequence]["point"]
                for sequence in sequences
            ]))
        ap50_95[f"{left}_minus_{right}"] = {
            "equal_checkpoint_mean": float(np.mean(
                list(per_checkpoint.values())
            )),
            "conditional_sequence_by_run_interval": (
                prospective._bootstrap_interval(
                    rows,
                    models,
                )
            ),
            "positive_checkpoints": int(sum(
                value > 0 for value in per_checkpoint.values()
            )),
            "per_checkpoint": per_checkpoint,
        }

    payload = {
        "protocol": {
            "matching": "confidence-ordered one-to-one matching",
            "sequences": list(sequences),
            "models": list(models),
            "top_k": prospective.TOP_K,
            "frozen_margin": prospective.FROZEN_MARGIN,
            "block_size_frames": prospective.BLOCK_SIZE,
            "bootstrap_draws": prospective.BOOTSTRAP_DRAWS,
            "bootstrap_seed": prospective.BOOTSTRAP_SEED,
        },
        "confirmation": {
            "frozen_policy": frozen_summary,
            "conditional_sequence_by_run_interval": interval,
            "comparisons": comparisons,
            "ap50_95_descriptive": ap50_95,
        },
        "purpose": (
            "Sensitivity analysis of the prospective LLVIP conclusion to "
            "the matching rule. The primary score arrays use all-at-once "
            "maximum-cardinality matching; these arrays use the conventional "
            "confidence-ordered one-to-one matching."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "frozen_policy": frozen_summary,
        "comparisons": {
            key: {
                "point": value["summary"]["sequence_equal_point"],
                "interval": value["conditional_sequence_by_run_interval"],
            }
            for key, value in comparisons.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
