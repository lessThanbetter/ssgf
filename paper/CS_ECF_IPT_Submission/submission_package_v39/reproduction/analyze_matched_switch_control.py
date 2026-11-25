"""Compare CS-ECF with switch-frequency matched random allocations."""

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


FUSED_CANDIDATES = ("cdf", "pia", "spg")


def quantiles(values: np.ndarray) -> list[float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return [float(low), float(high)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=PACKAGE / "evidence" / "prospective",
    )
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=9413)
    parser.add_argument(
        "--out",
        type=Path,
        default=PACKAGE / "evidence" / "prospective"
        / "matched_switch_target_control.json",
    )
    args = parser.parse_args()

    sequences = ("050", "060", "070")
    models = tuple(prospective.MODEL_NAMES)
    rng = np.random.default_rng(args.seed)

    states = {}
    for sequence in sequences:
        score_dir = args.evidence_root / f"llvip{sequence}" / "scores"
        reference = np.load(
            score_dir / f"{models[0]}__vis.npz",
            allow_pickle=True,
        )
        stems = [str(value) for value in reference["stems"]]
        for model in models:
            states[(sequence, model)] = prospective._load_model(
                score_dir,
                model,
                stems,
            )

    blocks_by_checkpoint = {}
    visible_block_ap_by_checkpoint = {}
    csecf_gain = {}
    switched_indices = {}
    switched_candidates = {}
    decisions = {}
    for sequence in sequences:
        for model in models:
            state = states[(sequence, model)]
            selected = prospective._decisions(state)
            decisions[(sequence, model)] = selected
            blocks = prospective._blocks(state["vis"]["stems"])
            blocks_by_checkpoint[(sequence, model)] = blocks
            visible_block_ap = np.asarray([
                prospective._ap(state["vis"], block)
                for block in blocks
            ])
            visible_block_ap_by_checkpoint[(sequence, model)] = (
                visible_block_ap
            )
            csecf_gain[(sequence, model)] = float(np.mean([
                prospective._ap(
                    prospective._policy_stats(state, selected),
                    block,
                )
                - visible_block_ap[block_index]
                for block_index, block in enumerate(blocks)
            ]))
            switched_indices[(sequence, model)] = np.asarray(
                [index for index, name in enumerate(selected) if name != "vis"],
                dtype=np.int64,
            )
            switched_candidates[(sequence, model)] = [
                name for name in selected if name != "vis"
            ]

    checkpoint_names = [
        f"{sequence}/{model}"
        for model in models
        for sequence in sequences
    ]
    csecf_checkpoint_gain = np.asarray([
        np.mean([
            csecf_gain[(sequence, model)]
            for sequence in sequences
        ])
        for model in models
    ])
    random_checkpoint_gain = np.empty(
        (args.draws, len(models)),
        dtype=np.float64,
    )

    for draw in range(args.draws):
        for model_index, model in enumerate(models):
            sequence_gains = []
            for sequence in sequences:
                state = states[(sequence, model)]
                n_images = state["vis"]["n"]
                target_indices = switched_indices[(sequence, model)]
                candidate_pool = switched_candidates[(sequence, model)]

                sampled_indices = rng.choice(
                    n_images,
                    size=len(target_indices),
                    replace=False,
                )
                sampled_candidates = list(candidate_pool)
                rng.shuffle(sampled_candidates)
                selected = ["vis"] * n_images
                for image_index, candidate in zip(
                    sampled_indices,
                    sampled_candidates,
                ):
                    selected[int(image_index)] = candidate
                random_stats = prospective._policy_stats(state, selected)
                blocks = blocks_by_checkpoint[(sequence, model)]
                visible_block_ap = visible_block_ap_by_checkpoint[
                    (sequence, model)
                ]
                random_gain = float(np.mean([
                    prospective._ap(random_stats, block)
                    - visible_block_ap[block_index]
                    for block_index, block in enumerate(blocks)
                ]))
                sequence_gains.append(random_gain)
            random_checkpoint_gain[draw, model_index] = float(
                np.mean(sequence_gains)
            )

    random_policy_gain = random_checkpoint_gain.mean(axis=1)
    csecf_policy_gain = float(csecf_checkpoint_gain.mean())
    random_tail_gain = random_checkpoint_gain.min(axis=1)
    csecf_tail_gain = float(csecf_checkpoint_gain.min())

    matched_delta = csecf_policy_gain - random_policy_gain
    tail_improvement = csecf_tail_gain - random_tail_gain

    details = []
    for model in models:
        sequence_entries = []
        for sequence in sequences:
            selected = decisions[(sequence, model)]
            counts = {
                candidate: selected.count(candidate)
                for candidate in ("vis",) + FUSED_CANDIDATES
            }
            sequence_entries.append(
                {
                    "sequence": sequence,
                    "images": len(selected),
                    "switched": len(switched_indices[(sequence, model)]),
                    "selected_counts": counts,
                    "csecf_minus_visible": csecf_gain[(sequence, model)],
                }
            )
        details.append(
            {
                "model": model,
                "sequences": sequence_entries,
                "equal_sequence_csecf_minus_visible": float(
                    np.mean([
                        csecf_gain[(sequence, model)]
                        for sequence in sequences
                    ])
                ),
            }
        )

    payload = {
        "purpose": (
            "Test whether the observed CS-ECF gain is explained only by "
            "its lower switching frequency or candidate mix."
        ),
        "scope": "LLVIP confirmation sequences 050, 060, and 070",
        "draws": args.draws,
        "seed": args.seed,
        "matching_rule": (
            "Within each checkpoint and sequence, generate a random policy "
            "with the same number of switched images and the same multiset "
            "of CDF, PIAFusion, and SPGFusion labels as CS-ECF. Shuffle image "
            "positions and candidate labels independently from detector "
            "confidence."
        ),
        "estimand": (
            "Equal-group mean AP50 relative to visible, averaged equally "
            "within each sequence and then over the six final checkpoints."
        ),
        "inference_boundary": (
            "Conditional randomization summary over the observed three "
            "sequences and six checkpoints. It is not a cross-scene "
            "confidence guarantee."
        ),
        "checkpoint_names": checkpoint_names,
        "csecf_checkpoint_gain": csecf_checkpoint_gain.tolist(),
        "csecf_policy_gain": csecf_policy_gain,
        "matched_random_policy_gain": {
            "median": float(np.median(random_policy_gain)),
            "interval_95": quantiles(random_policy_gain),
            "all_draws": random_policy_gain.tolist(),
        },
        "csecf_minus_matched_random": {
            "point_difference_from_random_median": float(
                csecf_policy_gain - np.median(random_policy_gain)
            ),
            "interval_95": quantiles(matched_delta),
            "all_draws": matched_delta.tolist(),
        },
        "csecf_worst_checkpoint_gain": csecf_tail_gain,
        "matched_random_worst_checkpoint_gain": {
            "median": float(np.median(random_tail_gain)),
            "interval_95": quantiles(random_tail_gain),
            "all_draws": random_tail_gain.tolist(),
        },
        "tail_gain_improvement_over_matched_random": {
            "point_difference_from_random_median": float(
                csecf_tail_gain - np.median(random_tail_gain)
            ),
            "interval_95": quantiles(tail_improvement),
            "all_draws": tail_improvement.tolist(),
        },
        "checkpoint_details": details,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(
        {
            "csecf_policy_gain": payload["csecf_policy_gain"],
            "matched_random_median": payload[
                "matched_random_policy_gain"
            ]["median"],
            "csecf_minus_random_interval": payload[
                "csecf_minus_matched_random"
            ]["interval_95"],
            "csecf_worst_gain": payload["csecf_worst_checkpoint_gain"],
            "random_worst_median": payload[
                "matched_random_worst_checkpoint_gain"
            ]["median"],
            "tail_improvement_interval": payload[
                "tail_gain_improvement_over_matched_random"
            ]["interval_95"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
