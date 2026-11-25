"""Analyze the fixed visible-plus-PIAFusion temporal holdout."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from local_ap_reference import load, pooled


SCRIPT_PATH = Path(__file__).resolve()
PACKAGE = SCRIPT_PATH.parents[1]
MODELS = (
    "yolo26_s0",
    "yolo26_s1",
    "yolo26_s2",
    "yolo8_s0",
    "yolo8_s1",
    "rtdetr_l",
)
SEQUENCES = ("080", "090", "100")
TOP_K = 5
MARGIN = 1.5
BLOCK_SIZE = 100
DRAWS = 10000
SEED = 26011


def topk(values: np.ndarray) -> float:
    if len(values) == 0:
        return -1.0
    return float(np.sort(values)[-TOP_K:].mean())


def choices(states: dict[str, dict]) -> list[str]:
    selected = []
    for image in range(int(states["vis"]["n"])):
        visible_score = topk(states["vis"]["cf"][image])
        pia_score = topk(states["piafusion"]["cf"][image])
        if pia_score >= max(visible_score, MARGIN * visible_score):
            selected.append("piafusion")
        else:
            selected.append("vis")
    return selected


def selected_stats(
    states: dict[str, dict],
    selected: list[str],
) -> dict:
    tp = []
    confidence = []
    pred_classes = []
    target_classes = []
    det_counts = []
    gt_counts = []
    for image, candidate in enumerate(selected):
        state = states[candidate]
        tp.append(state["tp"][image])
        confidence.append(state["cf"][image])
        pred_classes.append(state["cl"][image])
        target_classes.append(state["gt"][image])
        det_counts.append(len(state["cf"][image]))
        gt_counts.append(len(state["gt"][image]))
    det_counts = np.asarray(det_counts, dtype=np.int64)
    gt_counts = np.asarray(gt_counts, dtype=np.int64)
    return {
        "n": state["n"],
        "stems": states["vis"]["stems"],
        "tp_all": np.concatenate(tp),
        "cf_all": np.concatenate(confidence),
        "cl_all": np.concatenate(pred_classes),
        "gt_all": np.concatenate(target_classes),
        "det_counts": det_counts,
        "det_offsets": np.concatenate(([0], np.cumsum(det_counts)[:-1])),
        "gt_counts": gt_counts,
        "gt_offsets": np.concatenate(([0], np.cumsum(gt_counts)[:-1])),
    }


def image_ap50(state: dict, image: int) -> float:
    target = state["gt"][image]
    if len(target) == 0:
        return 0.0
    tp = state["tp"][image][:, 0]
    confidence = state["cf"][image]
    pred_classes = state["cl"][image]
    order = np.argsort(-confidence)
    tp = tp[order]
    pred_classes = pred_classes[order]
    values = []
    for class_id, count in zip(*np.unique(target, return_counts=True)):
        selected = pred_classes == class_id
        if not selected.any():
            values.append(0.0)
            continue
        tpc = np.cumsum(tp[selected])
        fpc = np.cumsum(1 - tp[selected])
        recall = tpc / max(float(count), 1.0)
        precision = tpc / np.maximum(tpc + fpc, 1e-16)
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
        x = np.linspace(0.0, 1.0, 101)
        values.append(
            float(np.trapezoid(np.interp(x, mrec, mpre), x))
            if hasattr(np, "trapezoid")
            else float(np.trapz(np.interp(x, mrec, mpre), x))
        )
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores",
        type=Path,
        default=(
            PACKAGE
            / "evidence"
            / "prospective_pia_confirmation"
            / "scores"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PACKAGE / "evidence" / "prospective_pia_confirmation",
    )
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rows: dict[str, dict[str, dict]] = {}
    for model in MODELS:
        rows[model] = {}
        for sequence in SEQUENCES:
            score_dir = args.scores / f"llvip{sequence}"
            states = {
                candidate: load(score_dir / f"{model}__{candidate}.npz")
                for candidate in ("vis", "piafusion")
            }
            selected = choices(states)
            stats = selected_stats(states, selected)
            blocks = [
                np.arange(start, min(start + BLOCK_SIZE, 300), dtype=np.int64)
                for start in range(0, 300, BLOCK_SIZE)
            ]
            gain50 = np.asarray([
                pooled(stats, block)[0] - pooled(states["vis"], block)[0]
                for block in blocks
            ])
            gain95 = np.asarray([
                pooled(stats, block)[1] - pooled(states["vis"], block)[1]
                for block in blocks
            ])
            raw_gain50 = np.asarray([
                pooled(states["piafusion"], block)[0]
                - pooled(states["vis"], block)[0]
                for block in blocks
            ])
            raw_gain95 = np.asarray([
                pooled(states["piafusion"], block)[1]
                - pooled(states["vis"], block)[1]
                for block in blocks
            ])
            visible_image = np.asarray([
                image_ap50(states["vis"], image)
                for image in range(300)
            ])
            selected_image = np.asarray([
                image_ap50(states[selected[image]], image)
                for image in range(300)
            ])
            rows[model][sequence] = {
                "point50": float(gain50.mean()),
                "point95": float(gain95.mean()),
                "block_gains50": gain50,
                "block_gains95": gain95,
                "raw_block_gains50": raw_gain50,
                "raw_block_gains95": raw_gain95,
                "switch_fraction": float(np.mean([
                    value == "piafusion" for value in selected
                ])),
                "image_win_fraction": float(np.mean(
                    selected_image > visible_image + 1e-12
                )),
                "image_tie_fraction": float(np.mean(
                    np.abs(selected_image - visible_image) <= 1e-12
                )),
                "image_loss_fraction": float(np.mean(
                    selected_image < visible_image - 1e-12
                )),
            }

    per_sequence = {
        sequence: float(np.mean([
            rows[model][sequence]["point50"] for model in MODELS
        ]))
        for sequence in SEQUENCES
    }
    per_model = {
        model: float(np.mean([
            rows[model][sequence]["point50"] for sequence in SEQUENCES
        ]))
        for model in MODELS
    }
    point50 = float(np.mean(list(per_sequence.values())))
    point95 = float(np.mean([
        rows[model][sequence]["point95"]
        for model in MODELS
        for sequence in SEQUENCES
    ]))
    raw_point50 = float(np.mean([
        rows[model][sequence]["raw_block_gains50"].mean()
        for model in MODELS
        for sequence in SEQUENCES
    ]))
    raw_point95 = float(np.mean([
        rows[model][sequence]["raw_block_gains95"].mean()
        for model in MODELS
        for sequence in SEQUENCES
    ]))
    leave_one_sequence_out = {}
    for omitted in SEQUENCES:
        retained = [
            sequence for sequence in SEQUENCES
            if sequence != omitted
        ]
        checkpoint_values = {
            model: float(np.mean([
                rows[model][sequence]["point50"]
                for sequence in retained
            ]))
            for model in MODELS
        }
        leave_one_sequence_out[omitted] = {
            "retained_sequences": retained,
            "equal_sequence_point": float(
                np.mean(list(checkpoint_values.values()))
            ),
            "per_checkpoint": checkpoint_values,
            "worst_checkpoint": float(min(checkpoint_values.values())),
            "positive_checkpoints": int(
                sum(value > 0 for value in checkpoint_values.values())
            ),
        }

    def bootstrap(key: str) -> list[float]:
        rng = np.random.default_rng(args.seed)
        draws = []
        for _ in range(args.draws):
            models = rng.choice(MODELS, size=len(MODELS), replace=True)
            sequences = rng.choice(SEQUENCES, size=len(SEQUENCES), replace=True)
            model_values = []
            for model in models:
                sequence_values = []
                for sequence in sequences:
                    values = rows[model][sequence][key]
                    block_indices = rng.integers(
                        0, len(values), size=len(values)
                    )
                    sequence_values.append(float(values[block_indices].mean()))
                model_values.append(float(np.mean(sequence_values)))
            draws.append(float(np.mean(model_values)))
        return [
            float(value)
            for value in np.percentile(draws, [2.5, 97.5])
        ]

    image_metrics = {
        key: float(np.mean([
            rows[model][sequence][f"image_{key}_fraction"]
            for model in MODELS
            for sequence in SEQUENCES
        ]))
        for key in ("win", "tie", "loss")
    }
    payload = {
        "status": "prospective_temporal_holdout",
        "scope": "LLVIP sequences 080, 090, and 100, first 300 non-empty labels each",
        "n_images": 900,
        "policy": {
            "candidate_library": ["vis", "piafusion"],
            "top_k": TOP_K,
            "margin": MARGIN,
            "selection_rule": (
                "Select PIAFusion when its top-five score is at least "
                "max(visible score, 1.5 * visible score); otherwise visible."
            ),
        },
        "matching": "confidence-ordered one-to-one",
        "primary": {
            "estimand": "equal-sequence mean AP50 difference versus visible",
            "point": point50,
            "conditional_interval": bootstrap("block_gains50"),
            "per_sequence": per_sequence,
            "per_checkpoint": per_model,
            "positive_sequences": int(
                sum(value > 0 for value in per_sequence.values())
            ),
            "positive_checkpoints": int(
                sum(value > 0 for value in per_model.values())
            ),
            "worst_checkpoint": float(min(per_model.values())),
        },
        "ap50_95": {
            "point": point95,
            "conditional_interval": bootstrap("block_gains95"),
        },
        "raw_pia_vs_visible": {
            "ap50": {
                "point": raw_point50,
                "conditional_interval": bootstrap("raw_block_gains50"),
                "per_sequence": {
                    sequence: float(np.mean([
                        rows[model][sequence]["raw_block_gains50"].mean()
                        for model in MODELS
                    ]))
                    for sequence in SEQUENCES
                },
            },
            "ap50_95": {
                "point": raw_point95,
                "conditional_interval": bootstrap("raw_block_gains95"),
            },
        },
        "leave_one_sequence_out": {
            "status": "post_hoc_sensitivity_analysis",
            "endpoint": (
                "Equal-sequence AP50 difference versus visible after "
                "omitting one fixed sequence."
            ),
            "results": leave_one_sequence_out,
        },
        "switch_fraction": float(np.mean([
            rows[model][sequence]["switch_fraction"]
            for model in MODELS
            for sequence in SEQUENCES
        ])),
        "image_metrics": image_metrics,
        "runs": {
            model: {
                sequence: {
                    key: (
                        [float(value) for value in row[key]]
                        if key.startswith("block_gains")
                        else row[key]
                    )
                    for key in (
                        "point50",
                        "point95",
                        "switch_fraction",
                        "image_win_fraction",
                        "image_tie_fraction",
                        "image_loss_fraction",
                    )
                }
                for sequence, row in sequence_rows.items()
            }
            for model, sequence_rows in rows.items()
        },
        "limitations": (
            "Three fixed sequences, six fixed detector runs, and 300 "
            "lexicographically selected frames per sequence. The interval "
            "resamples within the observed sequence and checkpoint set."
        ),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "analysis.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.out / "runs.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "model",
            "sequence",
            "ap50_gain",
            "ap50_95_gain",
            "switch_fraction",
            "image_win_fraction",
            "image_tie_fraction",
            "image_loss_fraction",
        ])
        for model in MODELS:
            for sequence in SEQUENCES:
                row = rows[model][sequence]
                writer.writerow([
                    model,
                    sequence,
                    row["point50"],
                    row["point95"],
                    row["switch_fraction"],
                    row["image_win_fraction"],
                    row["image_tie_fraction"],
                    row["image_loss_fraction"],
                ])
    with (args.out / "leave_one_sequence_out.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "omitted_sequence",
            "retained_sequences",
            "equal_sequence_ap50_gain",
            "worst_checkpoint",
            "positive_checkpoints",
        ])
        for omitted, row in leave_one_sequence_out.items():
            writer.writerow([
                omitted,
                "+".join(row["retained_sequences"]),
                row["equal_sequence_point"],
                row["worst_checkpoint"],
                row["positive_checkpoints"],
            ])
    print(
        json.dumps(
            {
                "point50": point50,
                "interval50": payload["primary"]["conditional_interval"],
                "point95": point95,
                "interval95": payload["ap50_95"]["conditional_interval"],
                "raw_point50": raw_point50,
                "raw_interval50": payload["raw_pia_vs_visible"]["ap50"][
                    "conditional_interval"
                ],
                "positive_sequences": payload["primary"]["positive_sequences"],
                "positive_checkpoints": payload["primary"][
                    "positive_checkpoints"
                ],
                "leave_one_sequence_out": {
                    omitted: {
                        "point50": row["equal_sequence_point"],
                        "positive_checkpoints": row["positive_checkpoints"],
                    }
                    for omitted, row
                    in leave_one_sequence_out.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
