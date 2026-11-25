"""Analyze the preregistered LLVIP sequence holdout."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_ap_reference import compute_ap, load, pooled  # noqa: E402


TOP_K = 5
FROZEN_MARGIN = 1.5
MARGINS = (1.0, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0)
CDF_NAME = "consensus_intersection_pixel_mean_ifcnn_q095_a020_s05"
MODEL_NAMES = (
    "yolo26_s0",
    "yolo26_s1",
    "yolo26_s2",
    "yolo8_s0",
    "yolo8_s1",
    "rtdetr_l",
)
CANDIDATES = {
    "vis": "vis",
    "cdf": "cdf",
    "pia": "piafusion",
    "spg": "spgfusion",
}
CONFIRM_SEQUENCES = ("050", "060", "070")
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 2917
BLOCK_SIZE = 50


def _topk(values: np.ndarray) -> float:
    if len(values) == 0:
        return -1.0
    return float(np.sort(values)[-TOP_K:].mean())


def _decisions(
    states: dict[str, dict],
    candidates: tuple[str, ...] = ("vis", "cdf", "pia", "spg"),
    margin: float = FROZEN_MARGIN,
) -> list[str]:
    selected = []
    for image in range(states["vis"]["n"]):
        score = {
            candidate: _topk(states[candidate]["cf"][image])
            for candidate in candidates
        }
        best = max(candidates, key=lambda name: score[name])
        if best != "vis" and score[best] >= margin * score["vis"]:
            selected.append(best)
        else:
            selected.append("vis")
    return selected


def _policy_stats(states: dict[str, dict], selected: list[str]) -> dict:
    reference = states["vis"]
    tp = []
    conf = []
    pred_cls = []
    gt = []
    det_counts = []
    gt_counts = []
    for image, candidate in enumerate(selected):
        state = states[candidate]
        tp.append(state["tp"][image])
        conf.append(state["cf"][image])
        pred_cls.append(state["cl"][image])
        gt.append(state["gt"][image])
        det_counts.append(len(state["cf"][image]))
        gt_counts.append(len(state["gt"][image]))
    det_counts_array = np.asarray(det_counts, dtype=np.int64)
    gt_counts_array = np.asarray(gt_counts, dtype=np.int64)
    return {
        "n": reference["n"],
        "stems": reference["stems"],
        "tp_all": np.concatenate(tp),
        "cf_all": np.concatenate(conf),
        "cl_all": np.concatenate(pred_cls),
        "gt_all": np.concatenate(gt),
        "det_counts": det_counts_array,
        "det_offsets": np.concatenate((
            [0],
            np.cumsum(det_counts_array)[:-1],
        )),
        "gt_counts": gt_counts_array,
        "gt_offsets": np.concatenate((
            [0],
            np.cumsum(gt_counts_array)[:-1],
        )),
    }


def _ap(stats: dict, indices: np.ndarray) -> float:
    return float(pooled(stats, np.asarray(indices, dtype=np.int64))[0])


def _image_ap50(state: dict, image: int) -> float:
    tp = state["tp"][image]
    conf = state["cf"][image]
    pred_cls = state["cl"][image]
    target_cls = state["gt"][image]
    if len(target_cls) == 0:
        return 0.0
    order = np.argsort(-conf)
    tp = tp[order, 0]
    pred_cls = pred_cls[order]
    values = []
    for class_id, count in zip(*np.unique(target_cls, return_counts=True)):
        selected = pred_cls == class_id
        if not selected.any():
            values.append(0.0)
            continue
        tpc = np.cumsum(tp[selected])
        fpc = np.cumsum(1 - tp[selected])
        recall = tpc / max(float(count), 1.0)
        precision = tpc / np.maximum(tpc + fpc, 1e-16)
        values.append(compute_ap(recall, precision))
    return float(np.mean(values))


def _blocks(stems: list[str]) -> list[np.ndarray]:
    return [
        np.arange(start, min(start + BLOCK_SIZE, len(stems)), dtype=np.int64)
        for start in range(0, len(stems), BLOCK_SIZE)
    ]


def _load_model(score_dir: Path, model: str, stems: list[str]) -> dict:
    states = {}
    for short, filename in CANDIDATES.items():
        path = score_dir / f"{model}__{filename}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        state = load(path)
        if state["stems"] != stems:
            raise ValueError(f"stem order mismatch in {path.name}")
        states[short] = state
    return states


def _quality_choices(
    features: dict[str, dict],
    stems: list[str],
    metric: str,
    margin: float,
) -> list[str]:
    choices = []
    for stem in stems:
        row = features[stem][metric]
        best = max(("vis", "cdf", "pia", "spg"), key=lambda name: row[name])
        if best != "vis" and row[best] >= margin * row["vis"]:
            choices.append(best)
        else:
            choices.append("vis")
    return choices


def _selection_record(
    states: dict[str, dict],
    blocks: list[np.ndarray],
    selected: list[str],
) -> dict:
    selected_stats = _policy_stats(states, selected)
    block_gains = np.asarray([
        _ap(selected_stats, block) - _ap(states["vis"], block)
        for block in blocks
    ])
    visible_image_ap = np.asarray([
        _image_ap50(states["vis"], image)
        for image in range(states["vis"]["n"])
    ])
    selected_image_ap = np.asarray([
        _image_ap50(
            states[selected[image]],
            image,
        )
        for image in range(states["vis"]["n"])
    ])
    return {
        "states": states,
        "selected": selected,
        "selected_stats": selected_stats,
        "block_gains": block_gains,
        "point": float(block_gains.mean()),
        "switch_fraction": float(
            np.mean([candidate != "vis" for candidate in selected])
        ),
        "image_win_fraction": float(
            np.mean(selected_image_ap > visible_image_ap + 1e-12)
        ),
        "image_tie_fraction": float(
            np.mean(np.abs(selected_image_ap - visible_image_ap) <= 1e-12)
        ),
        "image_loss_fraction": float(
            np.mean(selected_image_ap < visible_image_ap - 1e-12)
        ),
    }


def _checkpoint_rows(
    states: dict[str, dict],
    blocks: list[np.ndarray],
) -> dict:
    return _selection_record(states, blocks, _decisions(states))


def _summarize(
    rows: dict[str, dict[str, dict]],
    sequences,
    model_names: tuple[str, ...] | None = None,
) -> dict:
    models = tuple(rows) if model_names is None else model_names
    values = np.asarray([
        rows[model][sequence]["point"]
        for model in models
        for sequence in sequences
    ])
    checkpoint_means = {
        model: float(np.mean([
            rows[model][sequence]["point"] for sequence in sequences
        ]))
        for model in models
    }
    return {
        "sequence_equal_point": float(values.mean()),
        "per_sequence": {
            sequence: float(np.mean([
                rows[model][sequence]["point"] for model in models
            ]))
            for sequence in sequences
        },
        "per_checkpoint": checkpoint_means,
        "positive_checkpoints": int(
            sum(value > 0 for value in checkpoint_means.values())
        ),
        "worst_checkpoint": float(min(checkpoint_means.values())),
        "switch_fraction": float(np.mean([
            rows[model][sequence]["switch_fraction"]
            for model in models
            for sequence in sequences
        ])),
        "image_win_fraction": float(np.mean([
            rows[model][sequence]["image_win_fraction"]
            for model in models
            for sequence in sequences
        ])),
        "image_loss_fraction": float(np.mean([
            rows[model][sequence]["image_loss_fraction"]
            for model in models
            for sequence in sequences
        ])),
    }


def _bootstrap_interval(
    rows: dict[str, dict[str, dict]],
    model_names: tuple[str, ...] | None = None,
) -> list[float]:
    models = tuple(rows) if model_names is None else model_names
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        sampled_sequences = rng.choice(
            CONFIRM_SEQUENCES,
            size=len(CONFIRM_SEQUENCES),
            replace=True,
        )
        sampled_models = rng.choice(
            models,
            size=len(models),
            replace=True,
        )
        model_values = []
        for model in sampled_models:
            sequence_values = []
            for sequence in sampled_sequences:
                block_values = rows[model][sequence]["block_gains"]
                sampled_blocks = rng.integers(
                    0,
                    len(block_values),
                    size=len(block_values),
                )
                sequence_values.append(
                    float(block_values[sampled_blocks].mean())
                )
            model_values.append(float(np.mean(sequence_values)))
        draws.append(float(np.mean(model_values)))
    draws = np.asarray(draws)
    return [
        float(value)
        for value in np.percentile(draws, [2.5, 97.5])
    ]


def _relative_rows(
    states_by_model: dict[str, dict[str, dict]],
    blocks_by_model: dict[str, list[np.ndarray]],
    left: str,
    right: str,
    model_names: tuple[str, ...],
) -> dict[str, dict[str, dict]]:
    rows = {}
    for model in model_names:
        rows[model] = {}
        states = states_by_model[model]
        left_choice = (
            ["vis"] * states["vis"]["n"]
            if left == "visible"
            else ["cdf"] * states["vis"]["n"]
            if left == "cdf"
            else _decisions(states, margin=0.0)
            if left == "no_shield"
            else _decisions(states)
        )
        right_choice = (
            ["vis"] * states["vis"]["n"]
            if right == "visible"
            else ["cdf"] * states["vis"]["n"]
            if right == "cdf"
            else _decisions(states, margin=0.0)
            if right == "no_shield"
            else _decisions(states)
        )
        left_stats = _policy_stats(states, left_choice)
        right_stats = _policy_stats(states, right_choice)
        block_gains = np.asarray([
            _ap(left_stats, block) - _ap(right_stats, block)
            for block in blocks_by_model[model]
        ])
        rows[model] = {
            "point": float(block_gains.mean()),
            "block_gains": block_gains,
            "switch_fraction": 0.0,
            "image_win_fraction": 0.0,
            "image_loss_fraction": 0.0,
        }
    return rows


def _comparison_payload(
    states_by_sequence: dict[str, dict[str, dict[str, dict]]],
    blocks_by_sequence: dict[str, dict[str, list[np.ndarray]]],
    comparisons: dict[str, tuple[str, str]],
    model_names: tuple[str, ...],
) -> dict:
    payload = {}
    for label, (left, right) in comparisons.items():
        rows = {}
        for model in model_names:
            rows[model] = {}
            for sequence in CONFIRM_SEQUENCES:
                single = _relative_rows(
                    {model: states_by_sequence[sequence][model]},
                    {model: blocks_by_sequence[sequence][model]},
                    left,
                    right,
                    (model,),
                )
                rows[model][sequence] = single[model]
        payload[label] = {
            "summary": _summarize(rows, CONFIRM_SEQUENCES, model_names),
            "conditional_sequence_by_run_interval": _bootstrap_interval(
                rows,
                model_names,
            ),
        }
    return payload


def _margin_audit(
    states_by_model: dict[str, dict[str, dict[str, dict]]],
    blocks_by_model: dict[str, dict[str, list[np.ndarray]]],
) -> dict:
    rows = {}
    for margin in MARGINS:
        model_values = {}
        switch_values = {}
        for model, states in states_by_model["040"].items():
            selected = _decisions(states, margin=margin)
            stats = _policy_stats(states, selected)
            model_values[model] = float(np.mean([
                _ap(stats, block) - _ap(states["vis"], block)
                for block in blocks_by_model["040"][model]
            ]))
            switch_values[model] = float(
                np.mean([candidate != "vis" for candidate in selected])
            )
        rows[f"{margin:.2f}"] = {
            "balanced_gain": float(np.mean(list(model_values.values()))),
            "per_checkpoint": model_values,
            "worst_checkpoint": float(min(model_values.values())),
            "switch_fraction": float(np.mean(list(switch_values.values()))),
        }
    feasible = {
        margin: row
        for margin, row in rows.items()
        if row["worst_checkpoint"] >= -0.0005
    }
    selected_margin = max(
        feasible,
        key=lambda margin: (
            feasible[margin]["balanced_gain"],
            float(margin),
        ),
    )
    return {
        "grid": rows,
        "selected_margin": float(selected_margin),
        "constraint": "every checkpoint gain >= -0.0005 AP50",
    }


def _quality_audit(
    selection_rows: dict[str, dict[str, dict[str, dict]]],
    features_by_sequence: dict[str, dict[str, dict]],
    stems_by_sequence: dict[str, list[str]],
) -> dict:
    result = {}
    for metric in ("vif", "ssim", "entropy"):
        grid = {}
        for margin in MARGINS:
            values = {}
            switches = {}
            for model, record in selection_rows["040"].items():
                states = record["states"]
                stems = stems_by_sequence["040"]
                selected = _quality_choices(
                    features_by_sequence["040"],
                    stems,
                    metric,
                    margin,
                )
                stats = _policy_stats(states, selected)
                blocks = _blocks(stems)
                values[model] = float(np.mean([
                    _ap(stats, block) - _ap(states["vis"], block)
                    for block in blocks
                ]))
                switches[model] = float(
                    np.mean([candidate != "vis" for candidate in selected])
                )
            grid[f"{margin:.2f}"] = {
                "balanced_gain": float(np.mean(list(values.values()))),
                "worst_checkpoint": float(min(values.values())),
                "switch_fraction": float(np.mean(list(switches.values()))),
            }
        feasible = {
            margin: row
            for margin, row in grid.items()
            if row["worst_checkpoint"] >= -0.0005
        }
        gain_margin = max(
            feasible,
            key=lambda margin: (
                feasible[margin]["balanced_gain"],
                float(margin),
            ),
        )
        target = float(np.mean([
            selection_rows["040"][model]["switch_fraction"]
            for model in MODEL_NAMES
        ]))
        budget_margin = min(
            feasible,
            key=lambda margin: (
                abs(
                    feasible[margin]["switch_fraction"] - target
                ),
                -feasible[margin]["balanced_gain"],
            ),
        )
        variants = {}
        for label, margin in (
            ("gain_tuned", gain_margin),
            ("budget_matched", budget_margin),
        ):
            test_rows = {}
            for model in MODEL_NAMES:
                test_rows[model] = {}
                for sequence in CONFIRM_SEQUENCES:
                    stems = stems_by_sequence[sequence]
                    states = selection_rows[sequence][model]["states"]
                    selected = _quality_choices(
                        features_by_sequence[sequence],
                        stems,
                        metric,
                        float(margin),
                    )
                    test_rows[model][sequence] = _selection_record(
                        states,
                        _blocks(stems),
                        selected,
                    )
            variants[label] = {
                "margin": float(margin),
                "summary": _summarize(
                    {
                        model: {
                            sequence: test_rows[model][sequence]
                            for sequence in CONFIRM_SEQUENCES
                        }
                        for model in MODEL_NAMES
                    },
                    CONFIRM_SEQUENCES,
                ),
            }
        result[metric] = {
            "validation_grid": grid,
            "selected_target_switch_fraction": target,
            "variants": variants,
        }
    return result


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

    sequences = ("040",) + CONFIRM_SEQUENCES
    selection_rows = {}
    features_by_sequence = {}
    stems_by_sequence = {}
    blocks_by_model = {}
    for sequence in sequences:
        source_dir = (
            data_root
            / f"erf_holdout_llvip{sequence}"
        )
        stems = sorted(
            path.stem for path in (source_dir / "visible").glob("*.jpg")
        )
        stems_by_sequence[sequence] = stems
        score_dir = root / f"llvip{sequence}" / "scores"
        selection_rows[sequence] = {}
        blocks_by_model[sequence] = {}
        for model in MODEL_NAMES:
            states = _load_model(score_dir, model, stems)
            blocks = _blocks(stems)
            selection_rows[sequence][model] = _checkpoint_rows(
                states,
                blocks,
            )
            blocks_by_model[sequence][model] = blocks
        feature_path = root / f"llvip{sequence}" / "features.json"
        feature_rows = json.loads(feature_path.read_text(encoding="utf-8"))
        features_by_sequence[sequence] = {
            row["stem"]: row for row in feature_rows
        }

    confirmation_rows = {
        model: {
            sequence: selection_rows[sequence][model]
            for sequence in CONFIRM_SEQUENCES
        }
        for model in MODEL_NAMES
    }
    frozen_summary = _summarize(confirmation_rows, CONFIRM_SEQUENCES)
    interval = _bootstrap_interval(confirmation_rows)
    comparison_states = {}
    comparison_blocks = {}
    for sequence in CONFIRM_SEQUENCES:
        comparison_states[sequence] = {}
        comparison_blocks[sequence] = {}
        for model in MODEL_NAMES:
            comparison_states[sequence][model] = (
                selection_rows[sequence][model]["states"]
            )
            comparison_blocks[sequence][model] = blocks_by_model[sequence][
                model
            ]
    comparisons = _comparison_payload(
        comparison_states,
        comparison_blocks,
        {
            "csecf_minus_visible": ("csecf", "visible"),
            "cdf_minus_visible": ("cdf", "visible"),
            "no_shield_minus_visible": ("no_shield", "visible"),
            "csecf_minus_cdf": ("csecf", "cdf"),
            "csecf_minus_no_shield": ("csecf", "no_shield"),
        },
        MODEL_NAMES,
    )

    # Candidate-margin audit is descriptive and cannot change the frozen policy.
    margin_audit = _margin_audit(
        {
            sequence: {
                model: selection_rows[sequence][model]["states"]
                for model in MODEL_NAMES
            }
            for sequence in sequences
        },
        blocks_by_model,
    )

    # Matched image-statistic gates use the same candidates and fallback.
    # `_quality_audit` expects checkpoint records with switch fractions.
    quality_rows = {
        sequence: selection_rows[sequence]
        for sequence in sequences
    }
    quality = _quality_audit(
        quality_rows,
        features_by_sequence,
        stems_by_sequence,
    )
    output = {
        "protocol": {
            "frozen_margin": FROZEN_MARGIN,
            "top_k": TOP_K,
            "validation_sequence": "040",
            "confirmation_sequences": list(CONFIRM_SEQUENCES),
            "models": list(MODEL_NAMES),
            "block_size_frames": BLOCK_SIZE,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "confirmation": {
            "frozen_policy": frozen_summary,
            "conditional_sequence_by_run_interval": interval,
            "comparisons": comparisons,
        },
        "validation_margin_audit": margin_audit,
        "matched_quality": quality,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
