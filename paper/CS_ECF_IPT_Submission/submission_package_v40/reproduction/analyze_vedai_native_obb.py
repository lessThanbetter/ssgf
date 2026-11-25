"""Analyze VEDAI OBB candidate scores under the frozen CS-ECF policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_ap_reference import compute_ap, load  # noqa: E402


TOP_K = 5
MARGIN = 1.5
BLOCKS = 10
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 4816


def topk(values: np.ndarray) -> float:
    if len(values) == 0:
        return -1.0
    return float(np.sort(values)[-TOP_K:].mean())


def decisions(states: dict[str, dict], margin: float = MARGIN) -> list[str]:
    selected = []
    for image in range(states["vis"]["n"]):
        scores = {
            candidate: topk(states[candidate]["cf"][image])
            for candidate in ("vis", "cdf", "pia", "spg")
        }
        best = max(("vis", "cdf", "pia", "spg"), key=lambda name: scores[name])
        if best != "vis" and scores[best] >= margin * scores["vis"]:
            selected.append(best)
        else:
            selected.append("vis")
    return selected


def policy_stats(states: dict[str, dict], selected: list[str]) -> dict:
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
        "n": states["vis"]["n"],
        "stems": states["vis"]["stems"],
        "tp_all": np.concatenate(tp),
        "cf_all": np.concatenate(conf),
        "cl_all": np.concatenate(pred_cls),
        "gt_all": np.concatenate(gt),
        "det_counts": det_counts_array,
        "det_offsets": np.concatenate(([0], np.cumsum(det_counts_array)[:-1])),
        "gt_counts": gt_counts_array,
        "gt_offsets": np.concatenate(([0], np.cumsum(gt_counts_array)[:-1])),
    }


def ap50(state: dict, indices: np.ndarray) -> float:
    return complete_ap50(state, indices)


def image_ap50(state: dict, image: int) -> float:
    if "det_offsets" in state:
        tp = state["tp_all"][
            state["det_offsets"][image] :
            state["det_offsets"][image] + state["det_counts"][image]
        ]
        conf = state["cf_all"][
            state["det_offsets"][image] :
            state["det_offsets"][image] + state["det_counts"][image]
        ]
        pred_cls = state["cl_all"][
            state["det_offsets"][image] :
            state["det_offsets"][image] + state["det_counts"][image]
        ]
        target_cls = state["gt_all"][
            state["gt_offsets"][image] :
            state["gt_offsets"][image] + state["gt_counts"][image]
        ]
    else:
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


def complete_ap50(state: dict, indices: np.ndarray) -> float:
    indices = np.asarray(indices, dtype=np.int64)
    if "det_offsets" in state:
        def gather(values, offsets_name, counts_name):
            offsets = state[offsets_name][indices]
            counts = state[counts_name][indices]
            total = int(counts.sum())
            if total == 0:
                return values[:0]
            group_starts = np.repeat(offsets, counts)
            starts = np.repeat(np.cumsum(counts) - counts, counts)
            positions = group_starts + np.arange(total) - starts
            return values[positions]

        tp = gather(state["tp_all"], "det_offsets", "det_counts")
        conf = gather(state["cf_all"], "det_offsets", "det_counts")
        pred_cls = gather(state["cl_all"], "det_offsets", "det_counts")
        gt = gather(state["gt_all"], "gt_offsets", "gt_counts")
    else:
        tp = np.concatenate([state["tp"][index] for index in indices])
        conf = np.concatenate([state["cf"][index] for index in indices])
        pred_cls = np.concatenate([state["cl"][index] for index in indices])
        gt = np.concatenate([state["gt"][index] for index in indices])
    if len(gt) == 0:
        return 0.0
    order = np.argsort(-conf)
    tp = tp[order, 0]
    pred_cls = pred_cls[order]
    values = []
    for class_id, count in zip(*np.unique(gt, return_counts=True)):
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


def class_ap50(state: dict) -> dict[str, float]:
    order = np.argsort(-state["cf_all"])
    tp = state["tp_all"][order, 0]
    pred_cls = state["cl_all"][order]
    gt_all = state["gt_all"]
    values = {}
    for class_id, class_name in enumerate(
        ("car", "truck", "tractor", "camping car", "van")
    ):
        count = int(np.sum(gt_all == class_id))
        selected = pred_cls == class_id
        if count == 0:
            continue
        if not selected.any():
            values[class_name] = 0.0
            continue
        tpc = np.cumsum(tp[selected])
        fpc = np.cumsum(1 - tp[selected])
        recall = tpc / count
        precision = tpc / np.maximum(tpc + fpc, 1e-16)
        values[class_name] = compute_ap(recall, precision)
    return values


def load_states(score_dir: Path, model: str, stems: list[str]) -> dict:
    states = {}
    for short, filename in (
        ("vis", "vis"),
        ("cdf", "cdf"),
        ("pia", "pia"),
        ("spg", "spg"),
    ):
        path = score_dir / f"{model}__{filename}.npz"
        state = load(path)
        if state["stems"] != stems:
            raise ValueError(f"stem order mismatch in {path}")
        states[short] = state
    return states


def summarize(states: dict[str, dict], blocks: list[np.ndarray]) -> dict:
    choices = {
        "visible": ["vis"] * states["vis"]["n"],
        "always_cdf": ["cdf"] * states["vis"]["n"],
        "always_pia": ["pia"] * states["vis"]["n"],
        "always_spg": ["spg"] * states["vis"]["n"],
        "no_shield": decisions(states, margin=0.0),
        "csecf": decisions(states, margin=MARGIN),
    }
    stats = {
        name: policy_stats(states, selected)
        for name, selected in choices.items()
    }
    block_ap = {
        name: np.asarray([ap50(state, block) for block in blocks])
        for name, state in stats.items()
    }
    comparisons = {}
    for left, right in (
        ("csecf", "visible"),
        ("always_cdf", "visible"),
        ("no_shield", "visible"),
        ("csecf", "always_cdf"),
        ("csecf", "no_shield"),
    ):
        gains = block_ap[left] - block_ap[right]
        comparisons[f"{left}_minus_{right}"] = {
            "equal_block_mean": float(gains.mean()),
            "block_gains": gains.tolist(),
            "positive_blocks": int(np.sum(gains > 0)),
            "worst_block": float(gains.min()),
        }
    visible_image = np.asarray(
        [image_ap50(stats["visible"], image) for image in range(states["vis"]["n"])]
    )
    csecf_image = np.asarray(
        [image_ap50(stats["csecf"], image) for image in range(states["vis"]["n"])]
    )
    csecf_classes = class_ap50(stats["csecf"])
    visible_classes = class_ap50(stats["visible"])
    return {
        "absolute_ap50": {
            name: complete_ap50(
                state,
                np.arange(states["vis"]["n"]),
            )
            for name, state in stats.items()
        },
        "class_ap50": {
            name: class_ap50(state)
            for name, state in stats.items()
        },
        "comparisons": comparisons,
        "csecf_minus_visible_by_class": {
            key: csecf_classes[key] - visible_classes[key]
            for key in csecf_classes
        },
        "switch_fraction": float(
            np.mean([candidate != "vis" for candidate in choices["csecf"]])
        ),
        "selected_counts": {
            candidate: int(sum(value == candidate for value in choices["csecf"]))
            for candidate in ("vis", "cdf", "pia", "spg")
        },
        "unshielded_counts": {
            candidate: int(
                sum(value == candidate for value in choices["no_shield"])
            )
            for candidate in ("vis", "cdf", "pia", "spg")
        },
        "image_win_fraction": float(np.mean(csecf_image > visible_image + 1e-12)),
        "image_tie_fraction": float(
            np.mean(np.abs(csecf_image - visible_image) <= 1e-12)
        ),
        "image_loss_fraction": float(
            np.mean(csecf_image < visible_image - 1e-12)
        ),
    }


def aggregate(models: dict[str, dict]) -> dict:
    names = tuple(models)
    result = {}
    comparison_keys = tuple(
        next(iter(models.values()))["comparisons"]
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for comparison in comparison_keys:
        per_model = {
            name: np.asarray(
                models[name]["comparisons"][comparison]["block_gains"]
            )
            for name in names
        }
        point = float(np.mean([values.mean() for values in per_model.values()]))
        draws = []
        for _ in range(BOOTSTRAP_DRAWS):
            sampled_models = rng.choice(names, size=len(names), replace=True)
            model_values = []
            for model in sampled_models:
                values = per_model[str(model)]
                sampled = rng.integers(0, len(values), size=len(values))
                model_values.append(float(values[sampled].mean()))
            draws.append(float(np.mean(model_values)))
        result[comparison] = {
            "equal_run_mean": point,
            "conditional_run_by_block_interval": [
                float(value) for value in np.percentile(draws, [2.5, 97.5])
            ],
            "positive_runs": int(
                sum(values.mean() > 0 for values in per_model.values())
            ),
            "worst_run": float(min(values.mean() for values in per_model.values())),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    stems = sorted(path.stem for path in args.labels.glob("*.txt"))
    blocks = np.array_split(np.arange(len(stems)), BLOCKS)
    models = {}
    for model in args.model:
        states = load_states(args.score_dir, model, stems)
        models[model] = summarize(states, blocks)
    payload = {
        "protocol": {
            "top_k": TOP_K,
            "margin": MARGIN,
            "blocks": BLOCKS,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "models": args.model,
        },
        "models": models,
        "aggregate": aggregate(models),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
