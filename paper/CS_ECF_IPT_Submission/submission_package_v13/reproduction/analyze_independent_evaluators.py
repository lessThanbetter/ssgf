"""Evaluate the frozen CS-ECF policy on unused evaluator checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analyze_csecf_vs_cdf import (
    VISIBLE_MARGIN,
    TOP_K,
    _ap,
    _decisions,
    _draw_by_group,
    _interval,
    _policy_stats,
)
from local_ap_reference import load


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "independent_evaluators"
PER_IMAGE = RESULT_DIR / "per_image"
OUT = RESULT_DIR / "policy_summary.json"
MODELS = ("yolov9t_coco", "yolov10s_coco", "yolov9t_domain")
MODEL_DIRS = {
    "yolov9t_coco": PER_IMAGE,
    "yolov10s_coco": PER_IMAGE,
    "yolov9t_domain": RESULT_DIR / "domain_yolov9t_v3" / "per_image",
}
MODEL_SEEDS = {
    "yolov9t_coco": 2219,
    "yolov10s_coco": 2220,
    "yolov9t_domain": 2221,
}
DRAWS = 1000
SEED = 2219


def groups(stems: list[str]) -> list[np.ndarray]:
    labels = [stem[:3] for stem in stems]
    unique = sorted(set(labels))
    return [np.flatnonzero(np.asarray(labels) == group) for group in unique]


def main() -> None:
    output = {
        "policy": {
            "top_k": TOP_K,
            "visible_margin": VISIBLE_MARGIN,
            "candidates": ["vis", "cdf", "pia", "spg"],
        },
        "bootstrap": {
            "draws": DRAWS,
            "base_seed": SEED,
            "model_seeds": MODEL_SEEDS,
            "interval": "percentile 95%",
            "unit": "LLVIP scene groups",
        },
        "models": {},
    }

    for model_name in MODELS:
        rng = np.random.default_rng(MODEL_SEEDS[model_name])
        states = {}
        filename_kind = {
            "vis": "vis",
            "cdf": "cdf",
            "pia": "piafusion",
            "spg": "spgfusion",
        }
        for candidate in ("vis", "cdf", "pia", "spg"):
            path = (
                MODEL_DIRS[model_name]
                / f"{model_name}__{filename_kind[candidate]}.npz"
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            states[candidate] = load(path)

        reference = states["vis"]
        if any(
            state["stems"] != reference["stems"]
            for state in states.values()
        ):
            raise ValueError(f"stem mismatch for {model_name}")
        n = reference["n"]
        image_indices = np.arange(n)
        scene_groups = groups(reference["stems"])

        shielded_names, unshielded_names = _decisions(states)
        shielded = _policy_stats(states, shielded_names)
        unshielded = _policy_stats(states, unshielded_names)

        ap = {
            "visible": _ap(reference, image_indices),
            "always_cdf": _ap(states["cdf"], image_indices),
            "no_shield": _ap(unshielded, image_indices),
            "csecf": _ap(shielded, image_indices),
        }
        image_draws = {
            "csecf_minus_visible": [],
            "csecf_minus_cdf": [],
            "csecf_minus_no_shield": [],
        }
        group_draws = {
            "csecf_minus_visible": [],
            "csecf_minus_cdf": [],
            "csecf_minus_no_shield": [],
        }
        per_group = []
        for index, indices in enumerate(scene_groups):
            values = {
                "visible": _ap(reference, indices),
                "always_cdf": _ap(states["cdf"], indices),
                "no_shield": _ap(unshielded, indices),
                "csecf": _ap(shielded, indices),
            }
            per_group.append(
                {
                    "scene": reference["stems"][indices[0]][:3],
                    "n_images": int(len(indices)),
                    "ap50": values,
                    "csecf_minus_visible": (
                        values["csecf"] - values["visible"]
                    ),
                    "no_shield_minus_visible": (
                        values["no_shield"] - values["visible"]
                    ),
                }
            )
        for _ in range(DRAWS):
            for unit, indices in (
                ("image", rng.integers(0, n, size=n)),
                ("group", _draw_by_group(rng, scene_groups)),
            ):
                values = {
                    "csecf_minus_visible": (
                        _ap(shielded, indices) - _ap(reference, indices)
                    ),
                    "csecf_minus_cdf": (
                        _ap(shielded, indices) - _ap(states["cdf"], indices)
                    ),
                    "csecf_minus_no_shield": (
                        _ap(shielded, indices) - _ap(unshielded, indices)
                    ),
                }
                destination = image_draws if unit == "image" else group_draws
                for key, value in values.items():
                    destination[key].append(value)

        output["models"][model_name] = {
            "ap50": ap,
            "delta": {
                "csecf_minus_visible": ap["csecf"] - ap["visible"],
                "csecf_minus_cdf": ap["csecf"] - ap["always_cdf"],
                "csecf_minus_no_shield": ap["csecf"] - ap["no_shield"],
            },
            "image_ci95": {
                key: _interval(np.asarray(values))
                for key, values in image_draws.items()
            },
            "group_ci95": {
                key: _interval(np.asarray(values))
                for key, values in group_draws.items()
            },
            "per_group": per_group,
            "worst_group": {
                "csecf_minus_visible": min(
                    row["csecf_minus_visible"] for row in per_group
                ),
                "no_shield_minus_visible": min(
                    row["no_shield_minus_visible"] for row in per_group
                ),
            },
            "selected_counts": {
                candidate: shielded_names.count(candidate)
                for candidate in ("vis", "cdf", "pia", "spg")
            },
            "unshielded_counts": {
                candidate: unshielded_names.count(candidate)
                for candidate in ("vis", "cdf", "pia", "spg")
            },
        }
        print(
            f"{model_name}: visible={ap['visible']:.4f} "
            f"CDF={ap['always_cdf']:.4f} no-shield={ap['no_shield']:.4f} "
            f"CS-ECF={ap['csecf']:.4f}"
        )
        print(
            "  CS-ECF-visible "
            f"{output['models'][model_name]['delta']['csecf_minus_visible']:+.4f} "
            f"imageCI={output['models'][model_name]['image_ci95']['csecf_minus_visible']} "
            f"groupCI={output['models'][model_name]['group_ci95']['csecf_minus_visible']}"
        )
        print(
            "  CS-ECF-CDF "
            f"{output['models'][model_name]['delta']['csecf_minus_cdf']:+.4f} "
            f"imageCI={output['models'][model_name]['image_ci95']['csecf_minus_cdf']} "
            f"groupCI={output['models'][model_name]['group_ci95']['csecf_minus_cdf']}"
        )

    OUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
