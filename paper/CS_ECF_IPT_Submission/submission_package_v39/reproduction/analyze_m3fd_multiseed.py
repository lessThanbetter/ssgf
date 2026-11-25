"""Summarize run-to-run variability of M3FD CS-ECF contrasts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
ROOT = next(
    path
    for path in (SCRIPT_PATH.parent, *SCRIPT_PATH.parents)
    if (
        path / "scripts" / "prospective_llvip_analysis.py"
    ).is_file()
    or (
        path / "reproduction" / "budgeted_candidate_selection.py"
    ).is_file()
)
for module_dir in (
    SCRIPT_PATH.parent,
    ROOT / "scripts",
    ROOT / "analysis" / "cs_ecf",
):
    if module_dir.is_dir():
        sys.path.insert(0, str(module_dir))

import budgeted_candidate_selection as budget  # noqa: E402


MODELS = (
    "yolo26_m3fd6_s0",
    "yolo26_m3fd6_s1",
    "yolo26_m3fd6_s2",
    "yolo8n_m3fd6_s0",
    "yolo8n_m3fd6_s1",
    "yolo8n_m3fd6_s2",
)
FAMILIES = {
    "YOLO26n": MODELS[:3],
    "YOLOv8n": MODELS[3:],
}
FILE_STEMS = {
    "vis": "vis",
    "cdf": "cdf",
    "pia": "piafusion",
    "spg": "spgfusion",
}
PACKAGE_PARENT = ROOT / "paper" / "CS_ECF_IPT_Submission"
DEFAULT_PACKAGE = next(
    (
        path
        for path in (SCRIPT_PATH.parent, *SCRIPT_PATH.parents)
        if (path / "main.tex").is_file()
        and (path / "evidence").is_dir()
    ),
    None,
)
if DEFAULT_PACKAGE is None:
    package_candidates = sorted(
        PACKAGE_PARENT.glob("submission_package_v*"),
        key=lambda path: int(path.name.rsplit("v", 1)[1]),
    )
    if not package_candidates:
        raise RuntimeError("No submission package containing evidence was found")
    DEFAULT_PACKAGE = package_candidates[-1]
DEFAULT_SCORES = (
    DEFAULT_PACKAGE
    / "evidence"
    / "m3fd_multiseed"
    / "test_scores"
)
if not DEFAULT_SCORES.is_dir():
    DEFAULT_SCORES = (
        ROOT
        / "results"
        / "selective_fusion"
        / "m3fd_multiseed_test_scores"
    )
DEFAULT_OUTPUT = (
    DEFAULT_PACKAGE / "evidence" / "m3fd_multiseed"
    if SCRIPT_PATH.parent.name == "reproduction"
    else ROOT / "analysis" / "cs_ecf" / "results" / "m3fd_multiseed"
)


def load_states(score_root: Path, model: str) -> dict[str, dict]:
    return {
        candidate: budget.load(
            score_root / f"{model}__{filename}.npz"
        )
        for candidate, filename in FILE_STEMS.items()
    }


def group_indices(stems: list[str]) -> list[np.ndarray]:
    labels = np.asarray([stem[:2] for stem in stems])
    return [
        np.flatnonzero(labels == group)
        for group in sorted(set(labels))
    ]


def evaluate_model(score_root: Path, model: str) -> dict:
    states = load_states(score_root, model)
    indices = np.arange(int(states["vis"]["n"]))
    groups = group_indices(states["vis"]["stems"])
    visible_ap = budget.pooled_ap50(states["vis"], indices)
    policies = {
        "full": budget.Policy(budget.CANDIDATES, budget.FROZEN_MARGIN),
        "selected": budget.Policy(("vis", "pia"), budget.FROZEN_MARGIN),
        "no_shield": budget.Policy(budget.CANDIDATES, 0.0),
        "always_cdf": budget.Policy(("cdf",), 1.0),
    }
    result = {
        "model": model,
        "visible_ap50": visible_ap,
        "groups": len(groups),
    }
    for name, policy in policies.items():
        choices = budget.candidate_choices(states, policy)
        stats = budget.selected_stats(states, choices)
        pooled_ap = budget.pooled_ap50(stats, indices)
        group_gains = [
            budget.pooled_ap50(stats, group)
            - budget.pooled_ap50(states["vis"], group)
            for group in groups
        ]
        result[name] = {
            "pooled_ap50": pooled_ap,
            "pooled_gain_vs_visible": pooled_ap - visible_ap,
            "equal_group_gain_vs_visible": float(np.mean(group_gains)),
            "group_gains": [float(value) for value in group_gains],
            "positive_groups": int(sum(value > 0 for value in group_gains)),
            "switch_fraction": float(np.mean([
                choice != "vis" for choice in choices
            ])),
        }
    return result


def summarize_family(rows: list[dict], policy: str) -> dict:
    pooled = np.asarray([
        row[policy]["pooled_gain_vs_visible"] for row in rows
    ])
    equal = np.asarray([
        row[policy]["equal_group_gain_vs_visible"] for row in rows
    ])
    return {
        "runs": len(rows),
        "pooled_mean": float(pooled.mean()),
        "pooled_sd": float(pooled.std(ddof=1)),
        "pooled_min": float(pooled.min()),
        "pooled_max": float(pooled.max()),
        "equal_group_mean": float(equal.mean()),
        "equal_group_sd": float(equal.std(ddof=1)),
        "equal_group_min": float(equal.min()),
        "equal_group_max": float(equal.max()),
        "positive_runs": int(sum(row[policy]["pooled_gain_vs_visible"] > 0
                                 for row in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores",
        type=Path,
        default=DEFAULT_SCORES,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = parser.parse_args()
    rows = [
        evaluate_model(args.scores, model)
        for model in MODELS
    ]
    families = {
        family: {
            model: next(row for row in rows if row["model"] == model)
            for model in models
        }
        for family, models in FAMILIES.items()
    }
    policy_summaries = {
        family: {
            policy: summarize_family(list(family_rows.values()), policy)
            for policy in ("full", "selected", "no_shield", "always_cdf")
        }
        for family, family_rows in families.items()
    }
    all_rows = rows
    payload = {
        "scope": "M3FD held-out test, 200 images, two frame groups",
        "matching": "all-at-once maximum-cardinality matching",
        "top_k": budget.TOP_K,
        "margin": budget.FROZEN_MARGIN,
        "runs": rows,
        "families": policy_summaries,
        "all_runs": {
            policy: summarize_family(all_rows, policy)
            for policy in ("full", "selected", "no_shield", "always_cdf")
        },
        "note": (
            "Seed variance is conditional on the fixed M3FD test frames "
            "and two training recipes. Range and SD describe observed runs, "
            "not a population confidence guarantee."
        ),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "analysis.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.out / "runs.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        fields = (
            "model",
            "visible_ap50",
            "full_pooled_gain",
            "full_equal_group_gain",
            "selected_pooled_gain",
            "selected_equal_group_gain",
            "no_shield_pooled_gain",
            "always_cdf_pooled_gain",
            "full_switch_fraction",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "model": row["model"],
                "visible_ap50": row["visible_ap50"],
                "full_pooled_gain": row["full"]["pooled_gain_vs_visible"],
                "full_equal_group_gain": row["full"][
                    "equal_group_gain_vs_visible"
                ],
                "selected_pooled_gain": row["selected"][
                    "pooled_gain_vs_visible"
                ],
                "selected_equal_group_gain": row["selected"][
                    "equal_group_gain_vs_visible"
                ],
                "no_shield_pooled_gain": row["no_shield"][
                    "pooled_gain_vs_visible"
                ],
                "always_cdf_pooled_gain": row["always_cdf"][
                    "pooled_gain_vs_visible"
                ],
                "full_switch_fraction": row["full"]["switch_fraction"],
            })
    print(json.dumps({
        "families": {
            family: {
                policy: summaries[policy]
                for policy in ("full", "selected")
            }
            for family, summaries in policy_summaries.items()
        },
        "all_runs": {
            policy: payload["all_runs"][policy]
            for policy in ("full", "selected")
        },
    }, indent=2))


if __name__ == "__main__":
    main()
