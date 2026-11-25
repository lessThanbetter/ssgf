"""Plot candidate-library cost and frozen-policy transfer results."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import ticker  # noqa: E402
import numpy as np  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
ROOT = next(
    path
    for path in (SCRIPT_PATH.parent, *SCRIPT_PATH.parents)
    if (path / "scripts" / "figure_style.py").is_file()
)
sys.path.insert(0, str(ROOT / "scripts"))
import figure_style as style  # noqa: E402

PACKAGE_PARENT = ROOT / "paper" / "CS_ECF_IPT_Submission"
DEFAULT_PACKAGE = next(
    (
        path
        for path in (SCRIPT_PATH.parent, *SCRIPT_PATH.parents)
        if (path / "main.tex").is_file()
        and (path / "evidence").is_dir()
    ),
    max(
        PACKAGE_PARENT.glob("submission_package_v*"),
        key=lambda path: int(path.name.rsplit("v", 1)[1]),
    ),
)
DEFAULT_RESULT_DIR = (
    DEFAULT_PACKAGE / "evidence" / "budgeted_selection"
    if SCRIPT_PATH.parent.name == "reproduction"
    else ROOT
    / "analysis"
    / "cs_ecf"
    / "results"
    / "budgeted_selection"
)
DEFAULT_OUT_PREFIX = (
    DEFAULT_PACKAGE / "figures" / "FigureBudgetedSelection"
    if SCRIPT_PATH.parent.name == "reproduction"
    else DEFAULT_RESULT_DIR / "FigureBudgetedSelection"
)


def read_grid(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=DEFAULT_OUT_PREFIX,
    )
    args = parser.parse_args()

    payload = json.loads(
        (args.result_dir / "analysis.json").read_text(encoding="utf-8")
    )
    prospective_path = (
        args.result_dir.parent
        / "prospective_pia_confirmation"
        / "analysis.json"
    )
    prospective = (
        json.loads(prospective_path.read_text(encoding="utf-8"))
        if prospective_path.is_file()
        else None
    )
    validation = [
        row for row in read_grid(args.result_dir / "validation_grid.csv")
        if abs(float(row["margin"]) - 1.5) < 1e-12
    ]
    selected_name = payload["selection"]["selected_policy"]

    style.use()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.15, 2.85),
        gridspec_kw={"width_ratios": [1.08, 1.0]},
    )

    ax = axes[0]
    times = np.asarray([
        float(row["estimated_serial_ms"]) for row in validation
    ])
    mean_pp = np.asarray([float(row["mean_gain"]) * 100 for row in validation])
    worst_pp = np.asarray([
        float(row["worst_cell_gain"]) * 100 for row in validation
    ])
    for xpos, mean, worst, row in zip(times, mean_pp, worst_pp, validation):
        color = (
            style.C["red"]
            if row["policy"] == selected_name
            else style.C["green"]
            if row["feasible"] == "True"
            else style.C["grey"]
        )
        ax.vlines(xpos, worst, mean, color=style.C["lightgrey"], lw=2.2)
        ax.scatter(
            xpos,
            mean,
            s=22,
            color=color,
            edgecolor="black",
            linewidth=0.35,
            zorder=3,
        )
        ax.scatter(
            xpos,
            worst,
            marker="|",
            s=30,
            color="black",
            linewidth=0.7,
            zorder=4,
        )
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(ticker.FixedLocator([40, 100, 200, 400, 700]))
    ax.xaxis.set_major_formatter(
        ticker.FixedFormatter(["40", "100", "200", "400", "700"])
    )
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_ylabel(r"Validation AP50 gain vs visible (pp)")
    ax.set_xlabel(r"Estimated serial components (ms, log)")
    ax.set_xlim(28, 850)
    style.tidy(ax, grid="both")
    style.panel(ax, "a", text="validation")
    ax.text(
        0.03,
        0.96,
        "dots: mean\nbars: worst cell",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
    )
    selected_row = next(
        row for row in validation if row["policy"] == selected_name
    )
    ax.annotate(
        "VIS + PIA\n(2 forwards)",
        xy=(
            float(selected_row["estimated_serial_ms"]),
            float(selected_row["mean_gain"]) * 100,
        ),
        xytext=(0.10, 0.62),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "-", "lw": 0.7, "color": style.C["red"]},
        color=style.C["red"],
        fontsize=7.7,
        ha="left",
    )
    full_row = next(
        row for row in validation
        if row["candidates"] == "vis+cdf+pia+spg"
    )
    ax.annotate(
        "Full CS-ECF\n(4 forwards)",
        xy=(
            float(full_row["estimated_serial_ms"]),
            float(full_row["mean_gain"]) * 100,
        ),
        xytext=(0.55, 0.22),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "-", "lw": 0.7, "color": style.C["blue"]},
        color=style.C["blue"],
        fontsize=7.7,
        ha="left",
    )

    ax = axes[1]
    domains = [
        ("LLVIP\n080--100", "prospective"),
        ("LLVIP\n050--070", "llvip_standard_matching"),
        ("VEDAI", "vedai_standard_matching"),
        ("DroneVehicle", "dronevehicle_standard_matching"),
        ("External", "external_evaluators"),
    ]
    if prospective is None:
        domains = domains[1:]
    y = np.arange(len(domains))
    for offset, key, label, color, marker in (
        (
            -0.13,
            "candidate_minus_visible",
            "VIS + PIA",
            style.C["red"],
            "o",
        ),
        (
            0.13,
            "full_csecf_minus_visible",
            "Full CS-ECF",
            style.C["blue"],
            "s",
        ),
    ):
        means = []
        lows = []
        highs = []
        for _, section in domains:
            if section == "prospective":
                if key == "candidate_minus_visible":
                    estimate = prospective["primary"]["point"]
                    low, high = prospective["primary"][
                        "conditional_interval"
                    ]
                    means.append(float(estimate) * 100)
                    lows.append(float(low) * 100)
                    highs.append(float(high) * 100)
                else:
                    means.append(float("nan"))
                    lows.append(float("nan"))
                    highs.append(float("nan"))
                continue
            row = payload[section]["comparisons"][key]
            if "mean" in row:
                means.append(float(row["mean"]) * 100)
                low, high = row["conditional_sequence_by_run_interval"]
            else:
                means.append(float(row["equal_run_mean"]) * 100)
                low, high = row["conditional_run_by_block_interval"]
            lows.append(float(low) * 100)
            highs.append(float(high) * 100)
        ax.errorbar(
            means,
            y + offset,
            xerr=[
                np.asarray(means) - np.asarray(lows),
                np.asarray(highs) - np.asarray(means),
            ],
            fmt=marker,
            color=color,
            ecolor=color,
            ms=4.0,
            lw=0,
            elinewidth=0.9,
            capsize=2.0,
            label=label,
            zorder=3,
        )
    ax.axvline(0, color="black", lw=0.7)
    ax.set_yticks(y, [label for label, _ in domains])
    ax.invert_yaxis()
    ax.set_xlabel(r"Frozen-policy AP50 difference vs visible (pp)")
    style.tidy(ax, grid="x")
    style.panel(ax, "b", text="frozen validation")
    style.legend(ax, loc="lower right")

    fig.tight_layout(w_pad=1.5)
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png", "svg"):
        kwargs = {"dpi": 300} if suffix == "png" else {}
        fig.savefig(args.out_prefix.with_suffix(f".{suffix}"), **kwargs)
    plt.close(fig)
    print(args.out_prefix)


if __name__ == "__main__":
    main()
