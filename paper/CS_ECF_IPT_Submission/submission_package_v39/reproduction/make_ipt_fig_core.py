"""Build the compact three-panel core evidence figure."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 6.3,
        "axes.linewidth": 0.6,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.unicode_minus": False,
    }
)

DATASET_COLORS = {
    "LLVIP": "#557FA8",
    "M3FD": "#3F8E89",
    "DroneVehicle": "#C07A57",
}

RETROSPECTIVE_ROWS = (
    ("llvip/yolo11", "LLVIP V11", "llvip"),
    ("llvip/agnostic", "LLVIP Agn", "llvip"),
    ("llvip/yolo26_s0", "LLVIP 26-0", "llvip"),
    ("llvip/rtdetr", "LLVIP RT", "llvip"),
    ("llvip/yolo26_s1", "LLVIP 26-1", "llvip"),
    ("llvip/yolo26_s2", "LLVIP 26-2", "llvip"),
    ("llvip/yolo8_s0", "LLVIP 8-0", "llvip"),
    ("llvip/yolo8_s1", "LLVIP 8-1", "llvip"),
    ("m3fd/yolo26_m3fd6", "M3FD 26", "m3fd"),
    ("m3fd/yolo8n_m3fd6", "M3FD 8", "m3fd"),
    ("dronevehicle_test/best", "DV best", "dronevehicle_test"),
    ("dronevehicle_test/last", "DV last", "dronevehicle_test"),
)

RETROSPECTIVE_POLICIES = (
    ("always_cdf", "CDF"),
    ("always_pia", "PIA"),
    ("always_spg", "SPG"),
)


def load_retrospective(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["gain"] = float(row["equal_group_gain_ap50_points"])
        row["low"] = float(row["ci_low_ap50_points"])
        row["high"] = float(row["ci_high_ap50_points"])
        row["dataset"] = row["checkpoint"].split("/", 1)[0]
    return rows


def prospective_contrast(prospective: dict, name: str) -> tuple[float, float, float]:
    if "llvip" in prospective:
        block = prospective["llvip"]["primary_six_last_checkpoints"][
            "comparisons"
        ][name]
    else:
        block = prospective["confirmation"]["comparisons"][name]
    low, high = block["conditional_sequence_by_run_interval"]
    return (
        100.0 * float(block["summary"]["sequence_equal_point"]),
        100.0 * float(low),
        100.0 * float(high),
    )


def vedai_contrast(vedai: dict, name: str) -> tuple[float, float, float]:
    block = (
        vedai["analysis"]["aggregate"][name]
        if "analysis" in vedai
        else vedai["aggregate"][name]
    )
    low, high = block["conditional_run_by_block_interval"]
    return 100.0 * float(block["equal_run_mean"]), 100.0 * float(low), 100.0 * float(high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--prospective", type=Path, required=True)
    parser.add_argument("--vedai", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-out", type=Path, required=True)
    args = parser.parse_args()

    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    prospective = json.loads(args.prospective.read_text(encoding="utf-8"))
    vedai = json.loads(args.vedai.read_text(encoding="utf-8"))

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.09, 3.25),
        gridspec_kw={"width_ratios": (1.48, 1.08, 0.76), "wspace": 0.48},
    )

    ax = axes[0]
    matrix = np.asarray([
        [
            100.0 * float(policy["per_checkpoint"][key][name])
            for name, _ in RETROSPECTIVE_POLICIES
        ]
        for key, _, _ in RETROSPECTIVE_ROWS
    ])
    limit = max(2.0, float(np.max(np.abs(matrix))))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "fixed_fusion_gain",
        ["#A64D50", "#E6B8B2", "#FBF8F4", "#B8D5E7", "#3E779D"],
    )
    image = ax.imshow(
        matrix,
        cmap=cmap,
        norm=mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        aspect="auto",
        interpolation="nearest",
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            red, green, blue, _ = image.cmap(image.norm(value))
            luminance = (
                0.299 * red + 0.587 * green + 0.114 * blue
            )
            ax.text(
                column,
                row,
                f"{value:+.1f}",
                ha="center",
                va="center",
                fontsize=5.4,
                color="white" if luminance < 0.48 else "#333333",
            )
    ax.set_xticks(range(len(RETROSPECTIVE_POLICIES)))
    ax.set_xticklabels(
        [label for _, label in RETROSPECTIVE_POLICIES],
        fontsize=6.0,
    )
    ax.set_yticks(range(len(RETROSPECTIVE_ROWS)))
    ax.set_yticklabels(
        [label for _, label, _ in RETROSPECTIVE_ROWS],
        fontsize=5.5,
    )
    ax.tick_params(axis="both", length=0)
    ax.set_title(
        "Fixed fusion minus visible\nAP50 difference across evaluators",
        loc="left",
        fontsize=6.5,
        pad=7,
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.055, pad=0.025)
    colorbar.ax.tick_params(labelsize=5.1, length=2)
    colorbar.outline.set_linewidth(0.35)
    colorbar.set_label("AP50 points", fontsize=5.4, labelpad=2)

    ax = axes[1]
    llvip = {
        "Visible": prospective_contrast(prospective, "csecf_minus_visible"),
        "Always CDF": prospective_contrast(prospective, "csecf_minus_cdf"),
        "No shield": prospective_contrast(prospective, "csecf_minus_no_shield"),
    }
    aerial = {
        "Visible": vedai_contrast(vedai, "csecf_minus_visible"),
        "Always CDF": vedai_contrast(vedai, "csecf_minus_always_cdf"),
        "No shield": vedai_contrast(vedai, "csecf_minus_no_shield"),
    }
    comparator_colors = {
        "Visible": "#737373",
        "Always CDF": "#416C94",
        "No shield": "#D17A32",
    }
    offsets = {"LLVIP": -0.17, "VEDAI": 0.17}
    for dataset_name, values in (("LLVIP", llvip), ("VEDAI", aerial)):
        for row_index, comparator in enumerate(("Visible", "Always CDF", "No shield")):
            point, low, high = values[comparator]
            y_value = row_index + offsets[dataset_name]
            ax.plot([low, high], [y_value, y_value], color=comparator_colors[comparator], lw=1.0, alpha=0.7)
            ax.scatter(point, y_value, s=18, color=comparator_colors[comparator], marker="o" if dataset_name == "LLVIP" else "D", zorder=3)
    ax.axvline(0, color="#333333", lw=0.65)
    ax.set_yticks(range(3))
    ax.set_yticklabels(["Visible", "Always CDF", "No shield"], fontsize=5.8)
    ax.set_xlabel("CS-ECF minus comparator (AP50 pp)", fontsize=5.8)
    ax.set_title(
        "Held-out contrasts\ncircle: LLVIP  diamond: VEDAI",
        loc="left",
        fontsize=6.3,
        pad=5,
    )
    ax.tick_params(axis="x", labelsize=5.3)
    ax.grid(axis="x", color="#E5E5E5", lw=0.45)

    ax = axes[2]
    groups = ("LLVIP", "VEDAI")
    no_shield = (3.270, 20.73)
    shield = (0.451, 7.51)
    x = np.arange(2)
    width = 0.28
    ax.bar(x - width / 2, no_shield, width=width, color="#D17A32", label="No shield")
    ax.bar(x + width / 2, shield, width=width, color="#3F8E89", label="CS-ECF")
    for index in range(2):
        ax.annotate(
            "",
            xy=(x[index] + width / 2, shield[index]),
            xytext=(x[index] - width / 2, no_shield[index]),
            arrowprops={"arrowstyle": "-|>", "color": "#555555", "lw": 0.7},
        )
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=5.8)
    ax.set_ylabel("Observed worst loss (AP50 pp)", fontsize=5.8)
    ax.set_title("Tail exposure", loc="left", fontsize=6.7, pad=7)
    ax.tick_params(axis="y", labelsize=5.3)
    ax.legend(frameon=False, fontsize=5.1, loc="upper right")
    ax.grid(axis="y", color="#E5E5E5", lw=0.45)

    fig.subplots_adjust(
        left=0.12,
        right=0.99,
        top=0.80,
        bottom=0.16,
        wspace=0.48,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for checkpoint, _, _ in RETROSPECTIVE_ROWS:
        for policy_name, policy_label in RETROSPECTIVE_POLICIES:
            rows.append(
                {
                    "panel": "a",
                    "scope": checkpoint,
                    "comparator": policy_label,
                    "point_ap50_points": round(
                        100.0
                        * float(policy["per_checkpoint"][checkpoint][policy_name]),
                        6,
                    ),
                    "low_ap50_points": "",
                    "high_ap50_points": "",
                    "no_shield_worst_ap50_points": "",
                    "csecf_worst_ap50_points": "",
                }
            )
    for scope, values in (("LLVIP", llvip), ("VEDAI", aerial)):
        for comparator, (point, low, high) in values.items():
            rows.append(
                {
                    "panel": "b",
                    "scope": scope,
                    "comparator": comparator,
                    "point_ap50_points": point,
                    "low_ap50_points": low,
                    "high_ap50_points": high,
                    "no_shield_worst_ap50_points": "",
                    "csecf_worst_ap50_points": "",
                }
            )
    for scope, no_shield_loss, shield_loss in zip(groups, no_shield, shield):
        rows.append(
            {
                "panel": "c",
                "scope": scope,
                "comparator": "No shield vs CS-ECF",
                "point_ap50_points": "",
                "low_ap50_points": "",
                "high_ap50_points": "",
                "no_shield_worst_ap50_points": no_shield_loss,
                "csecf_worst_ap50_points": shield_loss,
            }
        )
    args.source_out.parent.mkdir(parents=True, exist_ok=True)
    with args.source_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
