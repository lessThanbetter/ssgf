"""Figure 3: mean gain versus worst-checkpoint risk."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ipt_figure_common import (
    COLORS,
    add_panel_label,
    apply_style,
    load_json,
    save_figure,
    write_csv,
)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)
PUBLICATION_EXPORTS = (".svg", ".pdf", ".tiff", ".png")
PUBLICATION_DPI = 600


POLICIES = [
    ("always_cdf", "Always CDF", COLORS["cdf"]),
    ("always_pia", "Always PIAFusion", COLORS["pia"]),
    ("always_spg", "Always SPGFusion", COLORS["spg"]),
    ("no_shield", "No shield", COLORS["no_shield"]),
    ("cs_ecf", "CS-ECF", COLORS["csecf"]),
]


def pretty_checkpoint(name):
    dataset, checkpoint = name.split("/", 1)
    dataset_label = {
        "llvip": "LLVIP",
        "m3fd": "M3FD",
        "dronevehicle_test": "DroneVehicle",
    }[dataset]
    checkpoint_label = {
        "yolo11": "YOLO11n",
        "agnostic": "Modality-agnostic",
        "yolo26_s0": "YOLO26n seed 0",
        "yolo26_s1": "YOLO26n seed 1",
        "yolo26_s2": "YOLO26n seed 2",
        "rtdetr": "RT-DETR-l",
        "yolo8_s0": "YOLOv8n seed 0",
        "yolo8_s1": "YOLOv8n seed 1",
        "yolo26_m3fd6": "YOLO26n",
        "yolo8n_m3fd6": "YOLOv8n",
        "best": "best checkpoint",
        "last": "last checkpoint",
    }.get(checkpoint, checkpoint)
    dataset_short = {
        "LLVIP": "LLVIP",
        "M3FD": "M3FD",
        "DroneVehicle": "DV",
    }[dataset_label]
    return f"{dataset_short} {checkpoint_label}"


def main():
    apply_style()
    data = load_json("confirmatory_policy_comparison.json")
    group_summary = load_json("confidence_shield_group_hierarchical.json")
    source_rows = []
    fig = plt.figure(figsize=(7.2, 4.9))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.3], wspace=0.62)

    ax = fig.add_subplot(grid[0])
    for policy, label, color in POLICIES:
        row = data["summaries"][policy]
        x = 100 * row["dataset_balanced_mean"]
        y = 100 * row["min_gain"]
        xerr = None
        if policy == "cs_ecf":
            low, high = group_summary["aggregate"][
                "dataset_balanced_hierarchical_group_ci95"
            ]
            xerr = [
                [100 * (row["dataset_balanced_mean"] - low)],
                [100 * (high - row["dataset_balanced_mean"])],
            ]
            ax.errorbar(
                x,
                y,
                xerr=xerr,
                fmt="none",
                ecolor=COLORS["csecf"],
                capsize=3,
                linewidth=0.8,
                zorder=4,
            )
        ax.scatter(
            x,
            y,
            s=70 if policy == "cs_ecf" else 48,
            color=color,
            edgecolor="white" if policy == "cs_ecf" else COLORS["dark"],
            linewidth=0.8,
            zorder=5 if policy == "cs_ecf" else 3,
        )
        offsets = {
            "always_cdf": (-36, 10),
            "always_pia": (8, -4),
            "always_spg": (8, 4),
            "no_shield": (-38, -12),
            "cs_ecf": (7, 12),
        }
        ax.annotate(
            label,
            (x, y),
            xytext=offsets[policy],
            textcoords="offset points",
            fontsize=6.2,
            color=COLORS["dark"],
            ha="right" if policy in {"always_cdf", "no_shield"} else "left",
            annotation_clip=True,
        )
        source_rows.append(
            {
                "panel": "a",
                "policy": label,
                "dataset_balanced_mean_ap50_points": round(x, 6),
                "worst_checkpoint_ap50_points": round(y, 6),
                "mean_ci_low_points": (
                    round(100 * group_summary["aggregate"]["dataset_balanced_hierarchical_group_ci95"][0], 6)
                    if policy == "cs_ecf"
                    else ""
                ),
                "mean_ci_high_points": (
                    round(100 * group_summary["aggregate"]["dataset_balanced_hierarchical_group_ci95"][1], 6)
                    if policy == "cs_ecf"
                    else ""
                ),
            }
        )
    ax.axhline(0, color=COLORS["dark"], linewidth=0.8)
    ax.axvline(0, color=COLORS["dark"], linewidth=0.8)
    ax.set_xlim(-8.0, 4.5)
    ax.set_ylim(-31.0, 2.0)
    ax.set_xlabel("Dataset-balanced mean AP50 change (points)")
    ax.set_ylabel("Worst checkpoint AP50 change (points)")
    ax.set_title("Risk–return plane", loc="left", pad=8)
    add_panel_label(ax, "a", x=-0.18, y=1.02)

    ax2 = fig.add_subplot(grid[1])
    checkpoints = list(data["per_checkpoint"].keys())
    rows = sorted(
        checkpoints,
        key=lambda name: data["per_checkpoint"][name]["no_shield"],
    )
    no_shield = np.asarray(
        [100 * data["per_checkpoint"][name]["no_shield"] for name in rows]
    )
    csecf = np.asarray(
        [100 * data["per_checkpoint"][name]["cs_ecf"] for name in rows]
    )
    y = np.arange(len(rows))
    for index, name in enumerate(rows):
        improvement = csecf[index] >= no_shield[index]
        line_color = COLORS["gain"] if improvement else COLORS["loss"]
        ax2.plot(
            [no_shield[index], csecf[index]],
            [y[index], y[index]],
            color=line_color,
            linewidth=1.2,
            zorder=2,
        )
        source_rows.append(
            {
                "panel": "b",
                "checkpoint": name,
                "no_shield_ap50_points": round(no_shield[index], 6),
                "cs_ecf_ap50_points": round(csecf[index], 6),
                "shield_minus_no_shield_points": round(
                    csecf[index] - no_shield[index], 6
                ),
            }
        )
    ax2.scatter(
        no_shield,
        y,
        color=COLORS["no_shield"],
        s=22,
        edgecolor="white",
        linewidth=0.4,
        label="No shield",
        zorder=3,
    )
    ax2.scatter(
        csecf,
        y,
        color=COLORS["csecf"],
        s=22,
        edgecolor="white",
        linewidth=0.4,
        label="CS-ECF",
        zorder=3,
    )
    ax2.axvline(0, color=COLORS["dark"], linewidth=0.8)
    ax2.set_yticks(y)
    labels = [pretty_checkpoint(name) for name in rows]
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("AP50 change from visible (points)")
    ax2.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["no_shield"], markersize=4, label="No shield"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["csecf"], markersize=4, label="CS-ECF"),
        ],
        loc="lower right",
    )
    ax2.set_title("Paired checkpoint changes", loc="left", pad=8)
    add_panel_label(ax2, "b", x=-0.18, y=1.02)

    write_csv(
        "figure3_source.csv",
        [
            "panel",
            "policy",
            "dataset_balanced_mean_ap50_points",
            "worst_checkpoint_ap50_points",
            "mean_ci_low_points",
            "mean_ci_high_points",
            "checkpoint",
            "no_shield_ap50_points",
            "cs_ecf_ap50_points",
            "shield_minus_no_shield_points",
        ],
        source_rows,
    )
    save_figure(fig, "fig3_risk_return", dpi=PUBLICATION_DPI)


if __name__ == "__main__":
    main()
