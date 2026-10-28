"""Figure 3: average gain versus worst-checkpoint risk."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

from ipt_figure_common import (
    COLORS,
    MARKERS,
    add_panel_label,
    apply_style,
    clean_axis,
    load_json,
    marker_handle,
    save_figure,
    write_csv,
)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)
PUBLICATION_EXPORTS = (".svg", ".pdf", ".tiff", ".png")


POLICIES = [
    ("always_cdf", "Always CDF", COLORS["cdf"]),
    ("always_pia", "Always PIAFusion", COLORS["pia"]),
    ("always_spg", "Always SPGFusion", COLORS["spg"]),
    ("no_shield", "No shield", COLORS["no_shield"]),
    ("cs_ecf", "CS-ECF", COLORS["csecf"]),
]
DATASET_ORDER = ["llvip", "m3fd", "dronevehicle_test"]
DATASET_LABELS = {
    "llvip": "LLVIP",
    "m3fd": "M3FD",
    "dronevehicle_test": "DroneVehicle",
}
CHECKPOINT_LABELS = {
    "yolo11": "YOLO11n",
    "agnostic": "Modality-agnostic",
    "yolo26_s0": "YOLO26n s0",
    "yolo26_s1": "YOLO26n s1",
    "yolo26_s2": "YOLO26n s2",
    "rtdetr": "RT-DETR-l",
    "yolo8_s0": "YOLOv8n s0",
    "yolo8_s1": "YOLOv8n s1",
    "yolo26_m3fd6": "YOLO26n",
    "yolo8n_m3fd6": "YOLOv8n",
    "best": "best checkpoint",
    "last": "last checkpoint",
}


def main() -> None:
    apply_style()
    data = load_json("confirmatory_policy_comparison.json")
    paired = load_json("csecf_vs_cdf_paired.json")
    source_rows = []

    fig = plt.figure(figsize=(6.35, 4.9))
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.03, 1.35],
        wspace=0.38,
        left=0.10,
        right=0.97,
        top=0.90,
        bottom=0.13,
    )

    ax = fig.add_subplot(grid[0])
    ax.axhspan(-1.0, 2.0, color="#EAF3FA", zorder=0)
    ax.axhspan(-32.0, -5.0, color="#FBECE9", zorder=0)
    ax.axhline(0, color=COLORS["ink"], linewidth=0.75, zorder=2)
    ax.axvline(0, color=COLORS["ink"], linewidth=0.75, zorder=2)

    points = {}
    for policy, label, color in POLICIES:
        row = data["summaries"][policy]
        x = 100 * row["dataset_balanced_mean"]
        y = 100 * row["min_gain"]
        points[policy] = (x, y)
        if policy == "cs_ecf":
            low, high = paired["dataset_balanced"]["visible_run_cluster_by_group_ci95"]
            ax.errorbar(
                x,
                y,
                xerr=[[x - 100 * low], [100 * high - x]],
                fmt="none",
                ecolor=COLORS["csecf"],
                elinewidth=0.8,
                capsize=2.5,
                capthick=0.7,
                zorder=4,
            )
        ax.scatter(
            x,
            y,
            s=105 if policy == "cs_ecf" else 48,
            marker=MARKERS[policy],
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )

    offsets = {
        "always_cdf": (5, 8),
        "always_pia": (6, -8),
        "always_spg": (6, 3),
        "no_shield": (7, -7),
        "cs_ecf": (7, -12),
    }
    alignments = {
        "always_cdf": "left",
        "always_pia": "left",
        "always_spg": "left",
        "no_shield": "left",
        "cs_ecf": "left",
    }
    for policy, label, color in POLICIES:
        x, y = points[policy]
        ax.annotate(
            label,
            (x, y),
            xytext=offsets[policy],
            textcoords="offset points",
            ha=alignments[policy],
            va="center",
            fontsize=5.9,
            color=color if policy in {"no_shield", "cs_ecf"} else COLORS["ink"],
            fontweight="semibold" if policy in {"no_shield", "cs_ecf"} else "normal",
        )

    no_x, no_y = points["no_shield"]
    cs_x, cs_y = points["cs_ecf"]
    ax.add_patch(
        patches.FancyArrowPatch(
            (no_x, no_y),
            (cs_x, cs_y),
            arrowstyle="-|>",
            mutation_scale=7,
            connectionstyle="arc3,rad=0.18",
            linewidth=0.9,
            linestyle=(0, (2.2, 1.5)),
            color=COLORS["mid"],
            zorder=3,
        )
    )
    ax.text(
        -2.8,
        -7.4,
        "risk ↓ 5.4\nreturn ↓ 2.0",
        ha="left",
        va="top",
        fontsize=5.3,
        color=COLORS["mid"],
        linespacing=0.95,
    )
    ax.text(
        -8.7,
        1.2,
        "LOW-TAIL\nGUARD",
        ha="left",
        va="top",
        fontsize=5.1,
        color="#6F9FC4",
        fontweight="bold",
        linespacing=0.95,
    )
    ax.text(
        -8.7,
        -29.2,
        "LARGE-REGRESSION\nREGION",
        ha="left",
        va="bottom",
        fontsize=5.1,
        color="#B97669",
        fontweight="bold",
        linespacing=0.95,
    )
    ax.set_xlim(-9.0, 5.0)
    ax.set_ylim(-31.0, 2.0)
    ax.set_xticks([-8, -4, 0, 4])
    ax.set_yticks([-30, -20, -10, -5, 0])
    ax.set_xlabel("Dataset-balanced mean AP50 change (points)", fontsize=6.3)
    ax.set_ylabel("Worst-checkpoint AP50 change (points)", fontsize=6.3)
    ax.set_title("Risk–return plane", loc="left", pad=10, fontsize=7.2)
    clean_axis(ax, grid_axis="both")
    add_panel_label(ax, "a", x=-0.17, y=1.03)

    ax2 = fig.add_subplot(grid[1])
    grouped = []
    for dataset in DATASET_ORDER:
        names = [
            name
            for name in data["per_checkpoint"]
            if name.startswith(dataset + "/")
        ]
        names.sort(
            key=lambda name: data["per_checkpoint"][name]["cs_ecf"]
            - data["per_checkpoint"][name]["no_shield"],
            reverse=True,
        )
        grouped.extend((dataset, name) for name in names)

    y = np.arange(len(grouped))
    group_starts = {}
    for index, (dataset, name) in enumerate(grouped):
        group_starts.setdefault(dataset, index)
        no_shield = 100 * data["per_checkpoint"][name]["no_shield"]
        csecf = 100 * data["per_checkpoint"][name]["cs_ecf"]
        delta = csecf - no_shield
        color = COLORS["gain"] if delta >= 0 else COLORS["loss"]
        ax2.plot(
            [no_shield, csecf],
            [index, index],
            color=color,
            linewidth=1.1,
            alpha=0.9,
            zorder=2,
        )
        ax2.scatter(
            no_shield,
            index,
            s=21,
            marker="o",
            color=COLORS["no_shield"],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        ax2.scatter(
            csecf,
            index,
            s=31,
            marker="D",
            color=COLORS["csecf"],
            edgecolor="white",
            linewidth=0.45,
            zorder=4,
        )
        ax2.text(
            max(no_shield, csecf) + 0.42,
            index,
            f"{delta:+.1f}",
            ha="left",
            va="center",
            fontsize=5.25,
            color=color,
            fontweight="semibold",
        )
        source_rows.append(
            {
                "panel": "b",
                "checkpoint": name,
                "no_shield_ap50_points": round(no_shield, 6),
                "cs_ecf_ap50_points": round(csecf, 6),
                "shield_minus_no_shield_points": round(delta, 6),
            }
        )

    for policy, label, color in POLICIES:
        row = data["summaries"][policy]
        source_rows.append(
            {
                "panel": "a",
                "policy": label,
                "dataset_balanced_mean_ap50_points": round(
                    100 * row["dataset_balanced_mean"], 6
                ),
                "worst_checkpoint_ap50_points": round(100 * row["min_gain"], 6),
                "mean_ci_low_points": (
                    round(
                        100
                        * paired["dataset_balanced"][
                            "visible_run_cluster_by_group_ci95"
                        ][0],
                        6,
                    )
                    if policy == "cs_ecf"
                    else ""
                ),
                "mean_ci_high_points": (
                    round(
                        100
                        * paired["dataset_balanced"][
                            "visible_run_cluster_by_group_ci95"
                        ][1],
                        6,
                    )
                    if policy == "cs_ecf"
                    else ""
                ),
            }
        )

    previous_dataset = None
    for index, (dataset, _) in enumerate(grouped):
        if previous_dataset is not None and dataset != previous_dataset:
            ax2.axhline(index - 0.5, color="#DCE2E8", linewidth=0.55)
        previous_dataset = dataset
        if index == group_starts[dataset]:
            ax2.text(
                1.01,
                index + 0.05,
                DATASET_LABELS[dataset],
                transform=ax2.get_yaxis_transform(),
                ha="left",
                va="bottom",
                fontsize=5.0,
                color=COLORS["mid"],
                fontweight="semibold",
                clip_on=False,
            )

    ax2.axvline(0, color=COLORS["ink"], linewidth=0.75)
    ax2.set_xlim(-10.5, 11.8)
    ax2.set_ylim(len(grouped) - 0.45, -0.55)
    ax2.set_yticks(y)
    ax2.set_yticklabels(
        [CHECKPOINT_LABELS[name.split("/", 1)[1]] for _, name in grouped],
        fontsize=5.7,
    )
    ax2.set_xlabel("AP50 change from visible (points)", fontsize=6.2)
    ax2.set_title("Paired checkpoint changes", loc="left", pad=22, fontsize=7.0)
    clean_axis(ax2, grid_axis="x")
    ax2.tick_params(axis="y", length=0, pad=3)
    handles = [
        marker_handle("No shield", COLORS["no_shield"], "o", 4),
        marker_handle("CS-ECF", COLORS["csecf"], "D", 4),
    ]
    ax2.legend(
        handles=handles,
        ncol=2,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.005),
        columnspacing=1.2,
        handletextpad=0.35,
        fontsize=5.8,
    )
    add_panel_label(ax2, "b", x=-0.22, y=1.05)

    columns = [
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
    ]
    write_csv("figure3_source.csv", columns, source_rows)
    save_figure(fig, "fig3_risk_return", dpi=600)


if __name__ == "__main__":
    main()
