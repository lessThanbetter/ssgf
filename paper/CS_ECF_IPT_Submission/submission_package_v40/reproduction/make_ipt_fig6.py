"""Figure 4: equal-group checkpoint contrasts and resampling intervals."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

from ipt_figure_common import (
    COLORS,
    add_panel_label,
    apply_style,
    clean_axis,
    load_json,
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


CHECKPOINTS = [
    ("llvip/yolo11", "LLVIP · YOLO11n"),
    ("llvip/agnostic", "LLVIP · Modality-agnostic"),
    ("llvip/yolo26_s0", "LLVIP · YOLO26n s0"),
    ("llvip/rtdetr", "LLVIP · RT-DETR-l"),
    ("llvip/yolo26_s1", "LLVIP · YOLO26n s1"),
    ("llvip/yolo26_s2", "LLVIP · YOLO26n s2"),
    ("llvip/yolo8_s0", "LLVIP · YOLOv8n s0"),
    ("llvip/yolo8_s1", "LLVIP · YOLOv8n s1"),
    ("m3fd/yolo26_m3fd6", "M3FD · YOLO26n"),
    ("m3fd/yolo8n_m3fd6", "M3FD · YOLOv8n"),
    ("dronevehicle_test/best", "DV · OBB best"),
    ("dronevehicle_test/last", "DV · OBB last"),
]


def main() -> None:
    apply_style()
    data = load_json("major_revision_statistics.json")
    source_rows = []
    positive_count = sum(
        data["checkpoints"][name]["visible_delta"] > 0
        for name, _ in CHECKPOINTS
    )
    excluded_zero_count = sum(
        data["checkpoints"][name]["visible_group_ci95"][0] > 0
        or data["checkpoints"][name]["visible_group_ci95"][1] < 0
        for name, _ in CHECKPOINTS
    )

    fig = plt.figure(figsize=(6.35, 4.85))
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[2.45, 0.86],
        wspace=0.30,
        left=0.19,
        right=0.965,
        top=0.90,
        bottom=0.13,
    )

    ax = fig.add_subplot(grid[0])
    y = np.arange(len(CHECKPOINTS))
    for index, (name, label) in enumerate(CHECKPOINTS):
        row = data["checkpoints"][name]
        gain = 100 * row["visible_delta"]
        low, high = [100 * value for value in row["visible_group_ci95"]]
        color = COLORS["gain"] if gain >= 0 else COLORS["loss"]
        ax.plot([low, high], [index, index], color=color, linewidth=1.0, alpha=0.85)
        ax.scatter(
            gain,
            index,
            s=32,
            marker="o",
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
        )
        ax.text(
            high + 0.12,
            index,
            f"{gain:+.2f}",
            ha="left",
            va="center",
            fontsize=5.5,
            color=color,
            fontweight="semibold",
        )
        source_rows.append(
            {
                "checkpoint": name,
                "display_checkpoint": label,
                "equal_group_gain_ap50_points": round(gain, 6),
                "ci_low_ap50_points": round(low, 6),
                "ci_high_ap50_points": round(high, 6),
            }
        )

    for boundary in (7.5, 9.5):
        ax.axhline(boundary, color="#DCE2E8", linewidth=0.55)
    ax.axvline(0, color=COLORS["ink"], linewidth=0.75)
    ax.set_yticks(y)
    ax.set_yticklabels([label for _, label in CHECKPOINTS], fontsize=5.7)
    ax.invert_yaxis()
    ax.set_xlim(-2.0, 5.5)
    ax.set_xticks([-2, -1, 0, 1, 2, 3, 4, 5])
    ax.set_xlabel("CS-ECF minus visible equal-group AP50 (points)", fontsize=6.1)
    ax.set_title(
        "Equal-group checkpoint contrasts and intervals",
        loc="left",
        pad=12,
        fontsize=6.9,
    )
    clean_axis(ax, grid_axis="x")
    ax.tick_params(axis="y", length=0, pad=3)
    add_panel_label(ax, "a", x=-0.28, y=1.02)

    ax2 = fig.add_subplot(grid[1])
    ax2.axis("off")
    ax2.add_patch(
        patches.FancyBboxPatch(
            (0.02, 0.69),
            0.96,
            0.23,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            facecolor="#EEF4F9",
            edgecolor="#D7E3ED",
            linewidth=0.6,
        )
    )
    ax2.text(
        0.07,
        0.855,
        f"{positive_count} / {len(CHECKPOINTS)}",
        fontsize=14,
        fontweight="bold",
        color=COLORS["csecf"],
        va="center",
    )
    ax2.text(0.56, 0.865, "positive\ncheckpoints", fontsize=5.8, va="center", color=COLORS["ink"])
    ax2.text(
        0.07,
        0.745,
        "nine positive point estimates;\nno independent sign-test claim",
        fontsize=5.2,
        color=COLORS["mid"],
        va="center",
    )

    rows = [
        ("Positive point estimates", positive_count, COLORS["gain"]),
        ("Intervals exclude zero", excluded_zero_count, COLORS["csecf"]),
    ]
    for index, (label, count, color) in enumerate(rows):
        ypos = 0.55 - 0.15 * index
        ax2.text(0.04, ypos + 0.055, label, fontsize=5.4, color=COLORS["mid"])
        ax2.add_patch(
            patches.Rectangle(
                (0.04, ypos),
                0.92,
                0.036,
                facecolor="#EDF0F3",
                edgecolor="none",
            )
        )
        if count:
            ax2.add_patch(
                patches.Rectangle(
                    (0.04, ypos),
                    0.92 * count / 12,
                    0.036,
                    facecolor=color,
                    edgecolor="none",
                )
            )
        ax2.text(0.97, ypos + 0.018, f"{count}/12", fontsize=6.0, ha="right", va="center", color=COLORS["ink"], fontweight="semibold")

    ax2.text(
        0.04,
        0.17,
        (
            f"{excluded_zero_count} intervals exclude zero;\n"
            "two share one DroneVehicle run."
        ),
        fontsize=6.1,
        color=COLORS["ink"],
        linespacing=1.15,
    )
    ax2.text(
        0.04,
        0.055,
        "Conditional aggregate:\n+0.0092 [0.0026, 0.0166]",
        fontsize=5.4,
        color=COLORS["csecf"],
    )
    ax2.text(
        0.04,
        0.005,
        "Checkpoint intervals are in Figure4_source.csv.",
        fontsize=5.0,
        color=COLORS["mid"],
    )
    ax2.text(
        0.16,
        1.01,
        "Evidence boundary",
        transform=ax2.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.8,
        color=COLORS["ink"],
    )
    add_panel_label(ax2, "b", x=0.00, y=1.01)

    write_csv(
        "Figure4_source.csv",
        [
            "checkpoint",
            "display_checkpoint",
            "equal_group_gain_ap50_points",
            "ci_low_ap50_points",
            "ci_high_ap50_points",
        ],
        source_rows,
    )
    save_figure(fig, "Figure4_checkpoint_intervals", dpi=600)


if __name__ == "__main__":
    main()
