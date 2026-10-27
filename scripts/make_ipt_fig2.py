"""Figure 2: CS-ECF workflow and selected-input composition."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ipt_figure_common import (
    COLORS,
    add_panel_label,
    apply_style,
    arrow,
    box,
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


DATASETS = [
    ("llvip", "LLVIP"),
    ("m3fd", "M3FD"),
    ("dronevehicle_test", "DroneVehicle"),
]
CANDIDATES = [
    ("vis", "Visible", COLORS["visible"]),
    ("cdf", "CDF", COLORS["cdf"]),
    ("pia", "PIAFusion", COLORS["pia"]),
    ("spg", "SPGFusion", COLORS["spg"]),
]


def main():
    apply_style()
    data = load_json("confidence_shield_three_dataset_summary.json")
    counts = {}
    source_rows = []
    for dataset, display in DATASETS:
        totals = {candidate: 0 for candidate, _, _ in CANDIDATES}
        models = data[dataset]["models"]
        for model_name, model in models.items():
            for candidate, count in model["selected_counts"].items():
                totals[candidate] += int(count)
        total = sum(totals.values())
        counts[dataset] = totals
        for candidate, label, _ in CANDIDATES:
            source_rows.append(
                {
                    "dataset": dataset,
                    "display_dataset": display,
                    "selected_input": label,
                    "count": totals[candidate],
                    "fraction": round(totals[candidate] / total, 6),
                    "checkpoint_image_trials": total,
                }
            )

    fig = plt.figure(figsize=(7.2, 5.5))
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[2.25, 1.0],
        hspace=0.28,
    )
    ax = fig.add_subplot(grid[0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(
        ax,
        (0.01, 0.40),
        0.13,
        0.20,
        "Visible +\ninfrared pair",
        "#E8EDF3",
        fontsize=7,
    )
    candidate_x = 0.18
    candidate_width = 0.18
    candidate_height = 0.12
    candidate_y = [0.76, 0.60, 0.44, 0.28]
    for y, text, color in (
        (candidate_y[0], "Visible", COLORS["visible"]),
        (candidate_y[1], "CDF", COLORS["cdf"]),
        (candidate_y[2], "PIAFusion", COLORS["pia"]),
        (candidate_y[3], "SPGFusion", COLORS["spg"]),
    ):
        text_color = "white" if color != COLORS["neutral"] else COLORS["dark"]
        box(
            ax,
            (candidate_x, y),
            candidate_width,
            candidate_height,
            text,
            color,
            fontsize=6.2,
            textcolor=text_color,
        )
        arrow(
            ax,
            (0.14, 0.50),
            (candidate_x, y + candidate_height / 2),
        )

    box(
        ax,
        (0.41, 0.31),
        0.19,
        0.42,
        "Target\ndetector\n\nq_c =\nmean of\ntop-5\nconfidences",
        "#DCE8F5",
        fontsize=6.5,
    )
    for y in candidate_y:
        arrow(
            ax,
            (candidate_x + candidate_width, y + candidate_height / 2),
            (0.41, 0.52),
        )

    box(
        ax,
        (0.65, 0.39),
        0.18,
        0.20,
        "Safety shield\n\nuse best iff\nq_best ≥ 1.5 q_vis",
        "#E4E4F0",
        fontsize=6.5,
    )
    arrow(ax, (0.60, 0.50), (0.65, 0.49))

    box(
        ax,
        (0.87, 0.40),
        0.12,
        0.18,
        "Selected\nimage →\nfinal\ndetection",
        COLORS["csecf"],
        fontsize=6.2,
        textcolor="white",
    )
    arrow(ax, (0.83, 0.49), (0.87, 0.49))
    ax.text(
        0.01,
        0.17,
        "No labels are used for arbitration.",
        ha="left",
        va="center",
        fontsize=6.2,
        color=COLORS["dark"],
    )
    ax.text(
        0.01,
        0.09,
        "The target detector is evaluated on four candidate inputs.",
        ha="left",
        va="center",
        fontsize=6.2,
        color=COLORS["dark"],
    )
    ax.set_title("Frozen CS-ECF inference pathway", loc="left", pad=4)
    add_panel_label(ax, "a", x=-0.01, y=0.99)

    ax2 = fig.add_subplot(grid[1])
    y = np.arange(len(DATASETS))
    left = np.zeros(len(DATASETS))
    for candidate, label, color in CANDIDATES:
        values = np.asarray(
            [
                counts[dataset][candidate]
                / sum(counts[dataset].values())
                for dataset, _ in DATASETS
            ]
        )
        ax2.barh(
            y,
            values,
            left=left,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            label=label,
        )
        for row_index, value in enumerate(values):
            if value >= 0.08:
                ax2.text(
                    left[row_index] + value / 2,
                    y[row_index],
                    f"{100 * value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if candidate != "vis" else "black",
                )
        left += values
    ax2.set_yticks(y)
    ax2.set_yticklabels([display for _, display in DATASETS])
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Fraction of checkpoint–image decisions")
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax2.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.31))
    add_panel_label(ax2, "b", x=-0.12, y=1.18)

    write_csv(
        "figure2_source.csv",
        [
            "dataset",
            "display_dataset",
            "selected_input",
            "count",
            "fraction",
            "checkpoint_image_trials",
        ],
        source_rows,
    )
    save_figure(fig, "fig2_csecf_workflow", dpi=PUBLICATION_DPI)


if __name__ == "__main__":
    main()
