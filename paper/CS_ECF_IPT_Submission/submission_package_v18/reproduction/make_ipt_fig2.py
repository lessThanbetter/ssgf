"""Figure 2: frozen CS-ECF inference path and selected-input composition."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

from ipt_figure_common import (
    COLORS,
    add_panel_label,
    apply_style,
    arrow_patch,
    card,
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


def _lane(ax, x, width, label):
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, 4),
            width,
            86,
            boxstyle="round,pad=0.15,rounding_size=2.2",
            facecolor=COLORS["panel"],
            edgecolor="#E4E9EE",
            linewidth=0.6,
            zorder=0,
        )
    )
    ax.text(
        x + width / 2,
        88.5,
        label.upper(),
        ha="center",
        va="center",
        fontsize=5.8,
        fontweight="bold",
        color=COLORS["mid"],
    )


def _shield(ax, x, y, width, height):
    points = np.asarray(
        [
            [x, y + height],
            [x + width, y + height],
            [x + width * 0.92, y + height * 0.36],
            [x + width * 0.5, y],
            [x + width * 0.08, y + height * 0.36],
        ]
    )
    patch = patches.Polygon(
        points,
        closed=True,
        facecolor="#DCEAF6",
        edgecolor=COLORS["csecf"],
        linewidth=0.85,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * 0.54,
        "1.5×\nmargin\ngate",
        ha="center",
        va="center",
        fontsize=6.3,
        color=COLORS["csecf"],
        fontweight="bold",
        linespacing=0.92,
        zorder=4,
    )


def main() -> None:
    apply_style()
    data = load_json("confidence_shield_three_dataset_summary.json")

    counts = {}
    source_rows = []
    for dataset, display in DATASETS:
        totals = {name: 0 for name, _, _ in CANDIDATES}
        for model in data[dataset]["models"].values():
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

    fig = plt.figure(figsize=(6.35, 5.65))
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[2.65, 1.0],
        hspace=0.34,
        left=0.035,
        right=0.975,
        top=0.94,
        bottom=0.08,
    )
    ax = fig.add_subplot(grid[0])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    _lane(ax, 2, 34, "Candidate generation")
    _lane(ax, 39, 27, "Target evaluator")
    _lane(ax, 69, 29, "Safety decision")

    card(
        ax,
        4.5,
        59,
        10,
        17,
        "Visible",
        "RGB scene",
        facecolor="#E7EBEF",
        edgecolor="#D4DAE0",
        accent=COLORS["visible"],
        title_size=6.7,
        subtitle_size=5.8,
    )
    card(
        ax,
        4.5,
        37,
        10,
        17,
        "Infrared",
        "thermal scene",
        facecolor="#F8E9DA",
        edgecolor="#E9D7C5",
        accent="#C6783C",
        title_size=6.7,
        subtitle_size=5.8,
    )
    ax.plot([15.4, 17.2], [57.5, 57.5], color=COLORS["line"], linewidth=0.7)
    ax.plot([15.4, 17.2], [39.5, 39.5], color=COLORS["line"], linewidth=0.7)
    ax.plot([17.2, 17.2], [39.5, 57.5], color=COLORS["line"], linewidth=0.7)
    library_x, library_y, library_w, library_h = 18.0, 37.0, 16.4, 42.0
    ax.add_patch(
        patches.FancyBboxPatch(
            (library_x, library_y),
            library_w,
            library_h,
            boxstyle="round,pad=0.15,rounding_size=1.7",
            facecolor="white",
            edgecolor="#DDE4EA",
            linewidth=0.6,
            zorder=2,
        )
    )
    chip_positions = [
        (19.2, 66.0),
        (27.0, 66.0),
        (19.2, 47.5),
        (27.0, 47.5),
    ]
    for (candidate, label, color), (x, y) in zip(CANDIDATES, chip_positions):
        card(
            ax,
            x,
            y,
            6.4,
            11.0,
            label,
            facecolor=color,
            edgecolor="white",
            titlecolor="white" if candidate != "vis" else "white",
            title_size=5.4,
            shadow=False,
        )
    ax.text(
        library_x + library_w / 2,
        library_y + 2.4,
        "four frozen candidates",
        ha="center",
        va="center",
        fontsize=5.5,
        color=COLORS["mid"],
    )

    arrow_patch(ax, (14.9, 67.5), (20.0, 70.5), color=COLORS["cdf"], linewidth=0.8)
    arrow_patch(ax, (14.9, 45.5), (20.0, 52.0), color="#C6783C", linewidth=0.8)

    card(
        ax,
        42.0,
        43.0,
        20.5,
        35.0,
        "Target detector  D",
        "one forward pass\nper candidate",
        facecolor="#EAF1F8",
        edgecolor="#B7CCDD",
        accent=COLORS["cdf"],
        titlecolor=COLORS["ink"],
        subtitlecolor=COLORS["mid"],
        title_size=6.8,
        subtitle_size=5.9,
    )
    for column in range(4):
        ax.add_patch(
            patches.Rectangle(
                (45.8 + column * 3.2, 46.2),
                2.5,
                3.6,
                facecolor="#6889A3",
                edgecolor="#AFC7D8",
                linewidth=0.35,
                alpha=0.9,
                zorder=5,
            )
        )
    arrow_patch(ax, (34.4, 58.0), (42.0, 60.5), color=COLORS["ink"], linewidth=0.9)

    card(
        ax,
        44.3,
        17.5,
        16.0,
        17.5,
        "Confidence score",
        "mean of top 5\ndetections",
        facecolor="#F5F8FB",
        edgecolor="#DCE5EC",
        accent=COLORS["cdf"],
        title_size=6.6,
        subtitle_size=5.7,
    )
    arrow_patch(ax, (52.2, 43.0), (52.2, 35.0), color=COLORS["ink"], linewidth=0.8)

    _shield(ax, 73.5, 55.0, 11.5, 20.0)
    arrow_patch(
        ax,
        (60.3, 26.0),
        (73.2, 62.0),
        color=COLORS["cdf"],
        linewidth=0.9,
        connectionstyle="angle3,angleA=0,angleB=90",
    )

    card(
        ax,
        88.0,
        55.0,
        8.8,
        20.0,
        "Selected\ninput",
        "final\ndetection",
        facecolor=COLORS["csecf"],
        edgecolor=COLORS["csecf"],
        titlecolor="white",
        subtitlecolor="#DDE8F1",
        title_size=6.2,
        subtitle_size=5.6,
    )
    arrow_patch(ax, (85.0, 65.0), (88.0, 65.0), color=COLORS["csecf"], linewidth=0.9)
    arrow_patch(
        ax,
        (73.2, 60.2),
        (82.0, 39.0),
        color=COLORS["visible"],
        linewidth=0.8,
        connectionstyle="arc3,rad=-0.25",
        linestyle=(0, (2, 1.5)),
    )
    arrow_patch(ax, (82.0, 39.0), (88.0, 55.8), color=COLORS["visible"], linewidth=0.8, connectionstyle="arc3,rad=0.2")
    ax.text(80.0, 35.2, "visible\nfallback", ha="center", va="top", fontsize=5.6, color=COLORS["visible"])

    ax.text(
        42.0,
        38.1,
        "LABEL-FREE",
        ha="left",
        va="center",
        fontsize=5.5,
        fontweight="bold",
        color=COLORS["cdf"],
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#EAF2F8", "edgecolor": "none"},
    )
    ax.text(
        3.0,
        12.5,
        "A fused candidate is used only when its score clears 1.5× the visible score.",
        ha="left",
        va="center",
        fontsize=5.8,
        color=COLORS["ink"],
    )
    ax.text(
        3.0,
        7.0,
        "Otherwise the registered visible image is retained.",
        ha="left",
        va="center",
        fontsize=5.8,
        color=COLORS["mid"],
    )
    ax.text(
        0.045,
        1.025,
        "Frozen CS-ECF inference pathway",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.4,
        color=COLORS["ink"],
    )
    add_panel_label(ax, "a", x=-0.02, y=1.025)

    ax2 = fig.add_subplot(grid[1])
    y = np.arange(len(DATASETS))
    left = np.zeros(len(DATASETS))
    for candidate, label, color in CANDIDATES:
        values = np.asarray(
            [counts[dataset][candidate] / sum(counts[dataset].values()) for dataset, _ in DATASETS]
        )
        ax2.barh(
            y,
            values,
            left=left,
            height=0.48,
            color=color,
            edgecolor="white",
            linewidth=0.55,
        )
        for row_index, value in enumerate(values):
            if value >= 0.075:
                ax2.text(
                    left[row_index] + value / 2,
                    y[row_index],
                    f"{100 * value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=5.8,
                    color="white" if candidate != "vis" else "white",
                    fontweight="semibold",
                )
        left += values
    for row_index, (dataset, display) in enumerate(DATASETS):
        total = sum(counts[dataset].values())
        ax2.text(
            1.012,
            y[row_index],
            f"n={total:,}",
            ha="left",
            va="center",
            fontsize=5.5,
            color=COLORS["mid"],
        )
    ax2.set_yticks(y)
    ax2.set_yticklabels([display for _, display in DATASETS], fontsize=6.4)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1.08)
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_xticklabels(["0", "25%", "50%", "75%", "100%"], fontsize=5.8)
    ax2.set_xlabel("Fraction of checkpoint–image decisions", fontsize=6.2)
    ax2.set_title("Selected-input composition", loc="left", pad=18, fontsize=7.0)
    clean_axis(ax2, grid_axis="x")
    ax2.tick_params(axis="y", length=0, pad=3)
    handles = [marker_handle(label, color, "s", 4.6) for _, label, color in CANDIDATES]
    ax2.legend(
        handles=handles,
        ncol=4,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        columnspacing=1.15,
        handletextpad=0.35,
        fontsize=5.8,
    )
    add_panel_label(ax2, "b", x=-0.12, y=1.15)

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
    save_figure(fig, "fig2_csecf_workflow", dpi=600)


if __name__ == "__main__":
    main()
