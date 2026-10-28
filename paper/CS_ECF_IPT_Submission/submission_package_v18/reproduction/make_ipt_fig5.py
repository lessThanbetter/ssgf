"""Figure 5: pre-freeze margin selection and post-freeze sensitivity."""

from __future__ import annotations

import numpy as np
import matplotlib.colors as mcolors
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


MARGINS = [1.2, 1.5, 2.0]
KS = [1, 3, 5, 10]


def _heat_ax(ax, matrix, title, cmap, fmt, outline_selected=True):
    image = ax.imshow(matrix, cmap=cmap, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(MARGINS)))
    ax.set_xticklabels([f"{value:.1f}" for value in MARGINS], fontsize=5.7)
    ax.set_yticks(np.arange(len(KS)))
    ax.set_yticklabels([f"K={value}" for value in KS], fontsize=5.7)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(MARGINS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(KS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for row, k_value in enumerate(KS):
        for col, margin in enumerate(MARGINS):
            value = matrix[row, col]
            rgba = image.cmap(image.norm(value))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(
                col,
                row,
                fmt.format(value),
                ha="center",
                va="center",
                fontsize=5.8,
                color="white" if luminance < 0.45 else COLORS["ink"],
                fontweight="semibold" if k_value == 5 and margin == 1.5 else "normal",
            )
    if outline_selected:
        ax.add_patch(
            patches.Rectangle(
                (1 - 0.5, 2 - 0.5),
                1,
                1,
                fill=False,
                edgecolor=COLORS["csecf"],
                linewidth=1.3,
            )
        )
    ax.set_title(title, loc="left", fontsize=6.7, pad=6)
    return image


def main() -> None:
    apply_style()
    prefreeze = load_json("prefreeze_validation_summary.json")
    statistics = load_json("major_revision_statistics.json")
    source_rows = []

    fig = plt.figure(figsize=(6.35, 3.35))
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.25, 1.0, 1.0],
        wspace=0.42,
        left=0.08,
        right=0.97,
        top=0.88,
        bottom=0.19,
    )

    ax = fig.add_subplot(grid[0])
    margin_values = np.asarray([row["margin"] for row in prefreeze["rows"]])
    balanced = np.asarray([100 * row["balanced_gain"] for row in prefreeze["rows"]])
    worst = np.asarray([100 * row["worst_evaluator_gain"] for row in prefreeze["rows"]])
    feasible = worst >= -0.05
    ax.axhspan(-0.05, 0.2, color="#EAF3FA", zorder=0)
    ax.plot(
        margin_values,
        balanced,
        color=COLORS["cdf"],
        marker="o",
        markersize=3.7,
        linewidth=1.1,
        label="Balanced gain",
        zorder=4,
    )
    ax.plot(
        margin_values,
        worst,
        color=COLORS["no_shield"],
        marker="D",
        markersize=3.4,
        linewidth=1.0,
        label="Worst evaluator",
        zorder=4,
    )
    ax.axhline(-0.05, color=COLORS["loss"], linewidth=0.75, linestyle=(0, (2, 1.4)))
    ax.axvline(1.5, color=COLORS["csecf"], linewidth=0.85, linestyle=(0, (2.2, 1.4)))
    ax.scatter(
        [1.5],
        [balanced[np.where(margin_values == 1.5)[0][0]]],
        s=62,
        marker="*",
        color=COLORS["csecf"],
        edgecolor="white",
        linewidth=0.5,
        zorder=6,
    )
    for margin, gain, loss, ok in zip(margin_values, balanced, worst, feasible):
        source_rows.append(
            {
                "panel": "a",
                "margin": margin,
                "K": 5,
                "balanced_gain_ap50_points": round(gain, 6),
                "worst_evaluator_ap50_points": round(loss, 6),
                "feasible": bool(ok),
            }
        )
    ax.text(
        1.53,
        0.10,
        "selected",
        color=COLORS["csecf"],
        fontsize=5.4,
        fontweight="semibold",
    )
    ax.text(
        1.02,
        -0.30,
        "0.5-pt loss bound",
        color=COLORS["loss"],
        fontsize=5.1,
    )
    ax.set_xlim(0.96, 2.05)
    display_margins = [1.0, 1.2, 1.5, 1.75, 2.0]
    ax.set_xticks(display_margins)
    ax.set_xticklabels([f"{value:.2f}" for value in display_margins], fontsize=5.7)
    ax.set_ylim(-2.7, 0.22)
    ax.set_xlabel("Visible margin", fontsize=6.0)
    ax.set_ylabel("Validation AP50 change (points)", fontsize=6.0)
    ax.set_title("Pre-freeze selection", loc="left", fontsize=6.9, pad=13)
    clean_axis(ax, grid_axis="y")
    ax.legend(
        loc="lower right",
        fontsize=5.3,
        handlelength=1.2,
        borderaxespad=0.2,
    )
    add_panel_label(ax, "a", x=-0.19, y=1.04)

    test_grid = statistics["test_policy_grid"]
    gain_matrix = np.asarray(
        [
            [100 * test_grid[f"K{k}_m{margin:.1f}"]["dataset_balanced_mean"] for margin in MARGINS]
            for k in KS
        ]
    )
    worst_matrix = np.asarray(
        [
            [100 * test_grid[f"K{k}_m{margin:.1f}"]["min_checkpoint_gain"] for margin in MARGINS]
            for k in KS
        ]
    )
    for k_value, row_index in zip(KS, range(len(KS))):
        for col_index, margin in enumerate(MARGINS):
            source_rows.append(
                {
                    "panel": "b",
                    "margin": margin,
                    "K": k_value,
                    "test_balanced_gain_ap50_points": round(
                        gain_matrix[row_index, col_index], 6
                    ),
                    "test_worst_checkpoint_ap50_points": round(
                        worst_matrix[row_index, col_index], 6
                    ),
                    "positive_checkpoints": test_grid[
                        f"K{k_value}_m{margin:.1f}"
                    ]["positive_checkpoints"],
                    "switch_fraction": test_grid[
                        f"K{k_value}_m{margin:.1f}"
                    ]["mean_switch_fraction"],
                }
            )

    gain_cmap = mcolors.LinearSegmentedColormap.from_list(
        "validation_gain", ["#F3F7FA", "#C6DCEB", "#69A0C3", "#1D5E8D"]
    )
    ax2 = fig.add_subplot(grid[1])
    _heat_ax(
        ax2,
        gain_matrix,
        "Post-freeze: balanced gain",
        gain_cmap,
        "{:+.1f}",
    )
    add_panel_label(ax2, "b", x=-0.31, y=1.04)

    worst_cmap = mcolors.LinearSegmentedColormap.from_list(
        "validation_worst", ["#A44540", "#D77A69", "#F3D8D2", "#FAF8F6"]
    )
    ax3 = fig.add_subplot(grid[2])
    image = _heat_ax(
        ax3,
        worst_matrix,
        "Post-freeze: worst loss",
        worst_cmap,
        "{:+.1f}",
    )
    ax3.set_ylabel("")
    add_panel_label(ax3, "c", x=-0.31, y=1.04)

    cbar = fig.colorbar(
        image,
        ax=[ax2, ax3],
        fraction=0.026,
        pad=0.018,
        ticks=[-4, -3, -2, -1, 0],
    )
    cbar.ax.tick_params(labelsize=5.2, length=2)
    cbar.outline.set_linewidth(0.35)
    cbar.set_label("AP50 points", fontsize=5.5, labelpad=3)
    fig.text(
        0.08,
        0.035,
        "Margin 1.5 was selected before confirmation; all other test cells are descriptive.",
        fontsize=5.3,
        color=COLORS["mid"],
    )

    write_csv(
        "figure5_source.csv",
        [
            "panel",
            "margin",
            "K",
            "balanced_gain_ap50_points",
            "worst_evaluator_ap50_points",
            "test_balanced_gain_ap50_points",
            "test_worst_checkpoint_ap50_points",
            "positive_checkpoints",
            "switch_fraction",
            "feasible",
        ],
        source_rows,
    )
    save_figure(fig, "fig5_selection_sensitivity", dpi=600)


if __name__ == "__main__":
    main()
