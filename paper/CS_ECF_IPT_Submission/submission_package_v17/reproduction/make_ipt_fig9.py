"""Figure 9: calibration and switched-image degradation boundary."""

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


MODELS = [
    ("yolov9t_coco", "YOLOv9t zero-shot"),
    ("yolov10s_coco", "YOLOv10s zero-shot"),
    ("yolov9t_domain", "YOLOv9t LLVIP-trained"),
]
SCENE_ORDER = ["190", "200", "210", "220", "230", "240", "260"]


def main() -> None:
    apply_style()
    data = load_json("external_evaluator_diagnostics.json")["models"]
    source_rows = []

    fig = plt.figure(figsize=(6.35, 5.6))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.2, 0.9],
        height_ratios=[1.15, 0.95],
        hspace=0.50,
        wspace=0.38,
        left=0.20,
        right=0.97,
        top=0.92,
        bottom=0.11,
    )

    ax = fig.add_subplot(grid[0, 0])
    y = np.arange(2 * len(MODELS))
    for model_index, (model_key, model_label) in enumerate(MODELS):
        calibration = data[model_key]["calibration"]
        entries = [
            ("ECE", calibration["visible"]["detection_ece_10bin"], calibration["csecf"]["detection_ece_10bin"]),
            ("Brier", calibration["visible"]["detection_brier"], calibration["csecf"]["detection_brier"]),
        ]
        for metric_index, (metric, visible, csecf) in enumerate(entries):
            row = 2 * model_index + metric_index
            ax.plot([visible, csecf], [row, row], color=COLORS["line"], linewidth=0.9, zorder=1)
            ax.scatter(
                visible,
                row,
                s=24,
                marker="o",
                facecolor="white",
                edgecolor=COLORS["visible"],
                linewidth=0.75,
                zorder=3,
            )
            ax.scatter(
                csecf,
                row,
                s=30,
                marker="D",
                color=COLORS["csecf"],
                edgecolor="white",
                linewidth=0.45,
                zorder=4,
            )
            source_rows.append(
                {
                    "panel": "a",
                    "evaluator": model_label,
                    "metric": metric,
                    "visible": round(visible, 6),
                    "csecf": round(csecf, 6),
                }
            )
    ax.axvline(0, color=COLORS["ink"], linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [
            f"{label}\n{metric}"
            for _, label in MODELS
            for metric in ("ECE", "Brier")
        ],
        fontsize=5.2,
        linespacing=0.92,
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 0.095)
    ax.set_xticks([0.00, 0.02, 0.04, 0.06, 0.08])
    ax.set_xlabel("Detection-level calibration", fontsize=5.9)
    ax.set_title("Calibration shifts after fine-tuning", loc="left", pad=18, fontsize=6.8)
    clean_axis(ax, grid_axis="x")
    ax.tick_params(axis="y", length=0, pad=3)
    ax.legend(
        handles=[
            marker_handle("Visible", COLORS["visible"], "o", 4.0),
            marker_handle("CS-ECF", COLORS["csecf"], "D", 4.0),
        ],
        ncol=2,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        fontsize=5.3,
        handletextpad=0.35,
    )
    add_panel_label(ax, "a", x=-0.24, y=1.04)

    ax2 = fig.add_subplot(grid[0, 1])
    x = np.arange(len(MODELS))
    improved = []
    unchanged = []
    degraded = []
    switched_counts = []
    for model_key, model_label in MODELS:
        switched = data[model_key]["switched_images"]
        improved.append(switched["fraction_improved"])
        unchanged.append(switched["fraction_unchanged"])
        degraded.append(switched["fraction_degraded"])
        switched_counts.append(switched["count"])
        for outcome, fraction in (
            ("improved", switched["fraction_improved"]),
            ("unchanged", switched["fraction_unchanged"]),
            ("degraded", switched["fraction_degraded"]),
        ):
            source_rows.append(
                {
                    "panel": "b",
                    "evaluator": model_label,
                    "outcome": outcome,
                    "fraction": round(fraction, 6),
                    "switched_images": switched["count"],
                }
            )
    improved = np.asarray(improved)
    unchanged = np.asarray(unchanged)
    degraded = np.asarray(degraded)
    ax2.bar(x, improved, width=0.56, color="#72A8C8", edgecolor="white", linewidth=0.55, label="Improved")
    ax2.bar(x, unchanged, width=0.56, bottom=improved, color="#D9DFE5", edgecolor="white", linewidth=0.55, label="Unchanged")
    ax2.bar(x, degraded, width=0.56, bottom=improved + unchanged, color="#C55E4D", edgecolor="white", linewidth=0.55, label="Degraded")
    for index in range(len(MODELS)):
        ax2.text(index, improved[index] / 2, f"{100*improved[index]:.0f}", ha="center", va="center", fontsize=5.3, color="white")
        ax2.text(index, improved[index] + unchanged[index] + degraded[index] / 2, f"{100*degraded[index]:.0f}", ha="center", va="center", fontsize=5.3, color="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [
            f"YOLOv9t\nzero-shot\n(n={switched_counts[0]})",
            f"YOLOv10s\nzero-shot\n(n={switched_counts[1]})",
            f"YOLOv9t\nfine-tuned\n(n={switched_counts[2]})",
        ],
        fontsize=5.0,
    )
    ax2.set_ylim(0, 1.0)
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_yticklabels(["0", "25%", "50%", "75%", "100%"], fontsize=5.4)
    ax2.set_title("Switched-image outcomes", loc="left", pad=18, fontsize=6.7)
    clean_axis(ax2, grid_axis="y")
    ax2.legend(
        ncol=3,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        fontsize=5.1,
        handlelength=1.0,
        columnspacing=0.8,
    )
    add_panel_label(ax2, "b", x=-0.26, y=1.04)

    ax3 = fig.add_subplot(grid[1, :])
    matrix = np.asarray(
        [
            [
                data[model]["per_scene"][
                    next(
                        index
                        for index, scene in enumerate(data[model]["per_scene"])
                        if scene["scene"] == scene_id
                    )
                ]["csecf_minus_visible"]
                for scene_id in SCENE_ORDER
            ]
            for model, _ in MODELS
        ]
    )
    norm = mcolors.TwoSlopeNorm(vmin=-0.15, vcenter=0, vmax=0.15)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "scene_delta",
        ["#B84E43", "#D58A7E", "#F7ECE9", "#F9F9F7", "#D8EAF3", "#69A4C3", "#255E86"],
    )
    image = ax3.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    ax3.set_xticks(np.arange(len(SCENE_ORDER)))
    ax3.set_xticklabels([f"Scene {scene}" for scene in SCENE_ORDER], fontsize=5.5)
    ax3.set_yticks(np.arange(len(MODELS)))
    ax3.set_yticklabels([label for _, label in MODELS], fontsize=5.5)
    ax3.tick_params(length=0)
    ax3.set_xticks(np.arange(-0.5, len(SCENE_ORDER), 1), minor=True)
    ax3.set_yticks(np.arange(-0.5, len(MODELS), 1), minor=True)
    ax3.grid(which="minor", color="white", linewidth=0.9)
    ax3.tick_params(which="minor", bottom=False, left=False)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            rgba = cmap(norm(value))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax3.text(
                col,
                row,
                f"{100 * value:+.1f}",
                ha="center",
                va="center",
                fontsize=5.5,
                color="white" if luminance < 0.45 else COLORS["ink"],
                fontweight="semibold" if value < 0 else "normal",
            )
            source_rows.append(
                {
                    "panel": "c",
                    "evaluator": MODELS[row][1],
                    "scene": SCENE_ORDER[col],
                    "csecf_minus_visible_ap50_points": round(100 * value, 6),
                }
            )
    for spine in ax3.spines.values():
        spine.set_visible(False)
    ax3.set_title("Per-scene sign changes", loc="left", pad=6, fontsize=6.7)
    cbar = fig.colorbar(image, ax=ax3, fraction=0.025, pad=0.018, ticks=[-0.15, -0.075, 0, 0.075, 0.15])
    cbar.ax.tick_params(labelsize=5.1, length=2)
    cbar.outline.set_linewidth(0.35)
    cbar.set_label("AP50 points", fontsize=5.4)
    add_panel_label(ax3, "c", x=-0.12, y=1.05)

    write_csv(
        "figure9_source.csv",
        [
            "panel",
            "evaluator",
            "metric",
            "visible",
            "csecf",
            "outcome",
            "fraction",
            "switched_images",
            "scene",
            "csecf_minus_visible_ap50_points",
        ],
        source_rows,
    )
    save_figure(fig, "fig9_calibration_boundary", dpi=600)


if __name__ == "__main__":
    main()
