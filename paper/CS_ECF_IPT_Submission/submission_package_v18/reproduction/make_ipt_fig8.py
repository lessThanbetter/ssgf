"""Figure 8: external evaluator portability stress tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ipt_figure_common import (
    COLORS,
    ROOT,
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


MODEL_LABELS = {
    "yolov9t_coco": "YOLOv9t\nzero-shot",
    "yolov10s_coco": "YOLOv10s\nzero-shot",
    "yolov9t_domain": "YOLOv9t\nLLVIP-trained",
}
POLICY_LABELS = [
    ("visible", "Visible", COLORS["visible"], "o"),
    ("always_cdf", "Always CDF", COLORS["cdf"], "s"),
    ("no_shield", "No shield", COLORS["no_shield"], "D"),
    ("csecf", "CS-ECF", COLORS["csecf"], "*"),
]


def main() -> None:
    apply_style()
    evaluator_root = ROOT / "results" / "independent_evaluators"
    coco = json.loads((evaluator_root / "policy_summary.json").read_text(encoding="utf-8"))
    domain = json.loads(
        (evaluator_root / "domain_yolov9t_v3" / "policy_summary.json").read_text(
            encoding="utf-8"
        )
    )
    diagnostics = load_json("external_evaluator_diagnostics.json")
    models = {
        "yolov9t_coco": coco["models"]["yolov9t_coco"],
        "yolov10s_coco": coco["models"]["yolov10s_coco"],
        "yolov9t_domain": domain["models"]["yolov9t_domain"],
    }
    q_bh = diagnostics["multiple_comparisons"]["bh"]
    source_rows = []

    fig = plt.figure(figsize=(6.35, 4.6))
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.02, 1.28],
        wspace=0.36,
        left=0.12,
        right=0.97,
        top=0.87,
        bottom=0.15,
    )

    ax = fig.add_subplot(grid[0])
    for row, model in enumerate(MODEL_LABELS):
        ax.axhspan(row - 0.40, row + 0.40, color=COLORS["panel"], zorder=0)
        offsets = np.linspace(-0.21, 0.21, len(POLICY_LABELS))
        for (policy, label, color, marker), offset in zip(POLICY_LABELS, offsets):
            value = models[model]["ap50"][policy]
            ax.scatter(
                value,
                row + offset,
                marker=marker,
                s=54 if policy == "csecf" else 30,
                color=color,
                edgecolor="white" if policy != "visible" else COLORS["line"],
                linewidth=0.5,
                zorder=4,
            )
            source_rows.append(
                {
                    "panel": "a",
                    "evaluator": MODEL_LABELS[model].replace("\n", " "),
                    "policy": label,
                    "ap50": round(value, 6),
                }
            )
    ax.set_yticks(range(len(MODEL_LABELS)))
    ax.set_yticklabels(list(MODEL_LABELS.values()), fontsize=5.9)
    ax.invert_yaxis()
    ax.set_xlim(0.63, 0.89)
    ax.set_xticks([0.65, 0.70, 0.75, 0.80, 0.85])
    ax.set_xlabel("AP50 on LLVIP test pairs", fontsize=6.1)
    ax.set_title("Absolute evaluator performance", loc="left", pad=28, fontsize=6.8)
    clean_axis(ax, grid_axis="x")
    ax.tick_params(axis="y", length=0, pad=3)
    handles = [marker_handle(label, color, marker, 4.2) for _, label, color, marker in POLICY_LABELS]
    ax.legend(
        handles=handles,
        ncol=2,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        fontsize=5.3,
        handletextpad=0.35,
        columnspacing=0.9,
    )
    add_panel_label(ax, "a", x=-0.22, y=1.04)

    ax2 = fig.add_subplot(grid[1])
    y_base = np.arange(len(MODEL_LABELS), dtype=float)
    for row, model in enumerate(MODEL_LABELS):
        for offset, contrast, color, label in (
            (-0.16, "csecf_minus_visible", COLORS["csecf"], "vs visible"),
            (0.16, "csecf_minus_cdf", COLORS["cdf"], "vs Always CDF"),
        ):
            estimate = 100 * models[model]["delta"][contrast]
            low, high = [100 * value for value in models[model]["group_ci95"][contrast]]
            ax2.plot([low, high], [y_base[row] + offset, y_base[row] + offset], color=color, linewidth=1.2)
            ax2.scatter(
                estimate,
                y_base[row] + offset,
                s=30,
                marker="o",
                color=color,
                edgecolor="white",
                linewidth=0.5,
                zorder=4,
            )
            ax2.text(
                estimate + (0.28 if estimate >= 0 else -0.28),
                y_base[row] + offset,
                f"{estimate:+.1f}",
                ha="left" if estimate >= 0 else "right",
                va="center",
                fontsize=5.5,
                color=color,
                fontweight="semibold",
            )
            source_rows.append(
                {
                    "panel": "b",
                    "evaluator": MODEL_LABELS[model].replace("\n", " "),
                    "comparison": label,
                    "delta_ap50_points": round(estimate, 6),
                    "ci_low_ap50_points": round(low, 6),
                    "ci_high_ap50_points": round(high, 6),
                    "bh_q_visible": q_bh[model],
                }
            )
        ax2.text(
            6.7,
            row,
            f"qBH {q_bh[model]:.3f}",
            ha="right",
            va="center",
            fontsize=5.1,
            color=COLORS["mid"],
        )
    ax2.axvline(0, color=COLORS["ink"], linewidth=0.8)
    ax2.set_yticks(y_base)
    ax2.set_yticklabels(list(MODEL_LABELS.values()), fontsize=5.8)
    ax2.invert_yaxis()
    ax2.set_xlim(-4.8, 6.9)
    ax2.set_xticks([-4, -2, 0, 2, 4, 6])
    ax2.set_xlabel("CS-ECF paired AP50 difference (points)", fontsize=6.0)
    ax2.set_title("Paired transfer contrasts", loc="left", pad=28, fontsize=6.8)
    clean_axis(ax2, grid_axis="x")
    ax2.tick_params(axis="y", length=0, pad=3)
    ax2.legend(
        handles=[
            marker_handle("vs visible", COLORS["csecf"], "o", 4.2),
            marker_handle("vs Always CDF", COLORS["cdf"], "o", 4.2),
        ],
        ncol=2,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        fontsize=5.3,
        handletextpad=0.35,
    )
    add_panel_label(ax2, "b", x=-0.23, y=1.04)

    write_csv(
        "figure8_source.csv",
        [
            "panel",
            "evaluator",
            "policy",
            "ap50",
            "comparison",
            "delta_ap50_points",
            "ci_low_ap50_points",
            "ci_high_ap50_points",
            "bh_q_visible",
        ],
        source_rows,
    )
    save_figure(fig, "fig8_external_evaluators", dpi=600)


if __name__ == "__main__":
    main()
