"""Final Figure 11: latency cost and provisional validation questions."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import patches

from ipt_figure_common import (
    COLORS,
    add_panel_label,
    apply_style,
    arrow_patch,
    card,
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


def _diamond(ax, center, width, height, text):
    x, y = center
    points = np.asarray(
        [
            [x, y + height / 2],
            [x + width / 2, y],
            [x, y - height / 2],
            [x - width / 2, y],
        ]
    )
    ax.add_patch(
        patches.Polygon(
            points,
            closed=True,
            facecolor="#EEF3F7",
            edgecolor="#AABBC8",
            linewidth=0.7,
            zorder=3,
        )
    )
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=5.6,
        color=COLORS["ink"],
        zorder=4,
        linespacing=0.95,
    )


def main() -> None:
    apply_style()
    runtime = load_json("cs_ecf_complexity.json")
    source_rows = []

    fig = plt.figure(figsize=(6.35, 5.85))
    outer = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.88, 1.45],
        width_ratios=[1.22, 1.0],
        hspace=0.42,
        wspace=0.40,
        left=0.12,
        right=0.96,
        top=0.92,
        bottom=0.08,
    )

    ax = fig.add_subplot(outer[0, 0])
    stages = [
        ("CDF", runtime["fusion_components_ms"]["cdf_pipeline"], COLORS["cdf"]),
        ("PIAFusion", runtime["fusion_components_ms"]["piafusion"], COLORS["pia"]),
        ("SPGFusion", runtime["fusion_components_ms"]["spgfusion"], COLORS["spg"]),
        ("4 detector\nforwards", runtime["four_detector_forwards_ms"], COLORS["no_shield"]),
    ]
    left = 0.0
    for label, value, color in stages:
        ax.barh(
            [0],
            [value],
            left=[left],
            height=0.48,
            color=color,
            edgecolor="white",
            linewidth=0.65,
        )
        if value > 60:
            ax.text(
                left + value / 2,
                0,
                f"{label}\n{value:.0f} ms",
                ha="center",
                va="center",
                fontsize=5.3,
                color="white",
                fontweight="semibold",
                linespacing=0.95,
            )
        source_rows.append(
            {
                "panel": "a",
                "stage": label.replace("\n", " "),
                "latency_ms": round(value, 6),
            }
        )
        left += value
    ax.text(
        left + 16,
        0,
        f"{left:.0f} ms",
        ha="left",
        va="center",
        fontsize=7.2,
        color=COLORS["ink"],
        fontweight="bold",
    )
    ax.set_xlim(0, 800)
    ax.set_ylim(-0.7, 0.7)
    ax.set_yticks([])
    ax.set_xticks([0, 200, 400, 600, 800])
    ax.set_xlabel("Serial latency (ms)", fontsize=5.9)
    ax.set_title("Serial inference cost", loc="left", pad=6, fontsize=6.8)
    clean_axis(ax, grid_axis="x")
    add_panel_label(ax, "a", x=-0.12, y=1.04)

    ax2 = fig.add_subplot(outer[0, 1])
    cost_points = [
        ("Detector", runtime["detector_once_ms"], runtime["per_stage_extra_peak_mib"]["detector_once"], 1, COLORS["no_shield"]),
        ("PIAFusion", runtime["fusion_components_ms"]["piafusion"], runtime["per_stage_extra_peak_mib"]["piafusion"], runtime["parameters"]["piafusion_m"], COLORS["pia"]),
        ("SPGFusion", runtime["fusion_components_ms"]["spgfusion"], runtime["per_stage_extra_peak_mib"]["spgfusion"], runtime["parameters"]["spgfusion_m"], COLORS["spg"]),
    ]
    for label, latency, memory, params, color in cost_points:
        ax2.scatter(
            latency,
            memory,
            s=36 + 1.8 * params,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
        )
        ax2.text(
            latency * (1.08 if latency > 40 else 0.92),
            memory + 65,
            label,
            ha="left" if latency > 40 else "right",
            va="bottom",
            fontsize=5.5,
            color=color if label == "SPGFusion" else COLORS["ink"],
            fontweight="semibold" if label == "SPGFusion" else "normal",
        )
        source_rows.append(
            {
                "panel": "b",
                "stage": label,
                "latency_ms": round(latency, 6),
                "extra_peak_mib": round(memory, 6),
                "parameters_m": round(params, 6),
            }
        )
    ax2.set_xscale("log")
    ax2.set_xlim(25, 750)
    ax2.set_ylim(0, 1620)
    ax2.xaxis.set_major_locator(mticker.FixedLocator([30, 50, 100, 300, 500]))
    ax2.xaxis.set_minor_locator(mticker.NullLocator())
    ax2.set_xticklabels(["30", "50", "100", "300", "500"], fontsize=5.2)
    ax2.set_yticks([0, 400, 800, 1200, 1600])
    ax2.set_xlabel("Latency (ms, log scale)", fontsize=5.8)
    ax2.set_ylabel("Extra peak memory (MiB)", fontsize=5.8)
    ax2.set_title("Candidate cost profile", loc="left", pad=6, fontsize=6.8)
    clean_axis(ax2, grid_axis="both")
    add_panel_label(ax2, "b", x=-0.25, y=1.04)

    ax3 = fig.add_subplot(outer[1, :])
    ax3.set_xlim(0, 100)
    ax3.set_ylim(0, 100)
    ax3.axis("off")

    ax3.text(
        2.0,
        83.0,
        "Use CS-ECF only when all four validation gates pass",
        ha="left",
        va="center",
        fontsize=7.0,
        color=COLORS["ink"],
        fontweight="semibold",
    )
    card(
        ax3,
        2,
        58,
        31,
        17,
        "1. Evaluator validated",
        "target detector tested on held-out scenes",
        facecolor="#EAF3F6",
        edgecolor="#BDD3DD",
        titlecolor=COLORS["cdf"],
        title_size=6.3,
        subtitle_size=5.4,
    )
    card(
        ax3,
        35,
        58,
        31,
        17,
        "2. Switch budget matched",
        "same switch count and candidate mix",
        facecolor="#E8F0F7",
        edgecolor="#BFD0DF",
        titlecolor=COLORS["csecf"],
        title_size=6.3,
        subtitle_size=5.4,
    )
    card(
        ax3,
        68,
        58,
        31,
        17,
        "3. AP50 is decision relevant",
        "do not transfer the AP50 effect to AP50-95",
        facecolor="#FFF4E8",
        edgecolor="#E6CFB8",
        titlecolor=COLORS["no_shield"],
        title_size=6.3,
        subtitle_size=5.4,
    )
    card(
        ax3,
        34,
        34,
        31,
        17,
        "4. Latency budget passes",
        "four detector passes fit the deployment budget",
        facecolor="#EEF2F5",
        edgecolor="#CED8DF",
        titlecolor=COLORS["neutral"],
        title_size=6.3,
        subtitle_size=5.4,
    )
    card(
        ax3,
        2,
        4,
        96,
        18,
        "If any gate fails, retain visible input or a stable fixed CDF",
        "These gates are validation requirements, not measured operating guarantees.",
        facecolor="#FBEDE7",
        edgecolor="#E6C5B8",
        titlecolor=COLORS["loss"],
        title_size=6.8,
        subtitle_size=5.5,
    )
    ax3.text(
        0.045,
        1.025,
        "Deployment decision gates",
        transform=ax3.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.9,
        color=COLORS["ink"],
    )
    add_panel_label(ax3, "c", x=-0.01, y=1.025)

    write_csv(
        "Figure11_source.csv",
        [
            "panel",
            "stage",
            "latency_ms",
            "extra_peak_mib",
            "parameters_m",
        ],
        source_rows,
    )
    save_figure(fig, "Figure11_cost_deployment", dpi=600)


if __name__ == "__main__":
    main()
