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


def _gate_icon(ax, kind, x, y, color):
    if kind == "split":
        widths = (3.9, 1.5, 1.5)
        fills = ("#D9E5EF", "#9CB7CC", color)
        cursor = x
        for width, fill in zip(widths, fills):
            ax.add_patch(
                patches.FancyBboxPatch(
                    (cursor, y - 1.35),
                    width,
                    2.7,
                    boxstyle="round,pad=0.02,rounding_size=0.2",
                    facecolor=fill,
                    edgecolor="white",
                    linewidth=0.35,
                    zorder=5,
                )
            )
            cursor += width
    elif kind == "matched":
        for row, offset in enumerate((1.8, -1.8)):
            for column in range(4):
                switched = column in {1, 3}
                ax.add_patch(
                    patches.FancyBboxPatch(
                        (x + 1.1 + column * 1.35, y + offset - 0.38),
                        0.76,
                        0.76,
                        boxstyle="round,pad=0.03,rounding_size=0.12",
                        facecolor=color if switched else "white",
                        edgecolor=color,
                        linewidth=0.45,
                        zorder=5,
                    )
                )
    elif kind == "metric":
        zero_x = x + 3.2
        ax.plot([zero_x, zero_x], [y - 3.0, y + 3.0], color="#9BA7B1", lw=0.55, zorder=4)
        ax.plot([zero_x + 1.0, x + 6.3], [y + 1.5, y + 1.5], color=color, lw=1.0, zorder=5)
        ax.plot([x + 0.8, x + 4.3], [y - 1.5, y - 1.5], color="#8C99A4", lw=1.0, zorder=5)
        ax.scatter([zero_x + 3.2, zero_x + 1.1], [y + 1.5, y - 1.5], s=9, color=[color, "#8C99A4"], edgecolor="white", linewidth=0.3, zorder=6)
    elif kind == "latency":
        center = (x + 4.0, y)
        ax.add_patch(
            patches.Circle(
                center,
                2.0,
                facecolor="#F6F8FA",
                edgecolor=color,
                linewidth=0.65,
                zorder=5,
            )
        )
        ax.plot([center[0], center[0]], [center[1], center[1] + 1.2], color=color, lw=0.7, zorder=6)
        ax.plot([center[0], center[0] + 1.0], [center[1], center[1] - 0.7], color=color, lw=0.7, zorder=6)


def _gate_card(ax, x, y, width, height, gate, title, subtitle, kind, color):
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.45",
            facecolor="white",
            edgecolor="#CBD8E2",
            linewidth=0.6,
            zorder=2,
        )
    )
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y + 1.1),
            0.45,
            height - 2.2,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            facecolor=color,
            edgecolor="none",
            zorder=3,
        )
    )
    ax.text(
        x + 1.2,
        y + height - 2.2,
        gate,
        ha="left",
        va="center",
        fontsize=5.6,
        fontweight="bold",
        color=color,
        zorder=5,
    )
    _gate_icon(ax, kind, x + 2.1, y + 8.0, color)
    ax.text(
        x + 9.2,
        y + 12.0,
        title,
        ha="left",
        va="center",
        fontsize=5.9,
        fontweight="bold",
        color=COLORS["ink"],
        zorder=5,
    )
    ax.text(
        x + 9.2,
        y + 5.6,
        subtitle,
        ha="left",
        va="center",
        fontsize=5.0,
        color=COLORS["mid"],
        linespacing=0.95,
        zorder=5,
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
    # Latencies are strictly positive before the log-scaled x-axis is drawn.
    assert all(latency > 0 for _, latency, _, _, _ in cost_points)
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
    ax3.text(
        2.0,
        83.0,
        "All four checks must pass before deployment",
        ha="left",
        va="center",
        fontsize=6.4,
        color=COLORS["ink"],
        fontweight="bold",
    )

    ax3.add_patch(
        patches.FancyBboxPatch(
            (2.0, 16.0),
            62.0,
            58.0,
            boxstyle="round,pad=0.15,rounding_size=1.4",
            facecolor="#F5F9FC",
            edgecolor="#CDDCE7",
            linewidth=0.7,
            zorder=0,
        )
    )
    ax3.text(
        33.0,
        70.7,
        "FOUR REQUIRED GATES",
        ha="center",
        va="center",
        fontsize=5.8,
        color=COLORS["mid"],
        fontweight="bold",
    )
    gate_cards = [
        (
            4.5,
            46.0,
            "g1",
            "Held-out evaluator",
            "held-out detector\ntest scenes",
            "split",
            COLORS["cdf"],
        ),
        (
            34.0,
            46.0,
            "g2",
            "Matched switch budget",
            "same switch count\nand candidate mix",
            "matched",
            COLORS["csecf"],
        ),
        (
            4.5,
            21.0,
            "g3",
            "AP50 decision relevant",
            "AP50 required;\nAP50-95 not implied",
            "metric",
            COLORS["no_shield"],
        ),
        (
            34.0,
            21.0,
            "g4",
            "Latency fit",
            "four detector\npasses fit budget",
            "latency",
            COLORS["visible"],
        ),
    ]
    for x, y, gate, title, subtitle, kind, color in gate_cards:
        _gate_card(
            ax3,
            x,
            y,
            27.0,
            19.0,
            gate,
            title,
            subtitle,
            kind,
            color,
        )

    arrow_patch(
        ax3,
        (64.4, 45.0),
        (68.9, 45.0),
        color=COLORS["ink"],
        linewidth=0.85,
    )
    _diamond(ax3, (77.0, 45.0), 15.5, 20.0, "ALL FOUR\nPASS?")

    card(
        ax3,
        86.5,
        61.0,
        12.0,
        13.0,
        "Deploy",
        "CS-ECF",
        facecolor="#EAF4EE",
        edgecolor="#9FC7B2",
        accent=COLORS["pia"],
        titlecolor=COLORS["pia"],
        title_size=6.2,
        subtitle_size=5.4,
    )
    card(
        ax3,
        86.5,
        22.0,
        12.0,
        13.0,
        "Fallback",
        "visible / fixed CDF",
        facecolor="#FBEDE7",
        edgecolor="#E6C5B8",
        accent=COLORS["loss"],
        titlecolor=COLORS["loss"],
        title_size=6.2,
        subtitle_size=5.2,
    )
    arrow_patch(
        ax3,
        (84.8, 49.0),
        (86.2, 67.0),
        color=COLORS["pia"],
        linewidth=0.85,
    )
    arrow_patch(
        ax3,
        (84.8, 41.0),
        (86.2, 29.0),
        color=COLORS["loss"],
        linewidth=0.85,
    )
    ax3.text(
        83.7,
        61.5,
        "YES",
        ha="center",
        va="center",
        fontsize=5.2,
        color=COLORS["pia"],
        fontweight="bold",
    )
    ax3.text(
        83.7,
        30.5,
        "NO",
        ha="center",
        va="center",
        fontsize=5.2,
        color=COLORS["loss"],
        fontweight="bold",
    )
    ax3.text(
        2.0,
        7.0,
        "Deployment indicator:  D = 1  ⇔  g1 ∧ g2 ∧ g3 ∧ g4",
        ha="left",
        va="center",
        fontsize=6.1,
        family="STIXGeneral",
        color=COLORS["ink"],
    )
    ax3.text(
        2.0,
        2.8,
        "Validation requirements, not measured operating guarantees.",
        ha="left",
        va="center",
        fontsize=5.1,
        style="italic",
        color=COLORS["mid"],
    )

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
