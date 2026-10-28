"""Figure 7: label-free controls, random bound and oracle headroom."""

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


POLICIES = [
    ("uniform_random", "Uniform random", COLORS["neutral"]),
    ("max_ssim", "Maximum SSIM", "#C68872"),
    ("max_entropy", "Maximum entropy", "#A587B3"),
    ("max_vif", "Maximum reference VIF", "#7B9EB3"),
    ("always_cdf", "Always CDF", COLORS["cdf"]),
    ("cs_ecf", "CS-ECF", COLORS["csecf"]),
    ("oracle", "Per-image oracle", "#C69435"),
]


def main() -> None:
    apply_style()
    # The constants mirror the released aggregate controls and are checked below.
    statistics = load_json("major_revision_statistics.json")
    quality = load_json("image_quality_baselines.json")
    confirmatory = load_json("confirmatory_policy_comparison.json")
    source_rows = []
    values = {}
    values["max_vif"] = (
        quality["selectors"]["max_vif_fused"]["dataset_balanced_gain"],
        quality["selectors"]["max_vif_fused"]["worst_checkpoint_gain"],
        quality["selectors"]["max_vif_fused"]["positive_checkpoints"],
    )
    values["max_ssim"] = (
        quality["selectors"]["max_ssim_fused"]["dataset_balanced_gain"],
        quality["selectors"]["max_ssim_fused"]["worst_checkpoint_gain"],
        quality["selectors"]["max_ssim_fused"]["positive_checkpoints"],
    )
    values["max_entropy"] = (
        quality["selectors"]["max_entropy"]["dataset_balanced_gain"],
        quality["selectors"]["max_entropy"]["worst_checkpoint_gain"],
        quality["selectors"]["max_entropy"]["positive_checkpoints"],
    )
    values["uniform_random"] = (
        statistics["baselines"]["random"]["dataset_balanced_gain"],
        statistics["baselines"]["random"]["min_checkpoint_gain"],
        2,
    )
    values["oracle"] = (
        statistics["baselines"]["oracle"]["dataset_balanced_gain"],
        statistics["baselines"]["oracle"]["min_checkpoint_gain"],
        12,
    )
    for policy in ("always_cdf", "cs_ecf"):
        summary = confirmatory["summaries"][policy]
        values[policy] = (
            summary["dataset_balanced_mean"],
            summary["min_gain"],
            summary["positive_checkpoints"],
        )

    fig = plt.figure(figsize=(6.35, 4.45))
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[2.2, 0.9],
        hspace=0.35,
        left=0.19,
        right=0.95,
        top=0.92,
        bottom=0.11,
    )

    ax = fig.add_subplot(grid[0])
    y = np.arange(len(POLICIES))
    ax.axvline(0, color=COLORS["ink"], linewidth=0.75)
    for index, (name, label, color) in enumerate(POLICIES):
        mean, worst, positive = [100 * value for value in values[name][:2]] + [values[name][2]]
        ax.plot(
            [worst, mean],
            [index, index],
            color=color,
            linewidth=0.85,
            alpha=0.55,
            zorder=2,
        )
        ax.scatter(
            worst,
            index,
            s=28,
            marker="D",
            facecolor="white" if name == "oracle" else color,
            edgecolor=color,
            linewidth=0.8,
            zorder=4,
        )
        ax.scatter(
            mean,
            index,
            s=58 if name == "cs_ecf" else 34,
            marker="*" if name in {"cs_ecf", "oracle"} else "o",
            facecolor="white" if name == "oracle" else color,
            edgecolor=color,
            linewidth=0.8,
            zorder=5,
        )
        ax.text(
            mean + (0.55 if mean >= 0 else -0.55),
            index,
            f"{mean:+.1f}",
            ha="left" if mean >= 0 else "right",
            va="center",
            fontsize=5.7,
            color=color if name in {"cs_ecf", "oracle"} else COLORS["ink"],
            fontweight="semibold" if name in {"cs_ecf", "oracle"} else "normal",
        )
        ax.text(
            13.2,
            index,
            f"{positive}/12",
            ha="right",
            va="center",
            fontsize=5.3,
            color=COLORS["mid"],
        )
        source_rows.append(
            {
                "policy": label,
                "dataset_balanced_gain_points": round(mean, 6),
                "worst_checkpoint_gain_points": round(worst, 6),
                "positive_checkpoints": positive,
            }
        )
    ax.text(
        13.2,
        -0.80,
        "positive",
        ha="right",
        va="bottom",
        fontsize=5.0,
        color=COLORS["mid"],
        fontweight="semibold",
    )
    ax.set_yticks(y)
    ax.set_yticklabels([label for _, label, _ in POLICIES], fontsize=6.0)
    ax.invert_yaxis()
    ax.set_xlim(-31.0, 13.8)
    ax.set_xticks([-30, -20, -10, 0, 10])
    ax.set_xlabel("Dataset-balanced / worst-checkpoint AP50 change (points)", fontsize=6.1)
    ax.set_title("Policy controls and label-free headroom", loc="left", pad=10, fontsize=7.0)
    clean_axis(ax, grid_axis="x")
    ax.tick_params(axis="y", length=0, pad=3)
    add_panel_label(ax, "a", x=-0.20, y=1.03)

    ax2 = fig.add_subplot(grid[1])
    ax2.set_xlim(-4.2, 9.2)
    ax2.set_ylim(-0.8, 1.05)
    ax2.axis("off")
    random_gain = 100 * values["uniform_random"][0]
    cs_gain = 100 * values["cs_ecf"][0]
    oracle_gain = 100 * values["oracle"][0]
    y0 = 0.15
    ax2.plot([random_gain, oracle_gain], [y0, y0], color=COLORS["line"], linewidth=5.0, solid_capstyle="round")
    filled_end = random_gain + (oracle_gain - random_gain) * 0.404
    ax2.plot([random_gain, filled_end], [y0, y0], color=COLORS["csecf"], linewidth=5.0, solid_capstyle="round")
    for value, label, color, marker in (
        (random_gain, "random", COLORS["neutral"], "o"),
        (cs_gain, "CS-ECF", COLORS["csecf"], "*"),
        (oracle_gain, "oracle", "#C69435", "o"),
    ):
        ax2.scatter(value, y0, s=72 if marker == "*" else 42, marker=marker, color=color, edgecolor="white", linewidth=0.6, zorder=4)
        ax2.text(value, y0 + 0.25, label, ha="center", va="bottom", fontsize=5.8, color=color)
        ax2.text(value, y0 - 0.20, f"{value:+.1f}", ha="center", va="top", fontsize=5.4, color=COLORS["mid"])
    ax2.annotate(
        "40% of random-to-oracle headroom",
        xy=(filled_end, y0 + 0.06),
        xytext=(filled_end - 1.4, 0.72),
        ha="center",
        fontsize=5.8,
        color=COLORS["csecf"],
        arrowprops={"arrowstyle": "-", "color": COLORS["csecf"], "linewidth": 0.6},
    )
    ax2.set_title("Balanced-gain positioning", loc="left", pad=5, fontsize=6.7)
    add_panel_label(ax2, "b", x=-0.10, y=1.10)

    write_csv(
        "figure7_source.csv",
        [
            "policy",
            "dataset_balanced_gain_points",
            "worst_checkpoint_gain_points",
            "positive_checkpoints",
        ],
        source_rows,
    )
    save_figure(fig, "fig7_controls_headroom", dpi=600)


if __name__ == "__main__":
    main()
