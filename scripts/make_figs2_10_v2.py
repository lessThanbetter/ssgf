"""Redraw Figures 2-10 from the frozen analysis outputs.

The script keeps Figures 1 and 11 untouched. It uses one journal-scale visual system:
Arial text, restrained semantic colours, direct labels, thin axes and vector output.
"""

# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# Fig. 2: interval/forest plots -> Forest (cross-type inherit) -> parameter inherit
# Fig. 3: classification matrix + line sweep -> BarCategorical/LineTrend -> parameter inherit
# Fig. 4: interval plots + rank bump chart -> Forest/LineTrend -> parameter inherit
# Fig. 5: scatter + correlation dot plot -> basic-plots/BarCategorical -> parameter inherit
# Fig. 6: criterion sweep -> LineTrend -> parameter inherit
# Fig. 7: recall/precision dot plots -> BarCategorical -> parameter inherit
# Fig. 8: temperature sweep -> LineTrend -> parameter inherit
# Fig. 9: scale and resolution comparisons -> Forest/LineTrend -> parameter inherit
# Fig. 10: corruption contrasts + resolved-input matrix -> BarCategorical -> parameter inherit
# RULE: all panels use parameter inheritance because the production assets encode unrelated
#       study variables; no production PNG is loaded as a native panel.

# Academic Figure Skill Typography Baseline - COPY VERBATIM, place at TOP of script
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "figure.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
})

# Academic Figure Skill Nature/Cell/Science Color Palette - COPY VERBATIM
CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
CATEGORICAL_EXTENDED = [
    "#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666",
    "#4393C3", "#D6604D", "#5AAE61", "#B35806", "#9970AB", "#999999",
]
DIVERGING = ["#2166AC", "#F7F7F7", "#B2182B"]
SEQUENTIAL = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED = "#B2182B"
GREY = "#999999"
BLACK = "#222222"

# Academic Figure Skill Export Baseline - COPY VERBATIM
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

def save_cns_figure(fig, filename):
    """Standard Academic Figure Skill export: vector PDF + 300dpi PNG preview."""
    fig.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{filename}.png", bbox_inches="tight", dpi=300)


import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
AGG = ROOT / "_deposit/ssgf-llvip-fusion-detection-v1/data/aggregate"
FID_PATH = ROOT / "_remote_jobs/results/j1_fidelity.json"
IOU = np.round(np.linspace(0.5, 0.95, 10), 2)

ORDER = [
    "spgfusion",
    "ifcnn",
    "piafusion",
    "activity_only",
    "mean",
    "probe",
    "tophat",
    "ensemble",
    "densefuse",
    "csf",
    "probe_wshuffle",
    "tophat_wshuffle",
    "probe_permuted",
    "tophat_permuted",
]

LABEL = {
    "mean": "mean",
    "activity_only": "activity-only",
    "tophat": "top-hat",
    "probe": "probe",
    "tophat_permuted": "top-hat perm.",
    "probe_permuted": "probe perm.",
    "tophat_wshuffle": "top-hat w-shuf.",
    "probe_wshuffle": "probe w-shuf.",
    "ensemble": "ensemble",
    "ifcnn": "IFCNN",
    "densefuse": "DenseFuse",
    "csf": "CSF",
    "piafusion": "PIAFusion",
    "spgfusion": "SPGFusion",
    "ir": "IR source",
    "vis": "VIS source",
}

EVALUATOR = {
    "yolo11n_vis": ("YOLO11n", CATEGORICAL[0], "o"),
    "yolo26n_vis": ("YOLO26n", CATEGORICAL[1], "s"),
    "rtdetr_l_vis": ("RT-DETR-l", CATEGORICAL[2], "^"),
    "agnostic": ("Agnostic", CATEGORICAL[4], "D"),
}

RULE_COLOR = {
    "mean": "#777777",
    "activity_only": "#F1A340",
    "tophat": "#1B7837",
    "probe": "#2166AC",
    "ensemble": "#4393C3",
    "ifcnn": "#B2182B",
    "densefuse": "#D6604D",
    "csf": "#999999",
    "piafusion": "#B35806",
    "spgfusion": "#762A83",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_frozen_inputs() -> None:
    clean = load(ROOT / "results/clean_r2/clean_r2_full.json")["evaluators"]
    visible_keys = (
        "yolo11n_visible_clean",
        "yolo26n_visible_clean",
        "rtdetr_l_visible_clean",
    )
    visible_values = [
        value["mean_contrast"]
        for key in visible_keys
        for value in clean[key]["inputs"].values()
    ]
    agnostic = clean["yolo11n_modality_agnostic_clean"]["inputs"]
    agnostic_stable = [
        name
        for name, value in agnostic.items()
        if all(run > 0 for run in value["per_run_contrast"])
    ]
    seed_ranges = load(AGG / "clean_split_seeds.json")["per_seed"]
    split_stable = [
        name for name, value in seed_ranges.items() if value["lo"] > 0
    ]
    vif = load(AGG / "vif_decomp.json")["rows"]
    reported = sorted(vif, key=lambda row: row["reported"], reverse=True)
    reference = sorted(vif, key=lambda row: row["reference"], reverse=True)
    assertions = {
        "42 visible-trained contrasts": len(visible_values) == 42,
        "all clean visible-trained contrasts negative": all(
            value < 0 for value in visible_values
        ),
        "four clean agnostic gains": set(agnostic_stable)
        == {"probe", "tophat", "ensemble", "piafusion"},
        "four seed-range gains": set(split_stable)
        == {"probe", "tophat", "ensemble", "piafusion"},
        "reported VIF endpoint": reported[0]["variant"] == "spgfusion",
        "reference VIF endpoint": reference[-1]["variant"] == "spgfusion",
    }
    failed = [name for name, passed in assertions.items() if not passed]
    if failed:
        raise RuntimeError("frozen-data assertions failed: " + ", ".join(failed))


def save(fig: plt.Figure, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    base = output / name
    save_cns_figure(fig, str(base))
    fig.savefig(f"{base}.svg", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(base)


def panel_tag(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.02,
        1.035,
        f"({letter})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
        color=BLACK,
        clip_on=False,
    )


def clean_axes(ax: plt.Axes, grid: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.55)
    ax.spines["bottom"].set_linewidth(0.55)
    ax.tick_params(axis="both", width=0.55, length=2.2, pad=2)
    if grid:
        ax.grid(
            axis=grid,
            color="#D9D9D9",
            linewidth=0.35,
            linestyle=(0, (1, 2)),
            alpha=0.8,
            zorder=0,
        )
    else:
        ax.grid(False)


def horizontal_series(
    ax: plt.Axes,
    rows: list[str],
    series: list[tuple[str, dict, str, str]],
    xlim: tuple[float, float],
    label_size: float = 5.8,
    offset_step: float = 0.22,
) -> None:
    y = np.arange(len(rows), dtype=float)
    for index, (_label, data, color, marker) in enumerate(series):
        offset = (index - (len(series) - 1) / 2) * offset_step
        values = np.asarray([data[row]["delta50"] for row in rows], dtype=float)
        lows = np.asarray([data[row]["lo50"] for row in rows], dtype=float)
        highs = np.asarray([data[row]["hi50"] for row in rows], dtype=float)
        ax.hlines(
            y + offset,
            lows,
            highs,
            color=color,
            linewidth=0.8,
            alpha=0.85,
            zorder=2,
        )
        ax.plot(
            values,
            y + offset,
            linestyle="none",
            marker=marker,
            markersize=3.4,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.25,
            zorder=3,
        )
    ax.axvline(0, color="#444444", linewidth=0.65, zorder=1)
    ax.set_xlim(*xlim)
    ax.set_ylim(len(rows) - 0.45, -0.55)
    ax.set_yticks(y, [LABEL[row] for row in rows])
    ax.tick_params(axis="y", length=0, labelsize=label_size)
    clean_axes(ax, grid="x")


def evaluator_handle(label: str, color: str, marker: str) -> Line2D:
    return Line2D(
        [0],
        [0],
        color=color,
        marker=marker,
        markersize=3.8,
        linewidth=0.9,
        label=label,
    )


def fig_claim(output: Path) -> None:
    clean = load(ROOT / "results/clean_r2/clean_r2_full.json")["evaluators"]
    pooled = {
        "yolo11n_vis": load(AGG / "contrasts_yolo11n_colour.json"),
        "yolo26n_vis": load(AGG / "contrasts_yolo26n_colour.json"),
        "rtdetr_l_vis": {
            key: {
                "delta50": value["mean_over_seeds"],
                "lo50": value["lo"],
                "hi50": value["hi"],
            }
            for key, value in load(AGG / "rtdetr_seeds.json")["per_seed"].items()
        },
    }

    mm = 1 / 25.4
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(180 * mm, 86 * mm),
        sharey=True,
        gridspec_kw={"wspace": 0.08, "width_ratios": [1.0, 1.0, 1.12]},
    )
    y = np.arange(len(ORDER), dtype=float)

    ax = axes[0]
    clean_visible = [
        (
            "YOLO11n",
            clean["yolo11n_visible_clean"]["inputs"],
            CATEGORICAL[0],
            "o",
            "runs",
        ),
        (
            "YOLO26n",
            clean["yolo26n_visible_clean"]["inputs"],
            CATEGORICAL[1],
            "s",
            "ci",
        ),
        (
            "RT-DETR-l",
            clean["rtdetr_l_visible_clean"]["inputs"],
            CATEGORICAL[2],
            "^",
            "ci",
        ),
    ]
    for index, (_name, data, color, marker, interval_kind) in enumerate(clean_visible):
        offset = (index - 1) * 0.24
        if interval_kind == "runs":
            means = np.asarray([data[row]["mean_contrast"] for row in ORDER])
            lows = np.asarray([data[row]["min_contrast"] for row in ORDER])
            highs = np.asarray([data[row]["max_contrast"] for row in ORDER])
        else:
            means = np.asarray([data[row]["mean_contrast"] for row in ORDER])
            lows = np.asarray([data[row]["lo50"] for row in ORDER])
            highs = np.asarray([data[row]["hi50"] for row in ORDER])
        ax.hlines(y + offset, lows, highs, color=color, lw=0.8, alpha=0.85)
        ax.plot(
            means,
            y + offset,
            "o" if marker == "o" else marker,
            ms=3.2,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.25,
            linestyle="none",
        )
    ax.axvline(0, color="#444444", lw=0.65)
    ax.set_xlim(-0.27, 0.035)
    ax.set_ylim(len(ORDER) - 0.45, -0.55)
    ax.set_yticks(y, [LABEL[row] for row in ORDER])
    ax.tick_params(axis="y", length=0, labelsize=5.8)
    ax.set_xlabel(r"$\Delta$ mAP50 vs VIS")
    ax.set_title("Clean visible-only", fontsize=7.2, pad=8)
    clean_axes(ax, grid="x")
    panel_tag(ax, "a")

    ax = axes[1]
    data = clean["yolo11n_modality_agnostic_clean"]["inputs"]
    means = np.asarray([data[row]["mean_contrast"] for row in ORDER])
    lows = np.asarray([data[row]["min_contrast"] for row in ORDER])
    highs = np.asarray([data[row]["max_contrast"] for row in ORDER])
    stable_positive = [
        index
        for index, row in enumerate(ORDER)
        if all(value > 0 for value in data[row]["per_run_contrast"])
    ]
    for index, (mean, low, high) in enumerate(zip(means, lows, highs)):
        color = ACCENT_RED if index in stable_positive else CATEGORICAL[4]
        ax.hlines(index, low, high, color=color, lw=0.8, alpha=0.9)
        ax.plot(
            mean,
            index,
            "D",
            ms=3.4,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.25,
        )
    ax.axvline(0, color="#444444", lw=0.65)
    ax.set_xlim(-0.06, 0.055)
    ax.set_xlabel(r"$\Delta$ mAP50 vs VIS")
    ax.set_title("Clean modality-agnostic", fontsize=7.2, pad=8)
    clean_axes(ax, grid="x")
    panel_tag(ax, "b")

    ax = axes[2]
    pooled_series = [
        ("YOLO11n", pooled["yolo11n_vis"], CATEGORICAL[0], "o"),
        ("YOLO26n", pooled["yolo26n_vis"], CATEGORICAL[1], "s"),
        ("RT-DETR-l", pooled["rtdetr_l_vis"], CATEGORICAL[2], "^"),
    ]
    horizontal_series(ax, ORDER, pooled_series, (-0.26, 0.14), label_size=5.8)
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelleft=False)
    ax.set_xlabel(r"$\Delta$ mAP50 vs VIS")
    ax.set_title("Pooled audit (optimistic)", fontsize=7.2, pad=8)
    panel_tag(ax, "c")

    handles = [
        evaluator_handle("YOLO11n (4 runs)", CATEGORICAL[0], "o"),
        evaluator_handle("YOLO26n (1 run)", CATEGORICAL[1], "s"),
        evaluator_handle("RT-DETR-l (1 run)", CATEGORICAL[2], "^"),
        evaluator_handle("Agnostic (4 runs)", CATEGORICAL[4], "D"),
        Line2D([0], [0], color=ACCENT_RED, marker="D", lw=0.8, ms=3.8,
               label="Stable positive"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=5,
        fontsize=5.7,
        handlelength=1.2,
        columnspacing=1.1,
        handletextpad=0.35,
    )
    fig.subplots_adjust(left=0.105, right=0.985, top=0.90, bottom=0.20)
    save(fig, output, "fig_claim_v2")


def fig_regime(output: Path) -> None:
    regime = load(AGG / "regime_conditional.json")
    curve = load(AGG / "regime_curve.json")["yolo11n_vis"]
    seg = load(AGG / "seg_regime_summary.json")["groups"]
    detectors = ["yolo11n_vis", "yolo26n_vis", "rtdetr_l_vis"]
    regimes = [
        ("all 300 images", "all"),
        ("low visible contrast (bottom tercile)", "low"),
        ("high visible contrast (top tercile)", "high"),
        (r"three or more pedestrians", r"$\geq$3"),
        ("small targets (below median area)", "small"),
    ]

    mm = 1 / 25.4
    fig = plt.figure(figsize=(180 * mm, 120 * mm))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.02],
        hspace=0.46,
        wspace=0.26,
    )
    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(-0.5, len(regimes) - 0.5)
    ax.set_ylim(len(detectors) - 0.5, -0.5)
    ax.set_xticks(range(len(regimes)), [short for _, short in regimes])
    ax.set_yticks(range(len(detectors)), [EVALUATOR[det][0] for det in detectors])
    ax.tick_params(length=0, labelsize=6.2)
    ax.set_axisbelow(True)
    status_color = {
        "below": "#1B7837",
        "unresolved": "#D9D9D9",
        "above": "#F1A340",
    }
    for row_index, detector in enumerate(detectors):
        for col_index, (regime_name, _short) in enumerate(regimes):
            ax.add_patch(
                Rectangle(
                    (col_index - 0.48, row_index - 0.46),
                    0.96,
                    0.92,
                    facecolor="#FAFAFA" if (row_index + col_index) % 2 else "#F3F6F9",
                    edgecolor="none",
                    zorder=0,
                )
            )
            cell = regime[detector][regime_name]
            sequence = (
                ["below"] * int(cell["below_resolved"])
                + ["unresolved"] * int(cell["unresolved"])
                + ["above"] * int(cell["above_resolved"])
            )
            for index, status in enumerate(sequence):
                dx = ((index % 7) - 3) * 0.095
                dy = ((index // 7) - 0.5) * 0.22
                ax.scatter(
                    col_index + dx,
                    row_index + dy,
                    s=8,
                    color=status_color[status],
                    edgecolor="white",
                    linewidth=0.18,
                    zorder=3,
                )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("evaluation regime", fontsize=6.6, labelpad=4)
    handles = [
        Line2D([0], [0], marker="o", lw=0, color=status_color[status], ms=4,
               label=status)
        for status in ("below", "unresolved", "above")
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.00),
        ncol=3,
        fontsize=5.7,
        handletextpad=0.35,
        columnspacing=1.0,
    )
    panel_tag(ax, "a")

    ax = fig.add_subplot(gs[1, 0])
    x = np.asarray([row["mean_contrast"] for row in curve], dtype=float)
    for row in ORDER:
        values = np.asarray([item["inputs"][row]["delta"] for item in curve], dtype=float)
        ax.plot(
            x,
            values,
            color="#BBBBBB",
            linewidth=0.6,
            alpha=0.5,
            zorder=1,
        )
    highlight = [
        ("mean", "#777777", "o"),
        ("probe", CATEGORICAL[0], "s"),
        ("spgfusion", CATEGORICAL[4], "D"),
    ]
    for row, color, marker in highlight:
        values = np.asarray([item["inputs"][row]["delta"] for item in curve], dtype=float)
        lows = np.asarray([item["inputs"][row]["lo"] for item in curve], dtype=float)
        highs = np.asarray([item["inputs"][row]["hi"] for item in curve], dtype=float)
        if row == "spgfusion":
            ax.fill_between(x, lows, highs, color=color, alpha=0.12, lw=0)
        ax.plot(
            x,
            values,
            marker=marker,
            color=color,
            lw=1.0,
            ms=3.5,
            markeredgecolor="white",
            markeredgewidth=0.25,
            zorder=3,
            label=LABEL[row],
        )
    ax.axhline(0, color="#444444", lw=0.65)
    ax.set_xlabel("VIS contrast (bin mean)")
    ax.set_ylabel(r"$\Delta$ mAP50")
    clean_axes(ax, grid="both")
    ax.legend(fontsize=5.6, loc="lower left", handlelength=1.3, ncol=3)
    panel_tag(ax, "b")

    ax = fig.add_subplot(gs[1, 1])
    names = [
        "low visible contrast (bottom tercile)",
        "middle tercile",
        "high visible contrast (top tercile)",
    ]
    labels = ["low", "middle", "high"]
    values = np.asarray([seg[name]["inputs"]["mean"]["delta"] for name in names])
    lows = np.asarray([seg[name]["inputs"]["mean"]["lo"] for name in names])
    highs = np.asarray([seg[name]["inputs"]["mean"]["hi"] for name in names])
    y = np.arange(len(names))
    ax.hlines(y, lows, highs, color=CATEGORICAL[0], lw=0.9)
    ax.plot(
        values,
        y,
        "o",
        color=CATEGORICAL[0],
        ms=4.0,
        markeredgecolor="white",
        markeredgewidth=0.25,
    )
    ax.axvline(0, color="#444444", lw=0.65)
    ax.set_yticks(y, labels)
    ax.set_ylim(len(names) - 0.45, -0.55)
    ax.set_xlabel(r"$\Delta$ mIoU (segmentation)")
    clean_axes(ax, grid="x")
    panel_tag(ax, "c")

    fig.subplots_adjust(left=0.12, right=0.985, top=0.91, bottom=0.10)
    save(fig, output, "fig_regime_v2")


def fig_robust(output: Path) -> None:
    seeds = load(AGG / "clean_split_seeds.json")["per_seed"]
    rtdetr = load(AGG / "rtdetr_seeds.json")["per_seed"]
    vif = load(AGG / "vif_decomp.json")["rows"]
    mm = 1 / 25.4
    fig = plt.figure(figsize=(180 * mm, 120 * mm))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.02],
        hspace=0.46,
        wspace=0.36,
    )

    ax = fig.add_subplot(gs[0, :])
    y = np.arange(len(ORDER))
    means = np.asarray([seeds[row]["mean_over_seeds"] for row in ORDER])
    lows = np.asarray([seeds[row]["lo"] for row in ORDER])
    highs = np.asarray([seeds[row]["hi"] for row in ORDER])
    stable_positive = [row for row in ORDER if lows[ORDER.index(row)] > 0]
    for index, row in enumerate(ORDER):
        color = ACCENT_RED if row in stable_positive else CATEGORICAL[4]
        ax.hlines(index, lows[index], highs[index], color=color, lw=0.85)
        ax.plot(
            means[index],
            index,
            "o",
            ms=3.2,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.25,
        )
    ax.axvline(0, color="#444444", lw=0.65)
    ax.set_xlim(-0.065, 0.055)
    ax.set_ylim(len(ORDER) - 0.45, -0.55)
    ax.set_yticks(y, [LABEL[row] for row in ORDER])
    ax.tick_params(axis="y", length=0, labelsize=5.8)
    ax.set_xlabel(r"$\Delta$ mAP50 vs VIS")
    clean_axes(ax, grid="x")
    panel_tag(ax, "a")

    ax = fig.add_subplot(gs[1, 0])
    y = np.arange(len(ORDER))
    means = np.asarray([rtdetr[row]["mean_over_seeds"] for row in ORDER])
    lows = np.asarray([rtdetr[row]["lo"] for row in ORDER])
    highs = np.asarray([rtdetr[row]["hi"] for row in ORDER])
    ax.hlines(y, lows, highs, color=CATEGORICAL[2], lw=0.8)
    ax.plot(
        means,
        y,
        "^",
        ms=3.2,
        color=CATEGORICAL[2],
        markeredgecolor="white",
        markeredgewidth=0.25,
    )
    ax.axvline(0, color="#444444", lw=0.65)
    ax.set_xlim(-0.13, 0.02)
    ax.set_ylim(len(ORDER) - 0.45, -0.55)
    ax.set_yticks(y, [LABEL[row] for row in ORDER])
    ax.tick_params(axis="y", length=0, labelsize=5.5)
    ax.set_xlabel(r"$\Delta$ mAP50 vs VIS")
    clean_axes(ax, grid="x")
    panel_tag(ax, "b")

    ax = fig.add_subplot(gs[1, 1])
    metrics = ["reported", "noise_only", "reference"]
    xcols = np.arange(len(metrics))
    ranks = {}
    for metric in metrics:
        sorted_rows = sorted(vif, key=lambda row: row[metric], reverse=True)
        ranks[metric] = {
            row["variant"]: index + 1 for index, row in enumerate(sorted_rows)
        }
    for row in vif:
        variant = row["variant"]
        ys = [ranks[metric][variant] for metric in metrics]
        highlighted = variant == "spgfusion"
        ax.plot(
            xcols,
            ys,
            color=ACCENT_RED if highlighted else "#B5B5B5",
            linewidth=1.25 if highlighted else 0.55,
            alpha=0.95 if highlighted else 0.55,
            marker="o",
            ms=3.2 if highlighted else 2.3,
            markeredgecolor="white",
            markeredgewidth=0.2,
            zorder=3 if highlighted else 1,
        )
    ax.set_xlim(-0.2, 2.2)
    ax.set_ylim(len(vif) + 0.6, 0.4)
    ax.set_xticks(xcols, ["reported", "noise only", "reference"])
    ax.set_yticks(np.arange(1, len(vif) + 1))
    ax.set_ylabel("VIF rank")
    ax.tick_params(axis="x", length=0)
    clean_axes(ax, grid="y")
    ax.text(2.08, ranks["reference"]["spgfusion"], "SPG",
            ha="left", va="center", fontsize=5.6, fontweight="bold",
            color=ACCENT_RED)
    panel_tag(ax, "c")

    fig.subplots_adjust(left=0.12, right=0.985, top=0.91, bottom=0.10)
    save(fig, output, "fig_robust_v2")


def fig_corr(output: Path) -> None:
    extra = load(AGG / "extra_analyses.json")
    corr = extra["rank_correlation"]
    fidelity = load(FID_PATH)
    mm = 1 / 25.4
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(180 * mm, 82 * mm),
        gridspec_kw={"width_ratios": [1.32, 1.0], "wspace": 0.32},
    )
    evaluators = [
        ("yolo11n_vis", CATEGORICAL[0], "o"),
        ("yolo26n_vis", CATEGORICAL[1], "s"),
        ("rtdetr_l_vis", CATEGORICAL[2], "^"),
        ("agnostic", CATEGORICAL[4], "D"),
    ]

    ax = axes[0]
    for evaluator, color, marker in evaluators:
        contrasts = load(AGG / f"contrasts_{evaluator.split('_vis')[0]}_colour.json")
        rows = [row for row in ORDER if row in fidelity and row in contrasts]
        x = np.asarray([fidelity[row]["VIF_vis"] for row in rows])
        y = np.asarray([contrasts[row]["delta50"] for row in rows])
        ax.scatter(
            x,
            y,
            s=19,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.25,
            alpha=0.9,
            label=EVALUATOR[evaluator][0],
            zorder=3,
        )
    ax.axhline(0, color="#444444", lw=0.65)
    ax.set_xlabel("reported-index VIF")
    ax.set_ylabel(r"$\Delta$ mAP50 vs VIS")
    ax.set_xlim(0.24, 0.87)
    ax.set_ylim(-0.17, 0.10)
    clean_axes(ax, grid="both")
    panel_tag(ax, "a")

    ax = axes[1]
    metrics = ["VIF", "SSIM_vis", "EN"]
    metric_labels = ["VIF", "SSIM$_{vis}$", "EN"]
    for metric_index, metric in enumerate(metrics):
        for evaluator_index, (evaluator, color, marker) in enumerate(evaluators):
            offset = (evaluator_index - 1.5) * 0.13
            value = corr[evaluator][metric]
            ax.plot(
                [metric_index + offset, metric_index + offset],
                [0, value],
                color=color,
                lw=0.65,
                alpha=0.75,
                zorder=1,
            )
            ax.plot(
                metric_index + offset,
                value,
                marker,
                color=color,
                ms=3.4,
                markeredgecolor="white",
                markeredgewidth=0.25,
                zorder=3,
            )
    ax.axhline(0, color="#444444", lw=0.65)
    ax.set_xticks(range(len(metrics)), metric_labels)
    ax.set_ylabel(r"Spearman $\rho$")
    ax.set_ylim(-0.75, 0.75)
    clean_axes(ax, grid="y")
    panel_tag(ax, "b")

    handles = [
        evaluator_handle(EVALUATOR[key][0], color, marker)
        for key, color, marker in evaluators
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        fontsize=5.8,
        handlelength=1.3,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.91, bottom=0.21)
    save(fig, output, "fig_corr_v2")


def fig_criterion(output: Path) -> None:
    criterion = load(AGG / "extra_analyses.json")["criterion"]
    rules = [
        ("mean", RULE_COLOR["mean"], "o", "-"),
        ("probe", RULE_COLOR["probe"], "s", "-"),
        ("ifcnn", RULE_COLOR["ifcnn"], "^", (0, (3, 1.4))),
        ("spgfusion", RULE_COLOR["spgfusion"], "D", "-"),
    ]
    mm = 1 / 25.4
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(180 * mm, 84 * mm),
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )
    for ax, evaluator, title, letter in zip(
        axes,
        ("yolo11n_vis", "agnostic"),
        ("YOLO11n visible-only", "Modality-agnostic"),
        ("a", "b"),
    ):
        for key, color, marker, linestyle in rules:
            values = criterion[evaluator]["contrasts"][key]
            ax.plot(
                IOU,
                values,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.0,
                markersize=3.4,
                markeredgecolor="white",
                markeredgewidth=0.25,
                label=LABEL[key],
            )
        ax.axhline(0, color="#444444", lw=0.65)
        ax.set_xlabel("IoU criterion")
        ax.set_title(title, fontsize=7.2, pad=8)
        ax.set_xlim(0.49, 0.96)
        ax.set_ylim(-0.18, 0.23)
        clean_axes(ax, grid="both")
        panel_tag(ax, letter)
    axes[0].set_ylabel(r"$\Delta$ mAP vs visible source")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        fontsize=5.8,
        handlelength=1.5,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.88, bottom=0.23)
    save(fig, output, "fig_criterion_v2")


def fig_operating(output: Path) -> None:
    operating = load(AGG / "extra_analyses.json")["operating_point"]
    inputs = ["vis", "mean", "probe", "ifcnn", "spgfusion"]
    mm = 1 / 25.4
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(180 * mm, 82 * mm),
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )
    evaluators = ["yolo11n_vis", "yolo26n_vis", "rtdetr_l_vis", "agnostic"]
    for ax, metric, title, letter in zip(
        axes,
        ("recall", "precision"),
        ("Recall at maximum F1", "Precision at maximum F1"),
        ("a", "b"),
    ):
        ax.axhspan(3.55, 4.45, color="#F8EEF0", zorder=0)
        for index, evaluator in enumerate(evaluators):
            offsets = (index - 1.5) * 0.17
            color = EVALUATOR[evaluator][1]
            marker = EVALUATOR[evaluator][2]
            values = [operating[evaluator][row][metric] for row in inputs]
            ax.scatter(
                values,
                np.arange(len(inputs)) + offsets,
                s=19,
                marker=marker,
                color=color,
                edgecolor="white",
                linewidth=0.25,
                zorder=3,
            )
        ax.set_xlabel(metric.capitalize())
        ax.set_xlim(0.58, 0.97)
        ax.set_title(title, fontsize=7.2, pad=8)
        clean_axes(ax, grid="x")
        panel_tag(ax, letter)
    axes[0].set_yticks(np.arange(len(inputs)), [LABEL[row] for row in inputs])
    axes[0].set_ylim(len(inputs) - 0.45, -0.55)
    axes[0].tick_params(axis="y", length=0)
    handles = [
        evaluator_handle(EVALUATOR[key][0], EVALUATOR[key][1], EVALUATOR[key][2])
        for key in evaluators
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        fontsize=5.8,
        handlelength=1.3,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.10, right=0.985, top=0.88, bottom=0.23)
    save(fig, output, "fig_operating_v2")


def fig_tau(output: Path) -> None:
    tau = load(AGG / "temperature_sweep.json")
    mm = 1 / 25.4
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(180 * mm, 84 * mm),
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )
    styles = {
        "activity_only": (LABEL["activity_only"], CATEGORICAL[1], "o"),
        "tophat": (LABEL["tophat"], CATEGORICAL[2], "s"),
        "probe": (LABEL["probe"], CATEGORICAL[0], "^"),
    }
    for ax, evaluator, title, letter in zip(
        axes,
        ("vis", "agn"),
        ("YOLO11n visible-only", "Modality-agnostic"),
        ("a", "b"),
    ):
        rows = [row for row in tau["rows"] if row["evaluator"] == evaluator]
        baseline = tau["mean_fusion"][evaluator]
        for rule, (label, color, marker) in styles.items():
            rule_rows = sorted(
                [row for row in rows if row["rule"] == rule],
                key=lambda row: row["tau"],
            )
            x = [row["tau"] for row in rule_rows]
            y = [row["mAP50"] - baseline for row in rule_rows]
            ax.plot(
                x,
                y,
                color=color,
                marker=marker,
                lw=1.0,
                ms=3.5,
                markeredgecolor="white",
                markeredgewidth=0.25,
                label=label,
            )
        ax.axhline(0, color="#444444", lw=0.75, linestyle=(0, (2, 1.5)))
        ax.set_xscale("log")
        ax.set_xticks([0.1, 0.3, 1.0, 3.0], ["0.1", "0.3", "1", "3"])
        ax.set_xlim(0.08, 4.0)
        ax.set_ylim(-0.095, 0.025)
        ax.set_xlabel(r"temperature $\tau$ (log scale)")
        ax.set_title(title, fontsize=7.2, pad=8)
        clean_axes(ax, grid="both")
        panel_tag(ax, letter)
    axes[0].set_ylabel(r"$\Delta$ mAP50 vs mean fusion")
    axes[1].text(2.8, 0.005, "mean fusion", fontsize=5.5, color="#555555",
                 ha="right", va="bottom")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        fontsize=5.8,
        handlelength=1.5,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.88, bottom=0.23)
    save(fig, output, "fig_tau_v2")


def fig_scale(output: Path) -> None:
    base = load(AGG / "contrasts_yolo11n_colour.json")
    large = load(AGG / "contrasts_yolo11s.json")["rows"]
    resolution = load(AGG / "resolution_sweep.json")
    rows = [row for row in ORDER + ["ir"] if row in large and row in base]
    mm = 1 / 25.4
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(180 * mm, 100 * mm),
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.20},
    )

    ax = axes[0]
    y = np.arange(len(rows))
    for index, row in enumerate(rows):
        x0 = base[row]["delta50"]
        x1 = large[row]["delta50"]
        ax.plot([x0, x1], [index, index], color="#D2D2D2", lw=0.8, zorder=1)
    ax.errorbar(
        [large[row]["delta50"] for row in rows],
        y,
        xerr=[
            np.asarray([large[row]["delta50"] - large[row]["lo50"] for row in rows]),
            np.asarray([large[row]["hi50"] - large[row]["delta50"] for row in rows]),
        ],
        fmt="o",
        ms=3.3,
        color=CATEGORICAL[0],
        ecolor=CATEGORICAL[0],
        elinewidth=0.7,
        capsize=1.7,
        zorder=3,
        label="YOLO11s",
    )
    ax.plot(
        [base[row]["delta50"] for row in rows],
        y,
        "s",
        ms=3.0,
        color=CATEGORICAL[1],
        markeredgecolor="white",
        markeredgewidth=0.2,
        linestyle="none",
        label="YOLO11n",
    )
    ax.axvline(0, color="#444444", lw=0.65)
    ax.set_yticks(y, [LABEL[row] for row in rows])
    ax.set_ylim(len(rows) - 0.45, -0.55)
    ax.tick_params(axis="y", length=0, labelsize=5.6)
    ax.set_xlim(-0.27, 0.20)
    ax.set_xlabel(r"$\Delta$ mAP50 vs VIS source")
    clean_axes(ax, grid="x")
    ax.legend(fontsize=5.7, loc="lower right", handletextpad=0.35)
    panel_tag(ax, "a")

    ax = axes[1]
    for row in rows:
        values = [
            resolution[key]["rows"][row]["delta50"]
            for key in ("i512", "i640", "i960")
        ]
        ax.plot(
            [512, 640, 960],
            values,
            color="#B7B7B7",
            linewidth=0.55,
            alpha=0.45,
            zorder=1,
        )
    focus = [
        ("mean", "#777777", "o"),
        ("probe", CATEGORICAL[0], "s"),
        ("ifcnn", CATEGORICAL[1], "^"),
        ("spgfusion", CATEGORICAL[4], "D"),
    ]
    for row, color, marker in focus:
        values = [
            resolution[key]["rows"][row]["delta50"]
            for key in ("i512", "i640", "i960")
        ]
        ax.plot(
            [512, 640, 960],
            values,
            color=color,
            marker=marker,
            lw=1.0,
            ms=3.6,
            markeredgecolor="white",
            markeredgewidth=0.25,
            label=LABEL[row],
            zorder=3,
        )
        ax.text(
            972,
            values[-1],
            LABEL[row],
            fontsize=5.4,
            color=color,
            va="center",
            ha="left",
            clip_on=False,
        )
    ax.axhline(0, color="#444444", lw=0.65)
    ax.set_xlim(490, 990)
    ax.set_xticks([512, 640, 960])
    ax.set_xlabel("inference resolution (px)")
    ax.set_ylim(-0.27, 0.09)
    clean_axes(ax, grid="both")
    panel_tag(ax, "b")
    fig.subplots_adjust(left=0.12, right=0.87, top=0.91, bottom=0.12)
    save(fig, output, "fig_scale_v2")


def fig_corrupt(output: Path) -> None:
    data = load(AGG / "corruption_robustness.json")
    rules = ["mean", "probe", "ifcnn", "spgfusion"]
    conditions = [
        ("yolo11n_vis_clean", "clean", CATEGORICAL[2], "o"),
        ("yolo11n_vis_jpeg30", "JPEG", CATEGORICAL[0], "s"),
        ("yolo11n_vis_blur", "blur", CATEGORICAL[3], "^"),
    ]
    mm = 1 / 25.4
    fig = plt.figure(figsize=(180 * mm, 96 * mm))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.25, 0.72, 1.02],
        wspace=0.48,
    )
    ax = fig.add_subplot(gs[0])
    y = np.arange(len(rules), dtype=float)
    for index, (key, label, color, marker) in enumerate(conditions):
        offset = (index - 1) * 0.21
        values = np.asarray([data[key]["rows"][row]["delta"] for row in rules])
        lows = np.asarray([data[key]["rows"][row]["lo"] for row in rules])
        highs = np.asarray([data[key]["rows"][row]["hi"] for row in rules])
        ax.hlines(y + offset, lows, highs, color=color, lw=0.75, alpha=0.85)
        ax.plot(
            values,
            y + offset,
            marker,
            color=color,
            ms=3.3,
            markeredgecolor="white",
            markeredgewidth=0.2,
            linestyle="none",
            label=label,
        )
    ax.axvline(0, color="#444444", lw=0.65)
    ax.set_yticks(y, [LABEL[row] for row in rules])
    ax.set_ylim(len(rules) - 0.45, -0.55)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.19, 0.035)
    ax.set_xlabel(r"$\Delta$ mAP50 vs degraded source")
    ax.set_title("Clean, JPEG, blur", fontsize=6.8, pad=8)
    clean_axes(ax, grid="x")
    ax.legend(fontsize=5.4, loc="lower left", ncol=3, handletextpad=0.3)
    panel_tag(ax, "a")

    ax = fig.add_subplot(gs[1])
    noise = "yolo11n_vis_noise"
    values = np.asarray([data[noise]["rows"][row]["delta"] for row in rules])
    lows = np.asarray([data[noise]["rows"][row]["lo"] for row in rules])
    highs = np.asarray([data[noise]["rows"][row]["hi"] for row in rules])
    ax.hlines(y, lows, highs, color=ACCENT_RED, lw=0.8)
    ax.plot(
        values,
        y,
        "D",
        color=ACCENT_RED,
        ms=3.3,
        markeredgecolor="white",
        markeredgewidth=0.2,
    )
    ax.set_yticks(y, [])
    ax.set_ylim(len(rules) - 0.45, -0.55)
    ax.set_xlim(0.12, 0.38)
    ax.set_xlabel(r"$\Delta$ mAP50 (noise)")
    ax.set_title("Additive noise", fontsize=6.8, pad=8)
    clean_axes(ax, grid="x")
    panel_tag(ax, "b")

    ax = fig.add_subplot(gs[2])
    conditions_all = [
        ("yolo11n_vis_clean", "clean"),
        ("yolo11n_vis_jpeg30", "JPEG"),
        ("yolo11n_vis_blur", "blur"),
        ("yolo11n_vis_noise", "noise"),
    ]
    status_color = {
        "below": "#1B7837",
        "unresolved": "#D9D9D9",
        "above": "#F1A340",
    }
    for row_index, (key, label) in enumerate(conditions_all):
        rows = data[key]["rows"]
        below = [name for name, result in rows.items() if result["hi"] < 0]
        above = [name for name, result in rows.items() if result["lo"] > 0]
        unresolved = [
            name for name in rows if name not in below and name not in above
        ]
        sequence = (
            ["below"] * len(below)
            + ["unresolved"] * len(unresolved)
            + ["above"] * len(above)
        )
        for col_index, status in enumerate(sequence):
            ax.add_patch(
                Rectangle(
                    (col_index + 0.06, row_index - 0.30),
                    0.88,
                    0.60,
                    facecolor=status_color[status],
                    edgecolor="white",
                    linewidth=0.35,
                )
            )
    ax.set_xlim(0, 15)
    ax.set_ylim(len(conditions_all) - 0.5, -0.5)
    ax.set_yticks(range(len(conditions_all)), [label for _, label in conditions_all])
    ax.set_xticks([0, 5, 10, 15])
    ax.set_xlabel("fused inputs (of 15)")
    ax.set_title("Resolved status", fontsize=6.8, pad=8)
    clean_axes(ax, grid=None)
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=status_color[status], edgecolor="none",
                  label=status)
        for status in ("below", "unresolved", "above")
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        fontsize=5.2,
        handlelength=0.9,
        columnspacing=0.7,
    )
    panel_tag(ax, "c")

    fig.subplots_adjust(left=0.11, right=0.985, top=0.90, bottom=0.16)
    save(fig, output, "fig_corrupt_v2")


def main(output: Path) -> None:
    validate_frozen_inputs()
    fig_claim(output)
    fig_regime(output)
    fig_robust(output)
    fig_corr(output)
    fig_criterion(output)
    fig_operating(output)
    fig_tau(output)
    fig_scale(output)
    fig_corrupt(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(ROOT / "paper"),
        help="Output directory for Figures 2-10",
    )
    options = parser.parse_args()
    main(Path(options.output))
