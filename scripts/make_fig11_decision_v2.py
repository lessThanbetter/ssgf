"""Draw the final Measurement Figure 11 schematic from verified results."""

# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# schematic -> assets/figures/other -> cross-type inherit with parameter inherit
# No renderable schematic production script is distributed in the asset package.

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


import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


EVALUATORS = (
    ("YOLO11n", "yolo11n_visible_clean", CATEGORICAL[0]),
    ("YOLO26n", "yolo26n_visible_clean", CATEGORICAL[0]),
    ("RT-DETR-l", "rtdetr_l_visible_clean", CATEGORICAL[0]),
    ("agnostic", "yolo11n_modality_agnostic_clean", CATEGORICAL[4]),
)


def verified_contrasts() -> tuple[list[dict], int, int]:
    combined = read_json(ROOT / "results/clean_r2/clean_r2_full.json")
    expected_variants = tuple(
        combined["evaluators"]["yolo11n_visible_clean"]["inputs"]
    )
    rows = []
    for label, key, color in EVALUATORS:
        evaluator = combined["evaluators"][key]
        if tuple(evaluator["inputs"]) != expected_variants:
            raise RuntimeError(f"input order differs for {key}")
        rows.append(
            {
                "label": label,
                "color": color,
                "inputs": evaluator["inputs"],
            }
        )

    visible_total = sum(
        len(row["inputs"]) for row in rows[:3]
    )
    visible_negative = sum(
        value["mean_contrast"] < 0
        for row in rows[:3]
        for value in row["inputs"].values()
    )

    stable_positive = sum(
        all(value > 0 for value in result.get("per_run_contrast", [result["mean_contrast"]]))
        for result in rows[3]["inputs"].values()
    )
    if (visible_total, visible_negative, stable_positive) != (42, 42, 4):
        raise RuntimeError("independent-test counts are inconsistent")
    return rows, visible_total, stable_positive


def verified_ranks() -> tuple[list[dict], dict[str, int], dict[str, int]]:
    vif = read_json(ROOT / "results/reference_vif/llvip_14_full.json")["rows"]
    reported_rank = sorted(vif, key=lambda row: row["reported"], reverse=True)
    reference_rank = sorted(vif, key=lambda row: row["reference"], reverse=True)
    reported_first = next(i for i, row in enumerate(reported_rank, 1) if row["variant"] == "spgfusion")
    reference_last = next(
        i for i, row in enumerate(reference_rank, 1) if row["variant"] == "spgfusion"
    )
    if (reported_first, len(reference_rank)) != (1, 14) or reference_last != 14:
        raise RuntimeError("VIF endpoint ranks are inconsistent")
    reported_positions = {
        row["variant"]: rank for rank, row in enumerate(reported_rank, 1)
    }
    reference_positions = {
        row["variant"]: rank for rank, row in enumerate(reference_rank, 1)
    }
    return vif, reported_positions, reference_positions


def arrow(ax, start, end) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=0.75,
            color="#666666",
            shrinkA=0,
            shrinkB=0,
            zorder=1,
        )
    )


def add_thumbnail(ax, path: Path, x: float, y: float, width: float, height: float, title: str) -> None:
    with Image.open(path) as image:
        array = np.asarray(image.crop((120, 480, 740, 860))).copy()
    ax.imshow(
        array,
        extent=(x, x + width, y, y + height),
        interpolation="lanczos",
        aspect="auto",
        cmap="gray" if array.ndim == 2 else None,
        vmin=0 if array.ndim == 2 else None,
        vmax=255 if array.ndim == 2 else None,
        zorder=2,
    )
    ax.add_patch(
        plt.Rectangle(
            (x, y),
            width,
            height,
            facecolor="none",
            edgecolor="#666666",
            linewidth=0.65,
            zorder=3,
        )
    )
    ax.text(
        x + width / 2,
        y - 2.4,
        title,
        ha="center",
        va="center",
        fontsize=5.8,
        color=BLACK,
        zorder=3,
    )


def draw_evaluator_panel(ax, rows: list[dict]) -> None:
    y_positions = np.asarray([4.0, 3.0, 2.0, 1.0])
    ax.axhspan(1.55, 4.45, facecolor="#F3F7FB", edgecolor="none", zorder=0)
    ax.axhspan(0.55, 1.45, facecolor="#F7F2F8", edgecolor="none", zorder=0)

    for row, y in zip(rows, y_positions):
        for result in row["inputs"].values():
            mean = float(result["mean_contrast"])
            values = result.get("per_run_contrast") or [mean]
            low, high = min(values), max(values)
            stable_gain = y == 1.0 and all(value > 0 for value in values)
            marker_color = ACCENT_RED if stable_gain else row["color"]
            ax.plot(
                [low, high],
                [y, y],
                color=marker_color,
                alpha=0.42,
                linewidth=0.65,
                solid_capstyle="round",
                zorder=2,
            )
            ax.scatter(
                [mean],
                [y],
                s=8.5,
                color=marker_color,
                edgecolor="white",
                linewidth=0.25,
                zorder=3,
            )

    ax.axvline(0, color="#333333", linewidth=0.65, zorder=1)
    ax.set_xlim(-0.265, 0.055)
    ax.set_ylim(0.55, 4.45)
    ax.set_yticks(y_positions, [row["label"] for row in rows])
    ax.set_xticks([-0.2, -0.1, 0])
    ax.set_xticklabels(["-0.2", "-0.1", "0"])
    ax.tick_params(axis="y", length=0, pad=2, labelsize=5.6)
    ax.tick_params(axis="x", length=2.0, width=0.45, pad=1.5, labelsize=5.3)
    ax.set_xlabel("mAP50 contrast", fontsize=5.5, labelpad=1.5)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def draw_rank_panel(ax, rows: list[dict], reported: dict[str, int], reference: dict[str, int]) -> None:
    ax.set_xlim(-0.24, 1.24)
    ax.set_ylim(14.7, 0.3)
    for row in rows:
        variant = row["variant"]
        left_rank = reported[variant]
        right_rank = reference[variant]
        highlighted = variant == "spgfusion"
        color = ACCENT_RED if highlighted else "#9A9A9A"
        ax.plot(
            [0, 1],
            [left_rank, right_rank],
            color=color,
            linewidth=1.25 if highlighted else 0.55,
            alpha=0.95 if highlighted else 0.38,
            zorder=2 if highlighted else 1,
        )
        ax.scatter(
            [0, 1],
            [left_rank, right_rank],
            s=11 if highlighted else 5.5,
            color=color,
            edgecolor="white",
            linewidth=0.25,
            alpha=0.95 if highlighted else 0.55,
            zorder=3 if highlighted else 2,
        )

    ax.text(-0.07, 0.72, "SPG", ha="right", va="center",
            fontsize=5.4, fontweight="bold", color=ACCENT_RED)
    ax.text(1.07, 14.30, "SPG", ha="left", va="center",
            fontsize=5.4, fontweight="bold", color=ACCENT_RED)
    ax.set_xticks([0, 1], ["reported", "reference"])
    ax.set_yticks([])
    ax.text(-0.20, 0.82, "1", ha="right", va="center",
            fontsize=5.3, color="#666666")
    ax.text(-0.20, 14.18, "14", ha="right", va="center",
            fontsize=5.3, color="#666666")
    ax.tick_params(axis="x", length=0, pad=2, labelsize=5.4, colors="#333333")
    ax.tick_params(axis="y", length=0, pad=2, labelsize=5.3, colors="#666666")
    ax.set_ylabel("rank", fontsize=5.5, labelpad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)


def main(source_root: Path, output: Path) -> None:
    evaluator_rows, _total, _stable_positive = verified_contrasts()
    rank_rows, reported_ranks, reference_ranks = verified_ranks()
    mm_to_inch = 1 / 25.4
    fig, ax = plt.subplots(figsize=(88 * mm_to_inch, 96 * mm_to_inch))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    add_thumbnail(
        ax,
        source_root / "test/visible/190002.jpg",
        2,
        79,
        26,
        17,
        "VIS",
    )
    add_thumbnail(
        ax,
        source_root / "test/infrared/190002.jpg",
        37,
        79,
        26,
        17,
        "IR",
    )
    add_thumbnail(
        ax,
        source_root / "k1_native/imgs/mean/190002.png",
        72,
        79,
        26,
        17,
        "FUSED",
    )
    ax.text(32.5, 87.5, "+", ha="center", va="center", fontsize=9, color="#555555")
    arrow(ax, (64.0, 87.5), (70.8, 87.5))

    ax.text(
        1.0,
        98.0,
        "(a)",
        ha="left",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        color=BLACK,
    )
    ax.plot([2, 98], [73.0, 73.0], color="#777777", lw=0.55, zorder=1)
    arrow(ax, (25.0, 73.0), (25.0, 67.2))
    arrow(ax, (74.0, 73.0), (74.0, 67.2))

    ax.text(3.0, 64.9, "(b)", ha="left", va="center", fontsize=7.2,
            fontweight="bold", color=BLACK)
    ax.text(8.0, 64.9, "Evaluator", ha="left", va="center", fontsize=7.0,
            fontweight="bold", color=CATEGORICAL[0])
    ax.text(54.0, 64.9, "(c)", ha="left", va="center", fontsize=7.2,
            fontweight="bold", color=BLACK)
    ax.text(59.0, 64.9, "Fidelity index", ha="left", va="center", fontsize=7.0,
            fontweight="bold", color=CATEGORICAL[4])

    evaluator_ax = fig.add_axes([0.125, 0.325, 0.385, 0.255])
    rank_ax = fig.add_axes([0.665, 0.345, 0.245, 0.240])
    draw_evaluator_panel(evaluator_ax, evaluator_rows)
    draw_rank_panel(rank_ax, rank_rows, reported_ranks, reference_ranks)

    ax.plot([4, 96], [20.5, 20.5], color="#CCCCCC", lw=0.45, zorder=1)
    ax.plot([4, 96], [8.5, 8.5], color="#CCCCCC", lw=0.45, zorder=1)
    ax.text(50, 14.5, "RANKING  =  f(EVALUATOR, INDEX, TEST SPLIT)",
            ha="center", va="center", fontsize=6.8, fontweight="bold",
            color=BLACK, zorder=3)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_cns_figure(fig, str(output))
    fig.savefig(f"{output}.svg", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        default=r"D:\Code\_ssgf_external\r2_clean",
        help="Directory containing test/visible, test/infrared and k1_native/imgs",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "paper" / "fig_decision_v2"),
        help="Output path without the file suffix",
    )
    options = parser.parse_args()
    main(Path(options.source_root), Path(options.output))
