"""Final Figure 1: two-stage CS-ECF framework and selection composition."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.lines import Line2D
from matplotlib.text import Text
from PIL import Image

from ipt_figure_common import (
    COLORS,
    add_panel_label,
    apply_style,
    arrow_patch,
    card,
    load_json,
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

SCRIPT_DIR = Path(__file__).resolve().parent
SSGF_ROOT = Path(
    os.environ.get("SSGF_ROOT", str(SCRIPT_DIR.parents[3]))
)
LOCAL_INPUTS = SCRIPT_DIR / "inputs"
CANDIDATE_IMAGE_FILE = LOCAL_INPUTS / "figure1_candidate_examples.npz"
FIGURE_ROOT = Path(
    os.environ.get(
        "SSGF_FIGURE_ROOT",
        str(SSGF_ROOT / "paper" / "figures_cs_ecf_ipt"),
    )
)

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

STAGE1_FILL = "#EEF5FB"
STAGE1_EDGE = "#8FB2D9"
STAGE2_FILL = "#FFF6EA"
STAGE2_EDGE = "#DDB77A"
SERIF = ["STIXGeneral", "DejaVu Serif"]


def _read_rgb(path: Path, size: tuple[int, int] = (640, 512)) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image = image.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


def _candidate_images() -> dict[str, np.ndarray]:
    if CANDIDATE_IMAGE_FILE.is_file():
        payload = np.load(CANDIDATE_IMAGE_FILE)
        return {name: payload[name] for name in payload.files}

    paths = {
        "vis": SSGF_ROOT / "data" / "erf_holdout_llvip070" / "visible" / "070142.jpg",
        "ir": SSGF_ROOT / "data" / "erf_holdout_llvip070" / "infrared" / "070142.jpg",
        "cdf": (
            SSGF_ROOT
            / "results"
            / "prospective_20261029"
            / "llvip070"
            / "candidates"
            / "consensus_intersection_pixel_mean_ifcnn_q095_a020_s05"
            / "070142.png"
        ),
        "pia": (
            SSGF_ROOT
            / "results"
            / "prospective_20261029"
            / "llvip070"
            / "candidates"
            / "piafusion"
            / "070142.png"
        ),
        "spg": (
            SSGF_ROOT
            / "results"
            / "prospective_20261029"
            / "llvip070"
            / "candidates"
            / "spgfusion"
            / "070142.png"
        ),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Figure 1 source images: " + ", ".join(missing))

    images = {name: _read_rgb(path) for name, path in paths.items()}
    LOCAL_INPUTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CANDIDATE_IMAGE_FILE, **images)
    return images


def _score_file(candidate: str) -> Path:
    name = {
        "vis": "vis",
        "cdf": "cdf",
        "pia": "piafusion",
        "spg": "spgfusion",
    }[candidate]
    return (
        SCRIPT_DIR.parent
        / "evidence"
        / "prospective"
        / "llvip070"
        / "scores"
        / f"yolo26_s0__{name}.npz"
    )


def _candidate_scores() -> dict[str, dict[str, float | int]]:
    scores = {}
    for candidate, _, _ in CANDIDATES:
        payload = np.load(_score_file(candidate), allow_pickle=True)
        stem_index = int(np.flatnonzero(payload["stems"] == "070142")[0])
        detection_indices = np.flatnonzero(payload["img_idx"] == stem_index)
        confidence = np.sort(payload["conf"][detection_indices])[::-1]
        top_five = confidence[:5]
        scores[candidate] = {
            "score": float(np.sum(top_five) / max(len(top_five), 1)),
            "detections": int(len(confidence)),
        }
    return scores


def _stage(
    ax,
    y: float,
    height: float,
    title: str,
    subtitle: str,
    facecolor: str,
    edgecolor: str,
) -> None:
    ax.add_patch(
        patches.FancyBboxPatch(
            (2.0, y),
            96.0,
            height,
            boxstyle="round,pad=0.15,rounding_size=1.8",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=0.9,
            zorder=0,
        )
    )
    ax.text(
        50.0,
        y + height - 3.3,
        title,
        ha="center",
        va="center",
        fontsize=7.4,
        fontweight="bold",
        family=SERIF,
        color="#202733",
    )
    ax.text(
        50.0,
        y + height - 7.0,
        subtitle,
        ha="center",
        va="center",
        fontsize=5.7,
        style="italic",
        family=SERIF,
        color="#536171",
    )


def _image_card(
    ax,
    image: np.ndarray,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    accent: str,
    detail: str | None = None,
) -> None:
    ax.add_patch(
        patches.FancyBboxPatch(
            (x - 0.45, y - 0.45),
            width + 0.9,
            height + 2.8,
            boxstyle="round,pad=0.08,rounding_size=0.55",
            facecolor="white",
            edgecolor="#D7DFE6",
            linewidth=0.55,
            zorder=1,
        )
    )
    ax.imshow(
        image,
        extent=(x, x + width, y, y + height),
        interpolation="lanczos",
        aspect="auto",
        zorder=2,
    )
    ax.add_patch(
        patches.Rectangle(
            (x, y),
            width,
            height,
            facecolor="none",
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
    )
    ax.add_patch(
        patches.FancyBboxPatch(
            (x - 0.25, y - 0.25),
            0.55,
            height + 0.5,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            facecolor=accent,
            edgecolor="none",
            zorder=4,
        )
    )
    ax.text(
        x + width / 2,
        y - 1.7,
        title,
        ha="center",
        va="center",
        fontsize=5.3,
        fontweight="bold",
        color=COLORS["ink"],
        zorder=5,
    )
    if detail:
        ax.text(
            x + width / 2,
            y - 3.05,
            detail,
            ha="center",
            va="center",
            fontsize=5.0,
            color=COLORS["mid"],
            zorder=5,
        )


def _chip_row(
    ax,
    x: float,
    y: float,
    label: str,
    value: float,
    color: str,
    max_value: float,
) -> None:
    ax.text(x, y, label, ha="left", va="center", fontsize=5.0, color=COLORS["ink"])
    ax.add_patch(
        patches.FancyBboxPatch(
            (x + 9.5, y - 0.65),
            8.2,
            1.3,
            boxstyle="round,pad=0.02,rounding_size=0.35",
            facecolor="#EBEFF3",
            edgecolor="none",
            zorder=1,
        )
    )
    ax.add_patch(
        patches.FancyBboxPatch(
            (x + 9.5, y - 0.65),
            8.2 * value / max_value,
            1.3,
            boxstyle="round,pad=0.02,rounding_size=0.35",
            facecolor=color,
            edgecolor="none",
            alpha=0.9,
            zorder=2,
        )
    )
    ax.text(
        x + 18.6,
        y,
        f"{value:.3f}",
        ha="left",
        va="center",
        fontsize=5.0,
        color=color,
        fontweight="bold",
    )


def _decision_gate(ax, x: float, y: float, width: float, height: float) -> None:
    points = np.asarray(
        [
            [x + width * 0.5, y + height],
            [x + width, y + height * 0.5],
            [x + width * 0.5, y],
            [x, y + height * 0.5],
        ]
    )
    ax.add_patch(
        patches.Polygon(
            points,
            closed=True,
            facecolor="#DCEAF6",
            edgecolor=COLORS["csecf"],
            linewidth=0.85,
            zorder=3,
        )
    )
    ax.text(
        x + width * 0.5,
        y + height * 0.59,
        "Use fused c* only if",
        ha="center",
        va="center",
        fontsize=5.2,
        color=COLORS["csecf"],
        fontweight="bold",
        zorder=4,
    )
    ax.text(
        x + width * 0.5,
        y + height * 0.41,
        "c* ≠ Visible and\nq(c*) ≥ 1.5 q(vis)",
        ha="center",
        va="center",
        fontsize=5.4,
        color=COLORS["csecf"],
        linespacing=1.0,
        zorder=4,
    )


def _write_example_sources(
    images: dict[str, np.ndarray],
    scores: dict[str, dict[str, float | int]],
) -> None:
    rows = [
        {
            "stem": "070142",
            "checkpoint": "YOLO26n-s0",
            "candidate": code,
            "display_candidate": label,
            "score_top5": f"{scores[code]['score']:.8f}",
            "detection_count": scores[code]["detections"],
        }
        for code, label, _ in CANDIDATES
    ]
    write_csv(
        "Figure1_example_scores.csv",
        [
            "stem",
            "checkpoint",
            "candidate",
            "display_candidate",
            "score_top5",
            "detection_count",
        ],
        rows,
    )

    image_rows = [
        ("070142", "Visible", "visible", images["vis"].shape[1], images["vis"].shape[0]),
        ("070142", "Infrared", "infrared", images["ir"].shape[1], images["ir"].shape[0]),
        ("070142", "CDF", "candidate", images["cdf"].shape[1], images["cdf"].shape[0]),
        ("070142", "PIAFusion", "candidate", images["pia"].shape[1], images["pia"].shape[0]),
        ("070142", "SPGFusion", "candidate", images["spg"].shape[1], images["spg"].shape[0]),
    ]
    write_csv(
        "Figure1_example_images.csv",
        ["stem", "input", "role", "display_width_px", "display_height_px"],
        [
            dict(zip(
                ["stem", "input", "role", "display_width_px", "display_height_px"],
                row,
                strict=True,
            ))
            for row in image_rows
        ],
    )


def main() -> None:
    apply_style()
    data = load_json("confidence_shield_three_dataset_summary.json")
    images = _candidate_images()
    scores = _candidate_scores()

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

    fig = plt.figure(figsize=(7.15, 5.45))
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[4.25, 0.92],
        hspace=0.22,
        left=0.025,
        right=0.985,
        top=0.965,
        bottom=0.075,
    )
    ax = fig.add_subplot(grid[0])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    _stage(
        ax,
        51.0,
        48.0,
        "STAGE 1: CANDIDATE GENERATION AND EVALUATOR SCORING",
        "registered VIS-IR pair → four frozen candidates → target-detector scores",
        STAGE1_FILL,
        STAGE1_EDGE,
    )

    ax.text(
        4.0,
        84.4,
        "INPUT PAIR",
        ha="left",
        va="center",
        fontsize=5.1,
        color=COLORS["mid"],
        fontweight="bold",
    )
    _image_card(ax, images["vis"], 4.0, 72.2, 10.2, 6.8, "Visible", COLORS["visible"])
    _image_card(ax, images["ir"], 4.0, 60.8, 10.2, 6.8, "Infrared", "#C6783C")
    ax.plot([14.6, 17.0], [75.6, 75.6], color=COLORS["line"], linewidth=0.65)
    ax.plot([14.6, 17.0], [64.2, 64.2], color=COLORS["line"], linewidth=0.65)
    ax.plot([17.0, 17.0], [64.2, 75.6], color=COLORS["line"], linewidth=0.65)

    ax.add_patch(
        patches.FancyBboxPatch(
            (18.0, 56.0),
            40.0,
            28.2,
            boxstyle="round,pad=0.12,rounding_size=1.15",
            facecolor="white",
            edgecolor="#DCE5EC",
            linewidth=0.55,
            zorder=1,
        )
    )
    ax.text(
        38.0,
        82.1,
        "FROZEN CANDIDATE LIBRARY",
        ha="center",
        va="center",
        fontsize=5.2,
        color=COLORS["ink"],
        fontweight="bold",
    )
    candidate_x = [19.6, 29.1, 38.6, 48.1]
    for (candidate, label, color), x in zip(CANDIDATES, candidate_x):
        image_key = "vis" if candidate == "vis" else candidate
        _image_card(
            ax,
            images[image_key],
            x,
            66.5,
            7.8,
            6.25,
            label,
            color,
            f"q={scores[candidate]['score']:.3f}",
        )
    ax.text(
        38.0,
        59.3,
        "the visible input remains available as the fallback",
        ha="center",
        va="center",
        fontsize=5.0,
        color=COLORS["mid"],
    )
    arrow_patch(ax, (55.4, 70.0), (60.8, 70.0), color=COLORS["ink"], linewidth=0.8)

    card(
        ax,
        61.5,
        65.4,
        14.0,
        12.0,
        "Target detector  D",
        "one forward pass\nper candidate",
        facecolor="#F7FAFD",
        edgecolor="#B7CCDD",
        accent=COLORS["cdf"],
        title_size=6.2,
        subtitle_size=5.1,
    )
    ax.text(
        68.5,
        61.8,
        "NO PER-IMAGE\nGROUND TRUTH",
        ha="center",
        va="center",
        fontsize=5.0,
        fontweight="bold",
        color=COLORS["cdf"],
        bbox={"boxstyle": "round,pad=0.26", "facecolor": "#EAF2F8", "edgecolor": "none"},
    )
    arrow_patch(ax, (75.8, 71.0), (79.6, 71.0), color=COLORS["ink"], linewidth=0.8)

    ax.add_patch(
        patches.FancyBboxPatch(
            (80.0, 58.0),
            17.6,
            25.4,
            boxstyle="round,pad=0.12,rounding_size=1.2",
            facecolor="white",
            edgecolor="#D8E1E8",
            linewidth=0.55,
            zorder=1,
        )
    )
    ax.text(
        88.8,
        80.8,
        "Candidate scores",
        ha="center",
        va="center",
        fontsize=5.7,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        88.8,
        77.1,
        "q(c) = top-five confidence score",
        ha="center",
        va="center",
        fontsize=5.0,
        color=COLORS["mid"],
    )
    max_score = max(float(row["score"]) for row in scores.values())
    for row_y, (candidate, label, color) in zip(
        [73.0, 69.7, 66.4, 63.1],
        CANDIDATES,
    ):
        _chip_row(
            ax,
            81.2,
            row_y,
            label,
            float(scores[candidate]["score"]),
            color,
            max_score,
        )
    ax.text(
        88.8,
        59.6,
        "example: LLVIP 070142",
        ha="center",
        va="center",
        fontsize=5.0,
        style="italic",
        color=COLORS["mid"],
    )

    arrow_patch(
        ax,
        (88.8, 57.8),
        (88.8, 50.1),
        color=COLORS["cdf"],
        linewidth=1.0,
    )
    ax.text(
        89.8,
        53.9,
        "score vector",
        ha="left",
        va="center",
        fontsize=5.0,
        color=COLORS["cdf"],
    )

    _stage(
        ax,
        2.3,
        45.0,
        "STAGE 2: CONFIDENCE-SHIELDED INPUT ARBITRATION",
        "rank candidates by score → test the visible-relative margin → select one input",
        STAGE2_FILL,
        STAGE2_EDGE,
    )

    card(
        ax,
        4.5,
        17.2,
        22.0,
        13.9,
        "Score preselection",
        "c* = argmax q(c)\nties favor Visible",
        facecolor="#FFFDF9",
        edgecolor="#E4D5BF",
        accent=COLORS["no_shield"],
        title_size=5.8,
        subtitle_size=5.3,
    )
    arrow_patch(ax, (26.8, 24.1), (33.0, 24.1), color=COLORS["ink"], linewidth=0.8)
    _decision_gate(ax, 33.2, 16.2, 20.2, 16.8)
    ax.text(
        43.3,
        13.9,
        "Example: SPG selected; 0.170 ≥ 1.5 × 0.068",
        ha="center",
        va="center",
        fontsize=5.0,
        color=COLORS["mid"],
        style="italic",
    )

    card(
        ax,
        60.5,
        26.0,
        16.5,
        8.0,
        "ACCEPT FUSED",
        "Selected input = c*",
        facecolor="#EAF4EE",
        edgecolor="#9FC7B2",
        accent=COLORS["pia"],
        title_size=5.3,
        subtitle_size=5.0,
    )
    card(
        ax,
        60.5,
        16.2,
        16.5,
        8.0,
        "RETAIN VISIBLE",
        "fallback input",
        facecolor="#EEF1F4",
        edgecolor="#CBD4DC",
        accent=COLORS["visible"],
        title_size=5.3,
        subtitle_size=5.0,
    )
    arrow_patch(
        ax,
        (53.7, 25.2),
        (60.2, 30.0),
        color=COLORS["pia"],
        linewidth=0.8,
        connectionstyle="angle3,angleA=0,angleB=-90",
    )
    arrow_patch(
        ax,
        (53.7, 22.8),
        (60.2, 20.2),
        color=COLORS["visible"],
        linewidth=0.8,
        connectionstyle="angle3,angleA=0,angleB=90",
    )
    ax.text(56.9, 32.5, "yes", ha="center", va="center", fontsize=5.0, color=COLORS["pia"])
    ax.text(56.9, 15.2, "no", ha="center", va="center", fontsize=5.0, color=COLORS["visible"])

    arrow_patch(ax, (77.3, 30.0), (81.0, 25.7), color=COLORS["pia"], linewidth=0.75)
    arrow_patch(ax, (77.3, 20.2), (81.0, 23.7), color=COLORS["visible"], linewidth=0.75)
    card(
        ax,
        81.3,
        17.0,
        15.5,
        13.0,
        "Selected input",
        "target detector\nfinal detections",
        facecolor="#E7F0F8",
        edgecolor="#A9C1D5",
        accent=COLORS["csecf"],
        title_size=5.8,
        subtitle_size=5.0,
    )
    ax.text(
        89.0,
        33.1,
        "GT-free arbitration",
        ha="center",
        va="center",
        fontsize=5.0,
        fontweight="bold",
        color=COLORS["csecf"],
    )

    add_panel_label(ax, "a", x=0.006, y=0.998, fontsize=8.0)

    ax2 = fig.add_subplot(grid[1])
    y = np.arange(len(DATASETS))
    left = np.zeros(len(DATASETS))
    for candidate, label, color in CANDIDATES:
        values = np.asarray(
            [
                counts[dataset][candidate] / sum(counts[dataset].values())
                for dataset, _ in DATASETS
            ]
        )
        ax2.barh(
            y,
            values,
            left=left,
            height=0.46,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
        )
        for row_index, value in enumerate(values):
            if value >= 0.075:
                ax2.text(
                    left[row_index] + value / 2,
                    y[row_index],
                    f"{100 * value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=5.2,
                    color="white",
                    fontweight="bold",
                    zorder=3,
                )
        left += values
    for row_index, (dataset, display) in enumerate(DATASETS):
        total = sum(counts[dataset].values())
        ax2.text(
            1.015,
            y[row_index],
            f"n={total:,}",
            ha="left",
            va="center",
            fontsize=5.0,
            color=COLORS["mid"],
        )
    ax2.set_yticks(y)
    ax2.set_yticklabels([display for _, display in DATASETS], fontsize=5.7)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1.14)
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_xticklabels(["0", "25%", "50%", "75%", "100%"], fontsize=5.0)
    ax2.set_xlabel("Fraction of checkpoint-image decisions", fontsize=5.5, labelpad=1.5)
    ax2.text(
        0.0,
        1.22,
        "Observed selection composition",
        transform=ax2.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.3,
        color=COLORS["ink"],
    )
    for side in ("top", "right", "left"):
        ax2.spines[side].set_visible(False)
    ax2.spines["bottom"].set_color(COLORS["line"])
    ax2.grid(axis="x", color="#E7ECF1", linewidth=0.45, zorder=0)
    ax2.set_axisbelow(True)
    ax2.tick_params(axis="y", length=0, pad=2)
    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.4,
            markersize=4.2,
            label=label,
        )
        for _, label, color in CANDIDATES
    ]
    ax2.legend(
        handles=handles,
        ncol=4,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.11),
        columnspacing=1.25,
        handletextpad=0.35,
        fontsize=5.1,
        borderaxespad=0.0,
    )
    add_panel_label(ax2, "b", x=-0.065, y=1.16, fontsize=7.8)

    write_csv(
        "Figure1_source.csv",
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
    _write_example_sources(images, scores)
    for artist in fig.findobj(Text):
        if artist.get_fontsize() < 6.0:
            artist.set_fontsize(6.0)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    export_kwargs = {
        "bbox_inches": "tight",
        "pad_inches": 0.02,
        "facecolor": "white",
    }
    fig.savefig(FIGURE_ROOT / "Figure1_workflow.svg", **export_kwargs)
    fig.savefig(FIGURE_ROOT / "Figure1_workflow.pdf", **export_kwargs)
    fig.savefig(FIGURE_ROOT / "Figure1_workflow.tiff", dpi=600, **export_kwargs)
    fig.savefig(FIGURE_ROOT / "Figure1_workflow.png", dpi=600, **export_kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
