"""Build the successful and failed CS-ECF switch case-study figure."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 6.4,
        "axes.linewidth": 0.6,
        "axes.unicode_minus": False,
    }
)


def load_ground_truth(
    path: Path,
    width: int,
    height: int,
) -> list[tuple[int, tuple[float, ...]]]:
    boxes = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        class_id, xc, yc, bw, bh = raw.split()
        xc, yc, bw, bh = map(float, (xc, yc, bw, bh))
        boxes.append(
            (
                int(float(class_id)),
                (
                    (xc - bw / 2.0) * width,
                    (yc - bh / 2.0) * height,
                    (xc + bw / 2.0) * width,
                    (yc + bh / 2.0) * height,
                ),
            )
        )
    return boxes


def box_iou(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(
        0.0,
        first[3] - first[1],
    )
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0,
        second[3] - second[1],
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def matched_true_positives(
    predictions: dict,
    ground_truth: list[tuple[int, tuple[float, ...]]],
    iou_threshold: float = 0.5,
) -> dict[int, int]:
    unmatched_ground_truth = set(range(len(ground_truth)))
    matches = {}
    for prediction_index, (box, class_id) in enumerate(zip(
        predictions["boxes"],
        predictions["classes"],
    )):
        best = None
        for ground_truth_index in unmatched_ground_truth:
            gt_class, gt_box = ground_truth[ground_truth_index]
            if int(class_id) != int(gt_class):
                continue
            overlap = box_iou(box, gt_box)
            if overlap >= iou_threshold and (
                best is None or overlap > best[1]
            ):
                best = (ground_truth_index, overlap)
        if best is not None:
            unmatched_ground_truth.remove(best[0])
            matches[prediction_index] = best[0]
    return matches


def draw_overlay(
    ax,
    image: Path,
    predictions: dict,
    label_path: Path,
) -> int:
    with Image.open(image) as payload:
        pixels = np.asarray(payload.convert("RGB"))
    height, width = pixels.shape[:2]
    ax.imshow(pixels)

    ground_truth = load_ground_truth(label_path, width, height)
    for _, (x1, y1, x2, y2) in ground_truth:
        ax.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="#2CA25F",
                linewidth=0.75,
                linestyle=(0, (2, 1.5)),
            )
        )

    matches = matched_true_positives(predictions, ground_truth)
    for prediction_index, (box, confidence) in enumerate(zip(
        predictions["boxes"],
        predictions["confidence"],
    )):
        if prediction_index not in matches:
            continue
        x1, y1, x2, y2 = box
        ax.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="#E07A28",
                linewidth=0.8,
            )
        )
        ax.text(
            x1,
            max(0, y1 - 2),
            f"{confidence:.2f}",
            color="white",
            fontsize=5.2,
            ha="left",
            va="bottom",
            bbox={"facecolor": "#E07A28", "edgecolor": "none", "pad": 0.4},
        )

    ax.set_axis_off()
    return len(matches)


def make_figure(
    selection: dict,
    predictions: dict,
    data_root: Path,
    candidate_root: Path,
    output: Path,
    source_output: Path,
) -> None:
    rows = (("success", "Successful switch"), ("failure", "Failed switch"))
    fig = plt.figure(figsize=(7.09, 3.75), facecolor="white")
    grid = fig.add_gridspec(
        2,
        4,
        width_ratios=(1.0, 1.0, 1.0, 0.78),
        wspace=0.035,
        hspace=0.09,
        left=0.15,
        right=0.985,
        top=0.93,
        bottom=0.08,
    )
    columns = ("Visible input", "Infrared input", "Selected fusion", "AP50 change")
    title_positions = (0.15, 0.355, 0.56, 0.765)
    for title_x, title in zip(title_positions, columns):
        fig.text(
            title_x,
            0.955,
            title,
            fontsize=6.5,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    source_rows = []
    for row_index, (key, row_label) in enumerate(rows):
        case = selection[key]
        sequence = case["sequence"]
        stem = case["stem"]
        candidate_name = case["selected"]
        candidate_dir = {
            "CDF": "consensus_intersection_pixel_mean_ifcnn_q095_a020_s05",
            "PIAFusion": "piafusion",
            "SPGFusion": "spgfusion",
        }[candidate_name]
        split_root = data_root / f"erf_holdout_llvip{sequence}"
        visible = split_root / "visible" / f"{stem}.jpg"
        infrared = split_root / "infrared" / f"{stem}.jpg"
        label = split_root / "labels" / f"{stem}.txt"
        fused = candidate_root / f"llvip{sequence}" / "candidates" / candidate_dir / f"{stem}.png"
        for path in (visible, infrared, label, fused):
            if not path.is_file():
                raise FileNotFoundError(path)

        visible_true_positives = draw_overlay(
            fig.add_subplot(grid[row_index, 0]),
            visible,
            predictions["images"][f"{key}:visible"],
            label,
        )
        with Image.open(infrared) as payload:
            ax_ir = fig.add_subplot(grid[row_index, 1])
            ax_ir.imshow(np.asarray(payload.convert("L")), cmap="gray")
            ax_ir.set_axis_off()
        selected_true_positives = draw_overlay(
            fig.add_subplot(grid[row_index, 2]),
            fused,
            predictions["images"][f"{key}:selected"],
            label,
        )

        ax_delta = fig.add_subplot(grid[row_index, 3])
        gain = 100.0 * float(case["ap50_gain"])
        ax_delta.barh(
            [1],
            [gain],
            color="#2CA25F" if gain > 0 else "#C94C4C",
            height=0.45,
        )
        ax_delta.axvline(0, color="#333333", linewidth=0.6)
        ax_delta.set_xlim(-45, 55)
        ax_delta.set_yticks([])
        fig.text(
            0.012,
            0.68 if row_index == 0 else 0.33,
            (
                f"{row_label}\n"
                f"matched TP {visible_true_positives}/"
                f"{case['n_gt']} to {selected_true_positives}/"
                f"{case['n_gt']}\n"
                f"confidence ratio "
                f"{case['selected_score'] / max(case['visible_score'], 1e-9):.2f}\n"
                f"AP50 {100 * case['ap50_gain']:+.1f} pp"
            ),
            fontsize=6.5,
            fontweight="bold",
            ha="left",
            va="center",
            color="#246A42" if key == "success" else "#8D2E2E",
        )
        ax_delta.set_xlabel("AP50 change (percentage points)", fontsize=5.8)
        ax_delta.tick_params(axis="x", labelsize=5.2)
        ax_delta.tick_params(axis="y", length=0)
        ax_delta.text(
            gain + (1.5 if gain >= 0 else -1.5),
            1,
            f"{gain:+.1f}",
            va="center",
            ha="left" if gain >= 0 else "right",
            fontsize=6,
            fontweight="bold",
            color="#246A42" if gain > 0 else "#8D2E2E",
        )
        ax_delta.spines[["right", "top", "left"]].set_visible(False)
        source_rows.append(
            {
                "case": key,
                "sequence": sequence,
                "stem": stem,
                "selected_input": candidate_name,
                "visible_top5_confidence": round(case["visible_score"], 6),
                "selected_top5_confidence": round(case["selected_score"], 6),
                "selected_ratio": round(
                    case["selected_score"] / max(case["visible_score"], 1e-9), 6
                ),
                "visible_ap50": round(case["visible_ap50"], 6),
                "selected_ap50": round(case["selected_ap50"], 6),
                "ap50_change": round(case["ap50_gain"], 6),
                "ground_truth_count": case["n_gt"],
                "detector": selection["model"],
                "visible_true_positives": visible_true_positives,
                "selected_true_positives": selected_true_positives,
                "visible_max_confidence": round(
                    max(predictions["images"][f"{key}:visible"]["confidence"]),
                    6,
                ),
                "selected_max_confidence": round(
                    max(predictions["images"][f"{key}:selected"]["confidence"]),
                    6,
                ),
            }
        )

    legend = [
        Line2D(
            [0],
            [0],
            color="#E07A28",
            lw=1.0,
            label="Matched true-positive prediction at IoU ≥0.50",
        ),
        Line2D(
            [0],
            [0],
            color="#2CA25F",
            lw=1.0,
            linestyle=(0, (2, 1.5)),
            label="Ground truth",
        ),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=5.9,
        bbox_to_anchor=(0.52, 0.005),
        handlelength=1.5,
        columnspacing=1.4,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)

    source_output.parent.mkdir(parents=True, exist_ok=True)
    with source_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=source_rows[0].keys())
        writer.writeheader()
        writer.writerows(source_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-out", type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    make_figure(
        selection,
        predictions,
        args.data_root,
        args.candidate_root,
        args.out,
        args.source_out,
    )


if __name__ == "__main__":
    main()
