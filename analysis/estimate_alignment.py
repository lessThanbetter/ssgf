"""Estimate a per-pair translational registration residual for source images."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


def edge_features(path: Path, max_side: int):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"cannot read {path}")
    h, w = image.shape
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, round(w * scale)), max(1, round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    image = cv2.GaussianBlur(image.astype(np.float32), (0, 0), 1.2)
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    edges = np.sqrt(gx * gx + gy * gy)
    high = float(np.percentile(edges, 99))
    edges = np.clip(edges / (high + 1e-8), 0, 1)
    return ((edges - edges.mean()) / (edges.std() + 1e-8)).astype(np.float32), (
        w,
        h,
    )


def estimate_pair(visible: Path, infrared: Path, max_side: int):
    template, (source_w, _) = edge_features(visible, max_side)
    moving, _ = edge_features(infrared, max_side)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        250,
        1e-6,
    )
    try:
        correlation, warp = cv2.findTransformECC(
            template,
            moving,
            warp,
            cv2.MOTION_TRANSLATION,
            criteria,
            None,
            5,
        )
    except cv2.error as error:
        return {
            "status": "failed",
            "ecc": float("nan"),
            "shift_x_px": float("nan"),
            "shift_y_px": float("nan"),
            "residual_px": float("nan"),
            "error": error.err.strip(),
        }
    resized_w = template.shape[1]
    scale = source_w / resized_w
    dx = float(warp[0, 2] * scale)
    dy = float(warp[1, 2] * scale)
    return {
        "status": "ok",
        "ecc": float(correlation),
        "shift_x_px": dx,
        "shift_y_px": dy,
        "residual_px": math.hypot(dx, dy),
        "error": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", required=True,
                        help="directory containing visible/ and infrared/")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-side", type=int, default=320)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.sources)
    visible = {path.stem: path for path in (root / "visible").glob("*.jpg")}
    infrared = {path.stem: path for path in (root / "infrared").glob("*.jpg")}
    stems = sorted(set(visible) & set(infrared))
    if args.limit:
        stems = stems[: args.limit]
    if not stems:
        raise SystemExit(f"no matched LLVIP sources under {root}")

    rows = []
    for index, stem in enumerate(stems, start=1):
        result = estimate_pair(visible[stem], infrared[stem], args.max_side)
        rows.append({"stem": stem, **result})
        if index % 25 == 0 or index == len(stems):
            print(f"[alignment] {index}/{len(stems)}", flush=True)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "n": len(rows),
        "ok": sum(row["status"] == "ok" for row in rows),
        "method": "ECC on normalized Sobel-edge maps, Euclidean translation",
        "max_side": args.max_side,
    }
    if summary["ok"]:
        valid = [row for row in rows if row["status"] == "ok"]
        summary.update({
            "residual_median_px": float(np.median(
                [row["residual_px"] for row in valid]
            )),
            "residual_p90_px": float(np.percentile(
                [row["residual_px"] for row in valid], 90
            )),
            "ecc_median": float(np.median([row["ecc"] for row in valid])),
        })
    output.with_suffix(".json").write_text(
        json.dumps({"summary": summary, "pairs": rows}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
