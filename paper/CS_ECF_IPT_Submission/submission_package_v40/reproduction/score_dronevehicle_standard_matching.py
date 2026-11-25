"""Score independent DroneVehicle runs with confidence-ordered OBB matching."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from rescore_dronevehicle_standard_matching import confidence_ordered_match


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from local_ap_reference import ap_per_class  # noqa: E402
from score_dronevehicle_native_obb import (  # noqa: E402
    CDF_NAME,
    find_image,
    load_ground_truth,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(r"D:\Code\_ssgf_external\dronevehicle_test300"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(r"D:\Code\_ssgf_external\dronevehicle_test300\manifest.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "prospective_20261029"
        / "dronevehicle" / "standard_matching_new_runs",
    )
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    stems = [
        str(stem)
        for stem in json.loads(
            args.manifest.read_text(encoding="utf-8")
        )["stems"]["test"]
    ]
    if args.limit:
        stems = stems[: args.limit]
    candidates = {
        "vis": args.source / "visible",
        "cdf": args.source / "erf_candidates" / CDF_NAME,
        "pia": args.source / "sota_candidates" / "piafusion",
        "spg": args.source / "sota_candidates" / "spgfusion",
    }
    ground_truth = {
        stem: load_ground_truth(args.source / "xml" / f"{stem}.xml")
        for stem in stems
    }
    args.out.mkdir(parents=True, exist_ok=True)

    for specification in args.model:
        name, path = specification.split("=", 1)
        model = YOLO(path)
        summary = {}
        for candidate, directory in candidates.items():
            paths = [find_image(directory, stem) for stem in stems]
            rows = []
            started = time.time()
            for offset in range(0, len(paths), args.batch_size):
                batch_paths = paths[offset:offset + args.batch_size]
                predictions = model.predict(
                    [str(path) for path in batch_paths],
                    conf=0.001,
                    imgsz=640,
                    device=args.device,
                    verbose=False,
                )
                for local, prediction in enumerate(predictions):
                    stem = stems[offset + local]
                    obb = prediction.obb
                    pred_polygons = obb.xyxyxyxy.cpu().numpy().astype(
                        np.float32
                    )
                    pred_classes = obb.cls.cpu().numpy().astype(np.int64)
                    confidence = obb.conf.cpu().numpy().astype(np.float32)
                    gt_polygons, gt_classes = ground_truth[stem]
                    rows.append((
                        confidence_ordered_match(
                            pred_polygons,
                            pred_classes,
                            confidence,
                            gt_polygons,
                            gt_classes,
                        ),
                        confidence,
                        pred_classes,
                        gt_classes,
                    ))
            tp = np.concatenate([row[0] for row in rows], axis=0)
            conf = np.concatenate([row[1] for row in rows], axis=0)
            pred_cls = np.concatenate([row[2] for row in rows], axis=0)
            gt_cls = np.concatenate([row[3] for row in rows], axis=0)
            mean_50, mean_95 = ap_per_class(tp, conf, pred_cls, gt_cls)
            np.savez_compressed(
                args.out / f"{name}__{candidate}.npz",
                tp=tp,
                conf=conf,
                pcls=pred_cls,
                img_idx=np.concatenate([
                    np.full(len(row[1]), index, np.int32)
                    for index, row in enumerate(rows)
                ]),
                tcls=gt_cls,
                stems=np.asarray(stems),
                n_img=len(stems),
                n_gt=np.asarray(
                    [len(row[3]) for row in rows],
                    dtype=np.int32,
                ),
            )
            summary[candidate] = {
                "mAP50": mean_50,
                "mAP50_95": mean_95,
                "seconds": time.time() - started,
            }
            print(
                f"{name} {candidate} mAP50={mean_50:.4f} "
                f"mAP50-95={mean_95:.4f}",
                flush=True,
            )
        (args.out / f"{name}_scores.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )

    (args.out / "protocol.json").write_text(
        json.dumps(
            {
                "matching": "confidence-ordered one-to-one matching",
                "iou": "probabilistic rotated-box IoU",
                "n_images": len(stems),
                "models": args.model,
                "device": args.device,
                "confidence_floor": 0.001,
                "nms_iou": 0.7,
                "image_size": 640,
                "status": "post-review sensitivity analysis",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
