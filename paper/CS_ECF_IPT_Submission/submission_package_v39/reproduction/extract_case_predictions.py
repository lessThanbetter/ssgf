"""Extract detector boxes for the two CS-ECF qualitative cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    detector = YOLO(str(args.model))
    payload = {"detector": selection["model"], "images": {}}

    for key in ("success", "failure"):
        case = selection[key]
        sequence = case["sequence"]
        stem = case["stem"]
        candidate_dir = {
            "CDF": "consensus_intersection_pixel_mean_ifcnn_q095_a020_s05",
            "PIAFusion": "piafusion",
            "SPGFusion": "spgfusion",
        }[case["selected"]]
        split_root = args.data_root / f"erf_holdout_llvip{sequence}"
        paths = {
            "visible": split_root / "visible" / f"{stem}.jpg",
            "selected": (
                args.candidate_root
                / f"llvip{sequence}"
                / "candidates"
                / candidate_dir
                / f"{stem}.png"
            ),
        }
        for name, path in paths.items():
            result = detector.predict(
                source=str(path),
                conf=0.001,
                imgsz=640,
                device="cpu",
                verbose=False,
            )[0]
            payload["images"][f"{key}:{name}"] = {
                "boxes": result.boxes.xyxy.cpu().numpy().tolist(),
                "confidence": result.boxes.conf.cpu().numpy().tolist(),
                "classes": result.boxes.cls.cpu().numpy().astype(int).tolist(),
            }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
