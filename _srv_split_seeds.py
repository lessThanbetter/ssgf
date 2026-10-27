"""Extra seeds for the clean-split detectors, to test the modality reversal across seeds.

Same recipe as the reported clean split: YOLO11n, 40 epochs, batch 16, 640 px, on
`datasets/det_split/vis_only.yaml` and `datasets/det_split/both.yaml`, which hold 1350
training and 150 validation pairs with the 300 test images untouched. The reported run
used seed 0; this adds seeds 1, 2 and 3 for both detectors.
"""
import os

BASE = "/root/autodl-tmp"
PROJ = f"{BASE}/ssgf/_split_runs"
WEIGHTS = [f"{BASE}/ultralytics-main/yolo11n.pt", f"{BASE}/ssgf/yolo11n.pt"]
JOBS = [
    ("vis_split_s%d", f"{BASE}/datasets/det_split/vis_only.yaml"),
    ("agn_split_s%d", f"{BASE}/datasets/det_split/both.yaml"),
]
SEEDS = (1, 2, 3)


def main():
    from ultralytics import YOLO

    weights = next((w for w in WEIGHTS if os.path.exists(w)), None)
    if weights is None:
        raise SystemExit("no YOLO11n base weights found")
    print("base weights:", weights, flush=True)
    for seed in SEEDS:
        for name_t, data in JOBS:
            name = name_t % seed
            best = os.path.join(PROJ, name, "weights", "best.pt")
            if os.path.exists(best):
                print(f"{name}: exists, skip", flush=True)
                continue
            print(f"=== {name} seed {seed} ===", flush=True)
            YOLO(weights).train(data=data, epochs=40, imgsz=640, batch=16, seed=seed,
                                device=0, workers=4, verbose=False, project=PROJ,
                                name=name, exist_ok=True)
            print(f"{name}: finished, best exists={os.path.exists(best)}", flush=True)
    print("SPLIT_SEEDS_DONE", flush=True)


if __name__ == "__main__":
    main()
