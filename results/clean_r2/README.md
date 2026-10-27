# R2-2 independent-test evaluation

This directory records the clean-split revision requested for Reviewer 2,
comment 2. The split was fixed before retraining the missing evaluators.

## Data split

`D:\Code\_ssgf_external\r2_clean\clean\split_manifest.json` lists all stems:

| Split | Images | Pedestrian boxes |
|---|---:|---:|
| training | 1,350 | 4,479 |
| validation | 150 | 419 |
| test | 300 | 744 |

The train/test, validation/test and train/validation intersections are all zero.
The 300 test images remain the same images used by the paper's fused-image archive.

## Training

Both missing evaluators used Ultralytics 8.4.189, PyTorch 2.14.0+cu126, 40 epochs,
640 px input and seed 0. YOLO26n used batch 16. RT-DETR-l used batch 4 to fit on the
11 GB RTX 2080 Ti. The best checkpoint was selected by the library's validation
fitness on the 150 held-out validation images.

```powershell
python ssgf/scripts/train_clean_r2.py `
  --kind yolo `
  --weights D:\Code\_ssgf_external\r2_clean\weights\yolo26n.pt `
  --data D:\Code\_ssgf_external\r2_clean\clean\data.yaml `
  --project D:\Code\_ssgf_external\r2_clean\runs `
  --name yolo26n_clean `
  --epochs 40 --imgsz 640 --batch 16 --workers 4 --seed 0

python ssgf/scripts/train_clean_r2.py `
  --kind rtdetr `
  --weights D:\Code\_ssgf_external\r2_clean\weights\rtdetr-l.pt `
  --data D:\Code\_ssgf_external\r2_clean\clean\data.yaml `
  --project D:\Code\_ssgf_external\r2_clean\runs `
  --name rtdetr_l_clean `
  --epochs 40 --imgsz 640 --batch 4 --workers 4 --seed 0
```

`collect_clean_training.py` records the exact completed recipe, software versions,
validation endpoint and SHA-256 of `best.pt` and `last.pt`.

## Scoring

Each evaluator is scored on the 14 fused inputs and the visible and infrared sources.
Detections use confidence 0.001 and image size 640. Contrasts resample the same 300
images 1,000 times. BH correction is applied separately to the 14 fused-input tests
for each single-run evaluator.

```powershell
python ssgf/scripts/score_clean_evaluators.py `
  --root D:\Code\_ssgf_external\r2_clean `
  --labels D:\Code\_ssgf_external\r2_clean\test\labels `
  --stems-npz ssgf\_deposit\ssgf-llvip-fusion-detection-v1\data\per_image\vis_split__vis.npz `
  --weights "yolo26n_clean=yolo:D:\Code\_ssgf_external\r2_clean\runs\yolo26n_clean\weights\best.pt" `
  --out ssgf\results\clean_r2\yolo26n_clean `
  --bootstrap 1000 --batch-size 4

python ssgf/scripts/score_clean_evaluators.py `
  --root D:\Code\_ssgf_external\r2_clean `
  --labels D:\Code\_ssgf_external\r2_clean\test\labels `
  --stems-npz ssgf\_deposit\ssgf-llvip-fusion-detection-v1\data\per_image\vis_split__vis.npz `
  --weights "rtdetr_l_clean=rtdetr:D:\Code\_ssgf_external\r2_clean\runs\rtdetr_l_clean\weights\best.pt" `
  --out ssgf\results\clean_r2\rtdetr_l_clean `
  --bootstrap 1000 --batch-size 4
```

The two score directories are merged with `merge_clean_score_runs.py`, summarized
with `summarize_clean_r2.py`, and converted into manuscript rows with
`render_clean_r2_tables.py`.

```powershell
python ssgf/scripts/merge_clean_score_runs.py `
  --scores ssgf\results\clean_r2\yolo26n_clean ssgf\results\clean_r2\rtdetr_l_clean `
  --out ssgf\results\clean_r2\full_16

python ssgf/scripts/summarize_clean_r2.py `
  --scores ssgf\results\clean_r2\full_16\scores.json `
  --multiplicity ssgf\results\clean_r2\full_16\multiplicity.json `
  --seed-ranges ssgf\_deposit\ssgf-llvip-fusion-detection-v1\data\aggregate\clean_split_seed_ranges.json `
  --yolo11n-multiplicity ssgf\_deposit\ssgf-llvip-fusion-detection-v1\data\aggregate\convention_multiplicity.json `
  --out ssgf\results\clean_r2\clean_r2_full.json
```

## Verified result

| Evaluator | Runs | Visible-source mAP50 | Fused-input contrast range | BH result |
|---|---:|---:|---:|---|
| YOLO11n visible-only | 4 | 0.8657 | -0.136 to -0.046 | every observed run negative |
| YOLO26n visible-only | 1 | 0.8122 | -0.223 to -0.079 | 14/14 at q<0.002 |
| RT-DETR-l visible-only | 1 | 0.8095 | -0.246 to -0.070 | 14/14 at q<0.002 |
| YOLO11n modality-agnostic | 4 | 0.8660 | -0.021 to +0.029 | four stable gains |

All 42 visible-only evaluator-input aggregate comparisons are negative. The fixed
split has zero pairwise overlap, and the 300 test images contain 744 annotated
pedestrians. Training manifests are `yolo26n_clean_training.json` and
`rtdetr_l_clean_training.json`.
