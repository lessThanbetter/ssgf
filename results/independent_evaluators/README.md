# Independent zero-shot evaluator check

Date: 2026-10-28

## Purpose

This directory tests the frozen CS-ECF policy on detector families that were not
used to construct the policy, select its threshold, or produce the primary
confirmatory results.

## Evaluators

| Model | Initialization | LLVIP fine-tuning | SHA-256 |
|---|---|---|---|
| YOLOv9t | COCO-pretrained Ultralytics weights | none | `410E3CACA789AA8FEC19971C05577616CBB963A416BCD0228171B2DFF7F718F7` |
| YOLOv10s | COCO-pretrained Ultralytics weights | none | `65FA8332C38A7189597A268897BCE6C1026F9499711B9761FB0C2C639B91D4D6` |

Only COCO class `0` (person) detections were retained. The frozen policy used
the same four candidate images, top-five confidence score, and 1.5 visible
margin as the primary experiment.

## Candidate generation

The visible, PIAFusion, and SPGFusion images were taken from the clean LLVIP
test archive. Pixel-mean and CDF were regenerated from the released source
pairs with:

```powershell
python scripts/generate_pixel_mean.py `
  --sources D:\Code\_ssgf_external\r2_clean\test `
  --out D:\Code\_ssgf_external\r2_clean\test_candidates\pixel_mean

python scripts/generate_consensus_detail.py `
  --sources D:\Code\_ssgf_external\r2_clean\test `
  --candidates D:\Code\_ssgf_external\r2_clean\test_candidates `
  --out D:\Code\_ssgf_external\r2_clean\test_candidates `
  --threshold 0.95 --alpha 0.20 --sigma 5 `
  --policy-names consensus_intersection_pixel_mean_ifcnn
```

## Results

| Model | Visible | CDF | No shield | CS-ECF |
|---|---:|---:|---:|---:|
| YOLOv9t | 0.7698 | 0.7888 | 0.8216 | 0.7948 |
| YOLOv10s | 0.8344 | 0.8617 | 0.8699 | 0.8557 |

CS-ECF minus visible:

- YOLOv9t: `+0.0250`, scene-group 95% CI `[+0.0053,+0.0523]`
- YOLOv10s: `+0.0213`, scene-group 95% CI `[+0.0041,+0.0439]`

CS-ECF minus Always CDF:

- YOLOv9t: `+0.0060`, scene-group 95% CI `[-0.0277,+0.0374]`
- YOLOv10s: `-0.0059`, scene-group 95% CI `[-0.0266,+0.0180]`

Both independent evaluators support improvement over visible. Neither supports
dominance over Always CDF. The shield improves the worst scene on YOLOv10s but
worsens it on YOLOv9t, so the tail-risk conclusion remains evaluator dependent.

## Domain-fine-tuned YOLOv9t boundary

A YOLOv9t initialized from COCO was fine-tuned for 40 pre-specified epochs on
the LLVIP visible training split and frozen before test scoring. The checkpoint
reached validation mAP50 `0.8456` and test visible AP50 `0.6666`.

| Evaluator | Visible | CDF | No shield | CS-ECF |
|---|---:|---:|---:|---:|
| YOLOv9t, LLVIP-trained | 0.6666 | 0.6598 | 0.6405 | 0.6558 |

- CS-ECF minus visible: `-0.0108`, scene-group CI
  `[-0.0342,+0.0050]`.
- CS-ECF minus CDF: `-0.0040`, scene-group CI
  `[-0.0206,+0.0108]`.
- CS-ECF minus no-shield: `+0.0153`, scene-group CI
  `[+0.0072,+0.0267]`.

The shield reduces, but does not remove, the loss relative to visible. This is
a negative portability boundary and is retained in the manuscript. The detailed
record is in `domain_yolov9t_v3/README.md`.

## Reproduction

The raw model outputs are in `per_image/`; fixed-candidate scores are in
`scores.json` and `multiplicity.json`; policy decisions and intervals are in
`policy_summary.json`. Regenerate the selection comparison with:

```powershell
python scripts/analyze_independent_evaluators.py
```
