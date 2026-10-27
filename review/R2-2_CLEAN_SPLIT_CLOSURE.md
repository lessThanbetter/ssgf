# Reviewer 2, comment 2: clean-split closure

## Comment

For LLVIP, several detectors used the same 300 images that were evaluated as their
validation set. The revised protocol had not been applied to every major evaluator and
comparison. The reviewer requested a test set fully independent of training and
validation as the primary result.

## Completed work

- Fixed one split before retraining the missing evaluators: 1,350 training, 150
  validation and 300 test images.
- Verified zero train/test, validation/test and train/validation stem overlap.
- Verified 4,479 training, 419 validation and 744 test pedestrian annotations.
- Retrained YOLO26n for 40 epochs at 640 px, batch 16 and seed 0.
- Retrained RT-DETR-l for 40 epochs at 640 px, batch 4 and seed 0 to fit the 11 GB GPU.
- Retained the four-run clean YOLO11n visible-only evaluator and four-run clean
  modality-agnostic YOLO11n control.
- Scored all 14 fused inputs plus the visible and infrared sources under both newly
  trained evaluators, producing 32 per-image result files.
- Applied a separate 14-test Benjamini-Hochberg family to each one-run clean evaluator.
- Repeated the six-stratum 84-test BH analysis under all three clean visible-only
  evaluators.

## Primary result

| Evaluator | Runs | Visible-source mAP50 | Contrast range | Resolution |
|---|---:|---:|---:|---|
| YOLO11n visible-only | 4 | 0.8657 | -0.136 to -0.046 | every observed run negative |
| YOLO26n visible-only | 1 | 0.8122 | -0.223 to -0.079 | 14/14 at q<0.002 |
| RT-DETR-l visible-only | 1 | 0.8095 | -0.246 to -0.070 | 14/14 at q<0.002 |
| YOLO11n modality-agnostic | 4 | 0.8660 | -0.021 to +0.029 | four stable gains |

All 42 visible-only evaluator-input aggregate comparisons are negative. The clean
deficits span -0.046 to -0.246 mAP50.

## Claim correction

The independent tests support the aggregate negative result under all three
visible-trained evaluators. They do not support a single architecture-invariant
stratification. For the six regimes in the order low contrast, high contrast, one
pedestrian, three or more pedestrians, small targets and large targets, the numbers
resolved below are:

- clean YOLO11n: 1, 1, 0, 3, 2, 0;
- clean YOLO26n: 13, 14, 11, 14, 14, 14;
- clean RT-DETR-l: 2, 14, 2, 14, 14, 5.

The manuscript now labels the pooled stratification as a sensitivity analysis and keeps
the clean stratum results exploratory.

## Reproducibility artifacts

- Split manifest:
  `_deposit/ssgf-llvip-fusion-detection-v1/data/splits/clean_split_manifest.json`
- Combined results: `results/clean_r2/clean_r2_full.json`
- Correlations: `results/clean_r2/clean_r2_correlations.json`
- Training manifests: `results/clean_r2/*_training.json`
- Per-image scores: `results/clean_r2/full_16/per_image/`
- Checkpoints: `data/det_clean/`
- Verification: `python scripts/verify_clean_r2.py --root .`

## Remaining qualification

YOLO26n and RT-DETR-l have one clean training run each. YOLO11n has four. The manuscript
states this explicitly and does not claim that single-run differences between YOLO26n
and RT-DETR-l are seed-stable.
