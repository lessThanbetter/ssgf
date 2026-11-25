# Prospective confirmation of the budgeted visible-plus-PIAFusion policy

Recorded: 2026-11-26T00:48:45+08:00

## Status and timing

This protocol was written before extracting or scoring any visible or infrared
image from LLVIP sequences 080, 090, or 100. The protocol is recorded through
a local SHA-256 sidecar. It is a post-review temporal external validation and
is not externally registered.

The policy and threshold were fixed before this protocol was written. The new
holdout is used only to test that policy. It is not used to select the
candidate subset, score depth, margin, detector checkpoints, matching rule, or
endpoint.

## Data separation

- Detector training used the clean LLVIP split in sequences 010, 020, and 030.
- Sequences 040 through 070 had already been used in the earlier CS-ECF
  evaluation.
- Sequences 080, 090, and 100 are the new confirmation set.
- The new sequences contain 990, 998, and 993 images with non-empty labels,
  respectively, for a total of 2,981 images.
- The pairwise stem intersection between the new set and the clean split is
  zero.
- Sequence stem-list SHA-256 values:
  - 080: `e35f58c7e448b3ee2294c2f22c69063d119e769b366904073cb32d0bfb6fddf5`
  - 090: `749bcc3b544b583f42ec2fb79b3825096355381191b0cbc42d97ac7528a42aaf`
  - 100: `6733dd6a54b5865df7fb39fefb3afc7e78dc2afd9b134f38e8872a3d7164f5b3`
- LLVIP source archive:
  `https://hf-mirror.com/datasets/jsonhash/LLVIP/resolve/main/LLVIP.zip`

## Frozen policy

- Candidate library: visible and PIAFusion.
- Candidate score: mean of the five highest target-detector confidences.
- Score depth: `K = 5`.
- Relative selection margin: `1.5`.
- Decision: select PIAFusion only when its score is at least 1.5 times the
  visible score. Otherwise retain visible input.
- Ground truth and target labels are not used by the selection rule.
- No alternative subset, score depth, margin, candidate, or checkpoint may
  replace the frozen policy after outcomes are observed.

## Fixed artifacts

PIAFusion checkpoints:

| Component | SHA-256 |
|---|---|
| fusion model | `bd164e9d3fad604fb7a597e43a8e8f453f3d357938a77169fd56ab5eef664f11` |
| illumination model | `a9441744fc6a535d3a27495f4934efb6e52956bc42a348186f4db87b0d97dc1d` |

Detector checkpoints:

| Name | SHA-256 |
|---|---|
| YOLO26n seed 0 | `7554c3c96d1a6070ba186a7b40ed662311d6e6e9974a44ee24d71d8da6d41a68` |
| YOLO26n seed 1 | `aa42e923110624c47d8a381e6c9831045a17508774426638da57ed1f8c9fe21f` |
| YOLO26n seed 2 | `4591a148b0a378415cb5fd88b3c00b741f63667974e164cc6054a7d2725e538c` |
| YOLOv8n seed 0 | `c8086d32e291a75c56989a728dbe38ec7c5461a255d4d49d217f5d7b38c3ce89` |
| YOLOv8n seed 1 | `2384b03750c0472caddb53936ff7a1366e9f62966e48c0b15c0500cd266d63a5` |
| RT-DETR-l | `f553126314e56c406683c10643a203f8d23bbf25c7da1d12f9b7603f6a4a069f` |

## Evaluation protocol

- Detector inference uses confidence floor 0.001, NMS IoU 0.7, image size 640,
  and at most 300 detections.
- Primary AP matching is confidence-ordered one-to-one assignment: at each IoU
  threshold, predictions are visited in descending confidence order and
  matched to the best unmatched same-class ground-truth box above threshold.
- AP50 uses the 101-point interpolated precision-recall curve.
- AP50--95 averages the ten IoU thresholds from 0.50 to 0.95.

## Endpoints

Primary endpoint:

- Equal-sequence mean difference in AP50 between the frozen visible-plus-PIA
  policy and visible input over sequences 080, 090, and 100.

Secondary endpoints:

- AP50 difference for each sequence and checkpoint.
- Equal-checkpoint mean and worst checkpoint.
- AP50--95 contrast.
- Fraction of images and checkpoints on which the selected policy is above,
  equal to, or below visible input.
- Fraction of images switched from visible to PIAFusion.
- Paired stratified bootstrap interval resampling contiguous 100-frame blocks
  within each sequence and checkpoint.

The interval is conditional on three sequences and six fixed training runs.
It is not a population confidence guarantee and will be reported as
descriptive even if its endpoints exclude zero.

## Completion rules

- Every visible, PIAFusion, prediction, and score artifact used in the analysis
  must be retained or summarized in a SHA-256 manifest.
- Negative, null, or reversed outcomes are reported unchanged.
- No result from sequences 080, 090, or 100 may be used to change the frozen
  policy or to choose among alternative detectors.
