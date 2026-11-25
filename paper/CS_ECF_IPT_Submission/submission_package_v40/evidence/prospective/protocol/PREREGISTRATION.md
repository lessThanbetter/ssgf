# Prospective CS-ECF confirmation protocol

Recorded: 2026-10-29T00:25:22+08:00

## Purpose

This protocol was written before scoring any candidate image from LLVIP
sequences 040, 050, 060, or 070. It defines a prospective, scene-disjoint
confirmation of the already frozen CS-ECF policy. It does not change the
original submission analyses.

## Data separation

- Detector training uses the existing LLVIP clean split in sequences 010, 020,
  and 030 only.
- Sequence 040 is the policy audit set. It may be used only to check whether
  the previously frozen margin grid again selects margin 1.5.
- Sequences 050, 060, and 070 are untouched confirmation sequences.
- No stem from sequences 040 to 070 occurs in the existing clean split.
- The LLVIP source archive is the dataset identified by the existing manifest
  and fetch script.

## Candidate set and score

The candidate set is fixed as visible, CDF, PIAFusion, and SPGFusion.
CDF uses the existing consensus-intersection pixel-mean/IFCNN construction with
`q=0.95`, `alpha=0.20`, and `sigma=5`. The candidate score remains the mean of
the five highest detector confidences, with no detections scored as `-1`.

The currently submitted policy is fixed at margin 1.5 and will be evaluated
unchanged on sequences 050, 060, and 070. No alternative margin, candidate
subset, score depth, or checkpoint may replace it after confirmation outcomes
are observed.

## Detector checkpoints

To remove validation-based checkpoint selection, this confirmation uses only
the deterministic final checkpoint from each training run.

| Run | Role | SHA-256 |
|---|---|---|
| yolo26n_clean | YOLO26n seed 0 | `7554c3c96d1a6070ba186a7b40ed662311d6e6e9974a44ee24d71d8da6d41a68` |
| yolo26n_clean_s1 | YOLO26n seed 1 | `aa42e923110624c47d8a381e6c9831045a17508774426638da57ed1f8c9fe21f` |
| yolo26n_clean_s2 | YOLO26n seed 2 | `4591a148b0a378415cb5fd88b3c00b741f63667974e164cc6054a7d2725e538c` |
| yolo8n_clean_holdout | YOLOv8n seed 0 | `c8086d32e291a75c56989a728dbe38ec7c5461a255d4d49d217f5d7b38c3ce89` |
| yolo8n_clean_holdout_s1 | YOLOv8n seed 1 | `2384b03750c0472caddb53936ff7a1366e9f62966e48c0b15c0500cd266d63a5` |
| rtdetr_l_clean | RT-DETR-l seed 0 | `f553126314e56c406683c10643a203f8d23bbf25c7da1d12f9b7603f6a4a069f` |

## Candidate-generator checkpoints

| Component | SHA-256 |
|---|---|
| PIAFusion fusion model | `bd164e9d3fad604fb7a597e43a8e8f453f3d357938a77169fd56ab5eef664f11` |
| PIAFusion illumination model | `a9441744fc6a535d3a27495f4934efb6e52956bc42a348186f4db87b0d97dc1d` |
| SPGFusion fusion model | `a2b187441a8b2457feba3e1c4f2810b9b289ddd1515bc092211dbb2ce9a77c86` |
| SPGFusion CLIP backbone | `1f16b1274ca25493b1f36225efe81d05f66a3f33c487acd1a07c5ba87920de6f` |
| IFCNN | `d59c1106739d6e12b07a14171cf099462d01e9381bbd92690bcdd4c781d79743` |

## Endpoints

Primary confirmation endpoint:

- Equal-sequence mean difference in AP50 between CS-ECF and visible across
  sequences 050, 060, and 070.

Secondary endpoints:

- Dataset-level AP50 gain for each confirmation sequence.
- Per-checkpoint AP50 gain.
- Fraction of images for which selected-candidate AP50 is higher than visible.
- Observed worst-checkpoint AP50 loss.
- Number and fraction of fused selections.
- Paired, stratified bootstrap intervals that resample only within sequence and
  checkpoint. With three confirmation sequences, the sequence-level interval is
  descriptive and conditional.

Selection audit on sequence 040:

- Apply the original margin grid
  `1.00, 1.10, 1.20, 1.30, 1.50, 1.75, 2.00` to the fixed four-candidate,
  top-five policy.
- Use the original rule, maximize equal-sequence AP50 gain subject to no
  checkpoint losing more than 0.0005 AP50.
- Report whether 1.5 is re-selected. This audit cannot change the predefined
  confirmation policy.

## Matched quality comparator

One primary image-statistic comparator is defined in advance. A
validation-tuned VIF gate uses the same four candidates, the same visible
fallback, and the same relative-margin grid. VIF is normalized by the maximum
source-reference VIF for the four inputs, and a fused candidate is selected
only when its score reaches the chosen multiple of the visible score. The
margin is selected on sequence 040 under the same per-checkpoint loss rule.
The selected VIF gate is then evaluated unchanged on sequences 050, 060, and
070.

SSIM and entropy gates with the same candidate set, visible fallback, and
margin grid are secondary controls.

## Analysis rules

- Detector inference uses confidence threshold `0.001`, NMS IoU `0.7`, image
  size 640, and the same one-to-one maximum-cardinality matching used in the
  existing AP reference implementation.
- AP is averaged over classes present in each dataset.
- All checkpoint and selector thresholds are frozen before confirmation
  sequence outcomes are inspected.
- No result from sequences 050, 060, or 070 may be used to change the policy,
  comparator, checkpoint set, endpoint, or analysis code.

## Output commitments

- Preserve every run log, source and candidate SHA-256 manifest, per-image
  detection array, selected candidate, AP summary, and interval.
- Regenerate all manuscript tables and figures from one canonical evidence
  file.
- Report negative or non-separated outcomes unchanged.
