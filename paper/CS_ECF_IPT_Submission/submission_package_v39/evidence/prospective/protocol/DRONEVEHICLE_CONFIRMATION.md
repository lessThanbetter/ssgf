# DroneVehicle independent OBB confirmation protocol

Recorded: 2026-10-29T01:17:14+08:00

This protocol was recorded before scoring any candidate image from the
DroneVehicle official test subset with the newly trained models.

## Source partition

The training data come from the three fixed 100-frame groups in the official
DroneVehicle validation partition, groups 4, 12, and 13. The official test
subset is separate and consists of fixed groups 7, 32, and 71 selected with
seed 2037.

Two independent runs are trained from `yolo11n-obb.pt`:

| Run | Training groups | Validation group | Seed |
|---|---|---|---|
| run1 | 4 and 12 | 13 | 0 |
| run2 | 4 and 13 | 12 | 1 |

Each run uses 30 epochs at 640 pixels, batch size 4, deterministic CuDNN, and
the native eight-coordinate polygon labels derived directly from the supplied
XML polygons.

## Evaluation

Each final checkpoint is evaluated once on the fixed official test subset.
The fixed policy is top-five mean detector confidence with a 1.5
visible-relative margin and visible fallback. Comparisons are visible, Always
CDF, no-shield selection, and CS-ECF.

The metric is native DroneVehicle oriented-box AP50. Matching uses
probabilistic rotated-box IoU and one-to-one maximum-cardinality assignment at
IoU thresholds from 0.50 to 0.95. The three 100-frame test groups are weighted
equally. Per-run and equal-run summaries are required; no result may be
discarded based on sign.

## Expected outputs

- Model training arguments, logs, and SHA-256 values.
- Per-image true-positive flags and confidence arrays for every candidate.
- Native OBB scoring adapter and software versions.
- Per-block and checkpoint summaries.
- Conditional group-resampling intervals.
