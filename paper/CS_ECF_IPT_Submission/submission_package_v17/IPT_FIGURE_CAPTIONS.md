# IPT Figure Captions

Final manuscript order, 2026-10-28. Each figure has a corresponding source-data
file in `CS_ECF_IPT_Submission/source_data/`.

## Fig. 1 | Confidence-shielded evaluator-conditioned fusion

**a**, The registered visible-infrared pair is passed to visible, CDF,
PIAFusion, and SPGFusion candidates. The target detector computes the mean of
its five highest confidences for every candidate. A fused candidate is selected
only when its score is at least 1.5 times the visible score; otherwise the
visible input is retained. No ground-truth labels are used for arbitration.
**b**, Composition of the frozen input decisions across all checkpoint-image
trials for LLVIP (8 x 300), M3FD (2 x 200), and the DroneVehicle official test
subset (2 x 300). Source data: `Figure1_source.csv`.

## Fig. 2 | Threshold selection and post-freeze sensitivity

**a**, Pre-freeze validation search over seven visible margins at K = 5. The
selected policy maximizes the dataset-balanced validation gain subject to no
evaluator losing more than 0.0005 AP50. **b**, **c**, Post-freeze descriptive
sensitivity of dataset-balanced gain and the worst checkpoint loss to score
depth K and margin. Margin 1.5 was fixed before confirmatory testing; the
expanded grid was not used to reselect the policy. Source data:
`Figure2_source.csv`.

## Fig. 3 | Evaluator-dependent utility of infrared-visible fusion candidates

**a**, Signed AP50 change from the visible input for CDF, PIAFusion,
SPGFusion, the unshielded selector, and CS-ECF across 12 detector checkpoints.
Cells are point contrasts. **b**, Dataset-level mean AP50 changes. Equal
candidate sets produce opposite signs across datasets. Source data:
`Figure3_source.csv`.

## Fig. 4 | Checkpoint-level evidence and multiplicity correction

**a**, CS-ECF-minus-visible AP50 changes and scene/frame-group 95% intervals for
the 12 confirmatory checkpoints. Horizontal rules separate datasets. **b**, The
nine positive point estimates are not a collection of significant checkpoint
wins: the exact sign test gives p = 0.073, and no checkpoint survives
Benjamini-Hochberg or Holm correction at 0.05. The dataset-balanced estimate is
`+0.0133` with interval `[0.0026, 0.0261]`. Raw p-values, Holm-adjusted p-values,
and BH q-values are supplied in `Figure4_source.csv`.

## Fig. 5 | CS-ECF versus the strongest fixed fusion baseline

**a**, Dataset-level and dataset-balanced CS-ECF-minus-Always CDF AP50
contrasts with 95% group-aware intervals. The dataset-balanced interval crosses
zero. **b**, Paired checkpoint AP50 positions relative to the equality line.
**c**, Paired checkpoint contrasts. The point estimates favour CS-ECF in six
checkpoints and Always CDF in six; the comparison therefore does not establish
dominance over Always CDF. Source data: `Figure5_source.csv`.

## Fig. 6 | Label-free controls and oracle headroom

**a**, Dataset-balanced and worst-checkpoint AP50 changes for uniform random
selection, three image-quality selectors, Always CDF, CS-ECF, and a per-image
oracle. The count at the right gives positive checkpoints out of 12. **b**,
Position of CS-ECF between the mean uniform-random policy and the label-using
oracle. The oracle is an optimistic upper bound, not a deployable selector. VIF
and SSIM selected only among fused candidates; entropy selected among all four
inputs. Source data: `Figure6_source.csv`.

## Fig. 7 | External-evaluator portability stress tests

**a**, Absolute AP50 for visible, Always CDF, the unshielded selector, and
CS-ECF under two zero-shot COCO evaluators and one LLVIP-fine-tuned YOLOv9t.
**b**, Paired CS-ECF contrasts against visible and Always CDF with scene-group
95% intervals. The fine-tuned evaluator reverses the visible-relative sign, and
the Benjamini-Hochberg q-values retain the exploratory status of these external
checks. Source data: `Figure7_source.csv`.

## Fig. 8 | Calibration and switched-image failure boundary

**a**, Detection-level ECE and Brier scores for visible and CS-ECF. **b**,
Improved, unchanged, and degraded per-image AP50 among images assigned to a
fused candidate. **c**, Scene-level CS-ECF-minus-visible AP50 changes. The
LLVIP-fine-tuned evaluator has substantially higher calibration error, more
degraded switches, and two positive versus five negative scene groups. Source
data: `Figure8_source.csv`.

## Fig. 9 | Risk-return trade-off of fusion-input policies

**a**, Dataset-balanced mean AP50 change versus worst-checkpoint AP50 change.
The CS-ECF horizontal error bar is the training-run-by-group hierarchical 95%
interval. **b**, Paired checkpoint changes under the unshielded selector and
CS-ECF. The shield sacrifices part of the DroneVehicle mean gain to reduce the
worst LLVIP loss. Source data: `Figure9_source.csv`.

## Fig. 10 | Thermal constraints and class-specific effects

**a**, M3FD AP50 changes for six classes with image-bootstrap 95% intervals and
annotation counts. The gain is concentrated in people. **b**, Fixed-input
DroneVehicle CDF gain with the thermal-validity mask or global detail
injection. **c**, Spearman association between the fused-selection fraction and
thermal top-hat contrast, visible luminance, or CDF-mask activity; asterisks
denote p-values below 0.05, 0.01, and 0.001, respectively. The association
reverses on DroneVehicle, which does not support a universal thermal switching
rule. Source data: `Figure10_source.csv`.

## Fig. 11 | Runtime cost and deployment boundaries

**a**, Serial latency of candidate generation and four detector passes at the
stated hardware and input sizes; the estimate excludes model swapping, data
loading, and final post-processing. **b**, Per-stage latency and extra peak
memory, with marker area scaled by parameter count. Per-stage memory values are
not additive. **c**, Decision matrix implied by the observed evidence: Always
CDF for a fixed, calibration-stable setting; CS-ECF after held-out revalidation
when the evaluator or domain changes; visible or a stronger shield when
calibration is unstable; and no four-candidate search for latency-sensitive
deployment unless pruning or early termination is measured. Source data:
`Figure11_source.csv`.
