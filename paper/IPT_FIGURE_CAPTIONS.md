# IPT Figure Captions

## Fig. 1 | Evaluator-dependent utility of infrared–visible fusion candidates

**a**, Signed AP50 change from the visible input for CDF, PIAFusion, SPGFusion,
the unshielded top-five selector, and CS-ECF across 12 detector checkpoints.
Cells are point AP50 contrasts; no cell-level significance test is implied.
**b**, Dataset-level mean AP50 changes for the same policies. Equal candidate
sets produce opposite signs on LLVIP, M3FD, and DroneVehicle, showing that fusion
utility depends on the detector evaluator. Source data are provided in
`figure1_source.csv`.

## Fig. 2 | Confidence-shielded evaluator-conditioned fusion

**a**, The visible–infrared pair is passed to visible, CDF, PIAFusion, and
SPGFusion candidates. The target detector computes the mean of its five highest
confidences for every candidate. A fused candidate is selected only if its score
is at least 1.5 times the visible score; otherwise the visible input is retained.
No ground-truth labels are used for arbitration. **b**, Composition of the frozen
input decisions across all checkpoint–image trials for LLVIP (8 × 300), M3FD
(2 × 200), and DroneVehicle official test (2 × 300). Source data are provided in
`figure2_source.csv`.

## Fig. 3 | Risk–return trade-off of fusion-input policies

**a**, Dataset-balanced mean AP50 change versus worst-checkpoint AP50 change for
five policies. The CS-ECF horizontal error bar is the training-run-by-group hierarchical
95% interval `[+0.0026,+0.0261]`. **b**, Paired checkpoint changes under the
unshielded selector and CS-ECF, sorted by the unshielded result. Green segments
indicate that the shield improves the checkpoint contrast and red segments
indicate a reduction. The shield sacrifices part of the high mean gain on
DroneVehicle to reduce the worst LLVIP loss from -0.0584 to -0.0040 AP50. Source
data are provided in `figure3_source.csv`.

## Fig. 4 | Thermal constraints and class-specific effects

**a**, M3FD AP50 changes for six classes under YOLO26n and YOLOv8n. The gain is
concentrated in the people class; the other five classes change by at most 0.0003
AP50. **b**, Fixed-input DroneVehicle CDF gain with the thermal-validity mask or
global detail injection. The mask improves the mean gain by 0.0007 AP50. **c**,
Spearman association between the image-level fused-selection fraction and thermal
top-hat contrast, visible luminance, or CDF-mask activity. The association
reverses on DroneVehicle and does not establish a universal thermal switching
rule. Source data are provided in `figure4_source.csv`.

