# IPT Figure Captions

Final v38 ordering, 2026-11-25. Each displayed figure has a source-data CSV in
`source_data/`.

## Main figures

### Fig. 1 | Confidence-shielded, evaluator-conditioned fusion-input selection

Stage 1 constructs visible, CDF, PIAFusion, and SPGFusion candidates from the
registered visible-infrared pair, and the target detector computes each
top-five confidence score without per-image ground truth. Stage 2 accepts the
highest-scoring fused input only when it is visible-distinct and reaches 1.5
times the visible score; otherwise visible is retained. The example uses LLVIP
frame 070142. Panel b reports the observed selection composition. Source data:
`Figure1_source.csv` and `Figure1_example_scores.csv`.

### Fig. 2 | Core evidence for detector-conditioned arbitration

Panel a is a grouped dot plot of fixed CDF, PIAFusion, and SPGFusion contrasts
across retrospective checkpoints. Panel b is a forest plot comparing CS-ECF
with visible, Always CDF, and no shield in held-out LLVIP and external VEDAI
data under standard matching. Panel c uses a dumbbell plot for the observed
worst-checkpoint loss. Intervals are conditional resampling summaries. Source
data:
`Figure3_source.csv`.

### Fig. 3 | Budgeted candidate-library selection and transfer boundary

Panel a shows development mean and worst-cell AP50 gain for every candidate
subset at the frozen K = 5, margin 1.5 policy, positioned by estimated serial
component cost. Panel b compares the development-selected visible plus
PIAFusion subset with visible input on standard-matched LLVIP, VEDAI,
DroneVehicle, and the external evaluators. The subset analysis is exploratory
because it was not part of the original prospective protocol. Source data:
`FigureBudgetedSelection_source.csv`.

### Fig. 4 | External-evaluator portability stress tests

Zero-shot detectors support a visible-relative gain but not Always-CDF
superiority, and their no-shield point estimates exceed CS-ECF. An
LLVIP-trained checkpoint reverses the visible-relative sign.
Intervals are descriptive and conditional on seven scene groups. Source data:
`Figure7_source.csv`.

### Fig. 5 | Successful and failed CS-ECF switches

The two rows show frame 070142, where the fused input recovers the annotated
pedestrian, and frame 070162, where the fused input loses one matched true
positive. Solid boxes are matched true-positive predictions at IoU 0.50;
dashed boxes are ground truth. The examples are fixed-rule illustrative
endpoints, not average effects. Source data:
`Figure6_source.csv`.

### Fig. 6 | Cost and deployment boundaries

Component timing, non-additive peak memory, and four validation gates required
before choosing CS-ECF. A failed gate returns the system to visible input or a
stable fixed CDF. The figure defines neither a validated deployment threshold
nor a latency guarantee. Source data:
`Figure11_source.csv`.

## Supplementary figures

### Fig. S1 | Controls and heuristic comparison

Matched controls, random selection, and a label-using heuristic are descriptive
and do not establish a confidence-specific mechanism. Source data:
`FigureS1_controls_source.csv`.

### Fig. S2 | Checkpoint and group intervals

All retrospective checkpoint and group-resampling contrasts are shown for
transparency. These results remain retrospective. Source data:
`FigureS2_checkpoint_source.csv`.

### Fig. S3 | Calibration and switched-image diagnostics

Detection-level calibration error and switch outcomes are descriptive. The
design does not establish causality. Source data:
`FigureS3_calibration_source.csv`.

### Fig. S4 | Thermal and class boundaries

Class concentration and thermal associations are domain dependent. The M3FD
gain is concentrated in people, and the thermal association changes sign
across domains. Source data: `FigureS4_thermal_class_source.csv`.

### Fig. S5 | Evaluator-dependent utility

All retrospective fusion candidates and checkpoints are shown as development
evidence. Source data: `FigureS5_evaluator_source.csv`.

### Fig. S6 | Observed mean-tail trade-off

No-shield and CS-ECF results are compared as an empirical summary, not a
future-risk guarantee. Source data: `FigureS6_tail_source.csv`.

### Fig. S7 | Complete retrospective CS-ECF versus Always CDF comparison

All retrospective checkpoint contrasts are shown. Held-out LLVIP and VEDAI
contrasts are reported separately. Source data: `FigureS7_cdf_source.csv`.

### Fig. S8 | Development-margin selection and sensitivity

The original development search selected margin 1.5 at K = 5 under the stated
validation criterion. Later descriptive K-by-margin grids show sensitivity to
score depth and margin. The expanded grid did not reselect the policy after
held-out testing. Source data: `FigureS8_selection_source.csv`.
