# Major-Revision Statistical Audit

Date: 2026-10-28

## Analysis hierarchy

- Independent model unit: detector checkpoint for cross-checkpoint summaries.
- Repeated-measure unit within a checkpoint: scene or consecutive frame group.
- Primary family: the 12 prespecified CS-ECF-minus-visible checkpoint
  contrasts.
- Multiplicity method: Holm family-wise correction, with Benjamini-Hochberg
  q values also reported.
- Bootstrap draws: 2,000 per-checkpoint group resamples and 10,000
  run-cluster-by-group aggregate resamples.

## Primary multiplicity result

The sign count was 9/12 (exact two-sided sign-test p = 0.0730). No individual
checkpoint contrast survived Holm correction at alpha = 0.05. The smallest raw
p-values were 0.0050 for LLVIP/YOLO11n (Holm-adjusted p = 0.0600) and 0.0310
for LLVIP/agnostic (Holm-adjusted p = 0.3408). Consequently, the manuscript
does not claim that any individual checkpoint improved significantly.

The prespecified dataset-balanced estimand remains the aggregate result:
+0.0133 AP50 with a 10,000-draw training-run-by-group 95% interval of
[+0.0026,+0.0261]. This is a small average effect, not evidence that each
checkpoint improves.

## Baseline and headroom checks

| Policy | Dataset-balanced AP50 change | Worst checkpoint change | Positive checkpoints |
|---|---:|---:|---:|
| CS-ECF | +0.0133 | -0.0040 | 9/12 |
| Always CDF | +0.0031 | -0.0059 | 10/12 |
| Top-5 no shield | +0.0332 | -0.0584 | 7/12 |
| Uniform random candidate | -0.0341 | -0.1341 | 2/12 |
| Maximum reference VIF among fused candidates | +0.0008 | -0.0059 | 8/12 |
| Maximum SSIM among fused candidates | -0.0946 | -0.2855 | 3/12 |
| Maximum entropy among all candidates | -0.0599 | -0.2432 | 2/12 |
| Per-image AP oracle | +0.0833 | +0.0205 | 12/12 |

The oracle uses test labels and is an upper bound only. The random baseline
samples uniformly from visible/CDF/PIAFusion/SPGFusion. Image-quality selectors
use 256 x 256 grayscale inputs and no detector confidence.

## Margin and K sensitivity

The top-five score and candidate set were fixed before margin selection. Among
the prespecified margin grid for the frozen four-candidate top-five rule,
margin 1.5 was the first setting satisfying the 0.0005 per-evaluator loss bound
and the largest balanced validation gain under that rule.

Validation results for the selected cell:

| Score | Margin | Balanced gain | Worst evaluator gain | Switch fraction |
|---|---:|---:|---:|---:|
| top-5 | 1.2 | -0.0002 | -0.0056 | 11.4% |
| top-5 | 1.5 | +0.0008 | -0.0001 | 4.7% |
| top-5 | 2.0 | +0.0001 | -0.0004 | 0.9% |
| top-5 | 2.5 | +0.0001 | 0.0000 | 0.2% |

Larger K values and alternative margins were evaluated as post-freeze
sensitivity analyses. They are not eligible for reselecting the primary
policy after test results were observed.

## External evaluator diagnostics

| Evaluator | Visible/fused | Visible ECE | CS-ECF ECE | Switched improved | Switched degraded | Positive/negative scenes |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv9t zero-shot | 77.7%/22.3% | 0.0094 | 0.0091 | 46.3% | 20.9% | 5/2 |
| YOLOv10s zero-shot | 81.7%/18.3% | 0.0103 | 0.0116 | 38.2% | 9.1% | 4/3 |
| YOLOv9t LLVIP-fine-tuned | 76.7%/23.3% | 0.0475 | 0.0459 | 25.7% | 41.4% | 2/5 |

ECE is a descriptive detection-level calibration proxy using ten fixed
confidence bins. The fine-tuned model switches at a similar rate to the
zero-shot models but has a much larger calibration error and degrades on a much
larger fraction of switched images. This supports calibration as a plausible
boundary mechanism without proving causality.

For the three-member external-evaluator family, the group-bootstrap
Benjamini-Hochberg q values were 0.0425 (YOLOv9t zero-shot), 0.0425
(YOLOv10s zero-shot), and 0.1192 (YOLOv9t fine-tuned). Holm-adjusted values
were 0.0567, 0.0497, and 0.1192, respectively.

## M3FD class audit

The 200-image M3FD test set contains 2,022 annotations: people 166, car 1,597,
bus 16, motorcycle 189, lamp 20, and truck 34. AP50 is class-balanced, so the
small people count still carries one-sixth of the six-class mean.

For YOLO26n, the people-class gain is +0.0214 with image-bootstrap 95% interval
[-0.0070,+0.0498], and accounts for 87.7% of the positive class-mean gain. For
YOLOv8n, the people-class gain is +0.1090 [0.0440,+0.1913] and accounts for
99.6% of the positive class-mean gain. No other class changes by more than
0.0030. These class intervals are descriptive because the test split contains
only two frame groups.

## Registration sensitivity

ECC translation residuals were available for all 300 LLVIP test pairs
(median 2.53 px). The best-aligned and worst-aligned quartiles contain 75 pairs
each. Across the eight LLVIP checkpoints, mean CS-ECF-minus-visible gains were
+0.0007 and +0.0020, with 5/8 and 6/8 positive contrasts. This does not support
an explanation in which the main effect is confined to well-aligned pairs.
Registration was not audited with the same ECC procedure for M3FD or
DroneVehicle.

## Remaining limitation

The two DroneVehicle checkpoints come from one training run. No independent
DroneVehicle training run is present in the released evidence, so the paper
treats its result as vehicle-focused preliminary evidence and does not claim
independent checkpoint replication.

