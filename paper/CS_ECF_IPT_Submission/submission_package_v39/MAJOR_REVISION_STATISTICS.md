# Retrospective Equal-Group Statistical Audit

Date: 2026-10-29

This note describes the original 12-checkpoint retrospective analysis. It is
not the prospective confirmation. See `PROSPECTIVE_CONFIRMATION.md` for the
scene-disjoint LLVIP and independent DroneVehicle results.

## Analysis hierarchy

- Detector checkpoint: unit for architecture-level summaries.
- Repeated-measure units: seven LLVIP scenes, two M3FD held-out frame groups,
  and three prespecified DroneVehicle 100-frame blocks.
- DroneVehicle blocks: identifiers 7, 32, and 71.
- The two DroneVehicle checkpoints remain one training-run cluster.
- Resampling draws: 2,000 per-checkpoint group draws and 5,000
  run-cluster-by-group aggregate draws.

## Descriptive checkpoint result

Nine of 12 CS-ECF-minus-visible point estimates were positive. Five
group-resampling intervals excluded zero, including both DroneVehicle
checkpoints. The two DroneVehicle checkpoints are not independent runs, and
all checkpoint contrasts share test groups. The manuscript therefore does not
apply an exact sign test or use checkpoint-level p values as confirmatory
inference.

The retrospective task-balanced point estimate is +0.0092 with a conditional 95%
run-cluster-by-group interval of [+0.0026,+0.0166]. The point, interval, and
20-run stability audit use the same estimator. Because only one DroneVehicle
training run is available, the interval is conditional on the observed
checkpoint set and does not estimate cross-run coverage.

## DroneVehicle grouping audit

The 300 test pairs were sampled as three fixed 100-frame blocks. The selected
stems contain source prefixes 007/008, 032/033, and 071/072 across the three
block boundaries. The prefixes are nested filename metadata and are not
additional evaluation groups. The exact block manifest is
`evidence/major_revision/dronevehicle_100frame_groups.json`.

## Baseline and headroom checks

| Policy | Balanced AP50 change | Worst checkpoint change | Positive checkpoints |
|---|---:|---:|---:|
| CS-ECF | +0.0092 | -0.0051 | 9/12 |
| Always CDF | +0.0021 | -0.0076 | 8/12 |
| Top-5 no shield | +0.0315 | -0.0745 | 7/12 |
| Uniform random candidate | -0.0341 | -0.1341 | 2/12 |
| Maximum reference VIF among fused candidates | +0.0008 | -0.0059 | 8/12 |
| Maximum SSIM among fused candidates | -0.0946 | -0.2855 | 3/12 |
| Maximum entropy among all candidates | -0.0599 | -0.2432 | 2/12 |
| Label-using per-image heuristic | +0.0833 | +0.0205 | 12/12 |

CS-ECF, Always CDF, and no-shield rows are equal-group estimates. The random,
image-quality, and label-using rows retain pooled-checkpoint calculations
because matched threshold tuning was unavailable. They are descriptive and
are not used as matched tests. The heuristic is not a formal pooled-AP50 upper
bound. Image-quality selectors use 256 x 256 grayscale inputs and no detector
confidence.

## Matched candidate-library check

The same top-five score and 1.5 margin were applied to visible and CDF only.
The task-balanced equal-group gain over visible was +0.00049, with conditional
interval [+0.00003,+0.0011]. The full four-candidate policy exceeded this matched
two-candidate gate by +0.0087 AP50, with interval [+0.0024,+0.0160]. This
supports a contribution from the larger candidate library in the observed
sample, especially on DroneVehicle, but remains conditional on one
DroneVehicle training run.

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
| YOLOv9t LLVIP-trained | 76.7%/23.3% | 0.0475 | 0.0459 | 25.7% | 41.4% | 2/5 |

ECE is a descriptive detection-level calibration proxy using ten fixed
confidence bins. The LLVIP-trained model switches at a similar rate to the
zero-shot models but has a much larger calibration error and degrades on a much
larger fraction of switched images. This supports calibration as a plausible
boundary mechanism without proving causality.

The external contrasts are reported as equal-scene exploratory summaries. With
seven scene groups, no multiplicity-adjusted p-value is used in the manuscript.

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

## Remaining limitations

- The two public DroneVehicle checkpoints come from one training run. The
  supplement separately reports two new independent runs; their equal-run mean
  is heterogeneous and is not pooled with the public lineage as independent
  replication.
- M3FD contains only two held-out frame groups, and the class gain is
  concentrated in the people class.
- LLVIP evaluates 300 of 3,463 eligible official test pairs. The result is
  conditional on that fixed subset rather than an evaluation of all eligible
  pairs.

