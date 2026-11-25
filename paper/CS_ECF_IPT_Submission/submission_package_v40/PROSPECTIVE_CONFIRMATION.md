# Held-out Evaluation and Independent-Run Audit

Date: 2026-10-29

## Scope

The original 12-checkpoint result was retrospective. Its LLVIP development
split was image-stem-disjoint but not scene-group-disjoint, and the original
policy search lacked an external timestamp. Version 24 reports a separately
recorded held-out evaluation of the already selected CS-ECF policy.

The protocol files and SHA-256 digests are stored under
`evidence/prospective/protocol/`. They were not externally registered. The
local protocol creation preceded the confirmation outputs, but it does not
retroactively establish a timestamp for the original policy search.

## LLVIP

| Analysis | Mean AP50 | Conditional interval | Positive checkpoints |
|---|---:|---:|---:|
| Standard matching, six final, CS-ECF minus visible | +0.00012 | [-0.00484,+0.00670] | 4/6 |
| Standard matching, six final, Always CDF minus visible | +0.00458 | [-0.00178,+0.01230] | 5/6 |
| Standard matching, six final, CS-ECF minus Always CDF | -0.00446 | [-0.01109,+0.00172] | 1/6 |
| Standard matching, six final, CS-ECF minus no shield | +0.00894 | [+0.00096,+0.02155] | 6/6 |
| Maximum-cardinality, six final, CS-ECF minus visible | -0.00065 | [-0.00550,+0.00528] | 3/6 |
| Maximum-cardinality, six final, CS-ECF minus no shield | +0.00976 | [+0.00113,+0.02252] | 6/6 |
| Eight archived, CS-ECF minus Always CDF | -0.00220 | [-0.00852,+0.00497] | 3/8 |
| Eight archived, CS-ECF minus no shield | +0.00467 | [-0.00915,+0.01665] | 6/8 |

The original `+0.0092` magnitude was not reproduced. Under standard matching,
the confidence shield separates from unconstrained selection in AP50 and
reduces the observed worst-checkpoint gain from `-0.03270` to `-0.00451`.
The protocol named visible as its primary endpoint, so the no-shield
comparison is secondary and exploratory.

A random allocation matched to each checkpoint's exact switch count and
candidate multiset is worse than CS-ECF. The CS-ECF mean is `+0.00012`, the
matched-random median is `-0.01380`, and CS-ECF minus random is
`+0.01392 [+0.00964,+0.02023]`.

The audit sequence 040 would have selected margin 2.0 under the original
loss-constrained rule. Margin 1.5 was not changed after this audit. It gave
`-0.00043` on sequence 040.

## Matched image-statistic controls

VIF selected visible at every margin from 1.0 to 2.0. SSIM switched only at
margin 1.0 and had a balanced gain of `-0.0360`. Entropy had a balanced gain of
`-0.0940` at margin 1.5 and selected no fused image at margin 2.0. No
non-degenerate image-statistic operating point matched the CS-ECF margin and
switch budget. A confidence-specific mechanism is therefore not established.

## DroneVehicle

The native OBB evaluation uses probabilistic rotated-box IoU and one-to-one
maximum-cardinality matching. Public best/last share one upstream run. The two
new YOLO11n-OBB runs are independent.

| Checkpoint | Mean AP50 | Block interval | Positive blocks |
|---|---:|---:|---:|
| Public best | +0.00718 | [-0.00055,+0.01772] | 2/3 |
| Public last | +0.00665 | [-0.00330,+0.01666] | 2/3 |
| New run 1 | -0.00051 | [-0.00564,+0.00517] | 1/3 |
| New run 2 | +0.00169 | [+0.00090,+0.00252] | 3/3 |
| New-run equal mean | +0.00059 | n/a | 1/2 runs |
| Four-label mean | +0.00375 | [-0.00038,+0.00893] | 3/4 labels |

The four-label estimate overweights the public lineage and is not presented as
an independent-run confirmation.

## VEDAI external-domain test

The official VEDAI 512 dataset was used with fold01 as the test set,
fold02 as validation, and folds03--10 as training. Two visible-only
YOLO11n-OBB runs were fixed at the final 30-epoch checkpoint before test
scoring.

| Contrast | Equal-run mean | Conditional interval | Positive runs |
|---|---:|---:|---:|
| CS-ECF minus visible | -0.0136 | [-0.0235,-0.0058] | 0/2 |
| Always CDF minus visible | +0.0106 | [-0.0012,+0.0357] | 2/2 |
| No shield minus visible | -0.0546 | [-0.0926,-0.0222] | 0/2 |
| CS-ECF minus Always CDF | -0.0242 | [-0.0496,-0.0087] | 0/2 |
| CS-ECF minus no shield | +0.0410 | [+0.0097,+0.0781] | 2/2 |

VEDAI therefore supports the shield's role in limiting unconstrained
switching, but not a visible-relative gain. Both outcomes are retained.
The equal-block estimand averages the classes present in each block because AP
is undefined for a class absent from a block. It is not a fixed five-class
inference.

## Canonical source

`evidence/prospective_standard_matching/analysis.json` and
`matched_switch_target_control.json` are the machine-readable sources for the
held-out AP50 values and matched control. The original
`evidence/prospective/canonical_prospective_evidence.json` is retained as the
maximum-cardinality sensitivity record.

VEDAI standard-matching values are stored in
`evidence/vedai/standard_matching/analysis.json`.
