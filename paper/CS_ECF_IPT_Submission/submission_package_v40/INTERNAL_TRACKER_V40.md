# Internal revision tracker v40

Status: author review, not reviewer-facing

Decision type: **Moderate Revision** (Editorial Manager)

Decision-label note: the email prose says ``minor revision and modification,''
but the managed system displays ``Moderate Revision.'' The portal label is
used for planning, and the revisions are not downgraded.

Decision date: 2026-11-25

Revision deadline: 2026-12-25

Required response mode: point-by-point response in the Editorial Manager
peer-review platform before submission.

## Reviewer 1

| ID | Concern | Action | Work status | Remaining input |
|---|---|---|---|---|
| R1.1 | Complete DroneVehicle table and report no-shield result | ACCEPT_ANALYSIS | Complete. Both independent runs were rescored with confidence-ordered OBB matching. Main and supplementary tables now include visible, CDF, no-shield, CS-ECF, and paired contrasts; native matching is retained as sensitivity. | Author confirmation and final numerical consistency check. |
| R1.2 | Explain margin rule, competing margins, audit selection, and no-detection behavior | ACCEPT_TEXT / CLARIFY_EXISTING | Draft complete. Main text states the loss-constrained rule, development alternatives, audit choice, direct no-shield-minus-visible interval, and the 1/3,400 sentinel frequency. | Author confirmation of the no-detection statement before submission. |
| R1.3 | Add single-candidate gates and CDF sensitivity | ACCEPT_ANALYSIS | Complete. All eight candidate subsets at frozen K and margin are reported; visible+PIA and visible+SPG are the two-candidate gates. One-at-a-time CDF sensitivity now covers qTH, alpha, sigma, and minimum-detail magnitude on M3FD validation and test. | Final author confirmation; interaction terms are not covered. |
| R1.4 | Report M3FD paired differences and run uncertainty | ACCEPT_TEXT / ACCEPT_EXPERIMENT | Complete. Equal-group M3FD differences and intervals remain in Results. Seeds 1 and 2 were trained for both model families under the original recipe and jointly rescored with seed 0. CS-ECF minus visible was positive in 6/6 runs; pooled and equal-group means, ranges, and the fixed-frame limitation are reported in the main text and the supplementary seed-replication table. | Final author confirmation. |

## Reviewer 2

| ID | Concern | Action | Work status | Remaining input |
|---|---|---|---|---|
| R2.1 | Novelty and heuristic framing | ACCEPT_ANALYSIS / ACCEPT_EXPERIMENT | Complete. Added a budgeted candidate-library procedure with explicit feasibility and cost selection. The selected visible-plus-PIA policy was then evaluated under a new locally recorded temporal protocol on 900 LLVIP frames: AP50 `+0.01560 [+0.00249,+0.02874]`, AP50--95 `+0.00946 [+0.00113,+0.01769]`, positive in 3/3 sequences and 6/6 checkpoints. Leave-one-sequence-out sensitivity remained positive in 6/6 checkpoints for every omission. | Author confirmation of the bounded temporal-validation wording. |
| R2.2 | Overbroad critique of prior fixed-detector studies | ACCEPT_TEXT | Draft complete. Introduction now states that fixed-deployment reporting is valid and that this study addresses evaluator or domain change. | None identified. |
| R2.3 | Weak practical value and four-pass latency | ACCEPT_ANALYSIS / ACCEPT_EXPERIMENT | Complete. A six-checkpoint single-GPU benchmark measured complete pipelines: visible-only 15.6 ms, visible+PIA 1672.6 ms, and full library 2515.5 ms equal-checkpoint medians. The selected pipeline is 33.5% faster but remains unsuitable for a low-latency claim. Target-domain failures still return visible input. | Author confirmation of the deployment wording. |
| R2.4 | Abstract emphasis, objectives, limitations, and supplemental findings | ACCEPT_TEXT | Draft complete. Abstract includes the new budget result and transfer failure, Introduction lists three questions, Limitations is consolidated, and important robustness results are in main text. | Final language edit and author confirmation. |

## Current readiness

`needs_author_input`

The v40 manuscript has completed the requested scientific experiments.
Every reviewer item is answered with an inspectable analysis or manuscript
change. The remaining inputs are author approval, platform-specific response
entry, editorial-manager upload checks, and final DOI or archive deposition.
The manuscript remains a bounded post-review revision rather than an externally
registered trial.
