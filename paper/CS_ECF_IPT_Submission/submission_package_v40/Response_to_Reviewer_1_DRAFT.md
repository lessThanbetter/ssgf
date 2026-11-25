# Response to Reviewer 1

Status: author review draft

## Comment 1: DroneVehicle table completeness

We agree that the original table did not report the DroneVehicle no-shield
cell and that the matching protocol needed to be explicit. Both independent
DroneVehicle runs have now been rescored with confidence-ordered one-to-one
rotated-box matching. Table 2 reports visible, Always CDF, no shield, and
full-CS-ECF contrasts for each run and for their equal-run mean. The native
maximum-cardinality results remain in the Supplementary Material as a
sensitivity analysis.

Under standard matching, CS-ECF minus visible was \(-0.00038\)
\([-0.00640,+0.00615]\) in run 1 and \(+0.00207\)
\([+0.00122,+0.00260]\) in run 2. The equal-run mean was \(+0.00085\)
\([-0.00339,+0.00381]\). The opposite signs are reported directly.

## Comment 2: Margin selection and no-detection behavior

We agree that the original text stated the selected margin without giving the
selection rule. The revised Methods states that a setting was admissible only
when every development evaluator gain was at least \(-0.0005\) AP50, and that
the largest balanced gain was selected among admissible settings. At \(K=5\),
margin 1.3 gave a balanced gain of \(+0.0000\) and a worst gain of \(-0.0026\);
margin 1.5 gave \(+0.0008\) and \(-0.0001\); margin 2.0 gave \(+0.0001\) and
\(-0.0004\).

The audit sequence selected margin 2.0 because the frozen 1.5 setting had a
balanced change of \(-0.00043\) and a worst-checkpoint change of \(-0.00171\),
whereas 2.0 gave \(-0.00001\) and \(-0.00022\). We retained 1.5 on the untouched
sequences to avoid a second selection round.

We also agree that the no-detection sentinel needed an explicit audit. Across
the 12 retrospective checkpoints, visible input had no detections in 1 of
3,400 image-checkpoint cases. In the primary 8,370 LLVIP holdout
image-checkpoint cases, the event did not occur. The one retrospective event
switched to PIAFusion and improved per-image AP50 from 0 to 0.505. We therefore
retain the sentinel and report it as a low-prevalence limitation. Adding an
absolute floor would have no estimable effect in the primary holdout because
the sentinel was not observed there.

We also added the direct no-shield-minus-visible contrast,
\(-0.00881\) \([-0.02593,+0.00256]\), and state that part of the no-shield
comparison is recovery from a selector that was below visible.

## Comment 3: Candidate gates and CDF settings

We agree that gate and candidate effects should be separated. The revision now
reports every non-empty candidate subset containing visible at the frozen
\(K=5\) and margin 1.5. It includes two-candidate CDF, PIAFusion, and SPGFusion
gates. The development-feasible minimum subset was visible plus PIAFusion.
Full details are in the supplementary candidate-gate and frozen-transfer
tables.

The CDF parameters are now treated as engineering defaults for one
study-internal candidate. We varied \(q_{\mathrm{TH}}\), \(\alpha\), and
\(\sigma\) one at a time on M3FD validation and test, and also evaluated
minimum-detail magnitude values of 0, 1, 2, and 4. Fixed-CDF AP50 changed by
up to 0.00712 on validation and 0.00686 on test. In contrast, the full-policy
validation gain was identical across settings and its test gain changed by at
most 0.00018. The visible-plus-PIA selected policy was unchanged. This is
reported as sensitivity evidence, not as post hoc selection of a new optimum.

## Comment 4: M3FD paired estimand and run variation

We agree that the pooled absolute AP50 entries were insufficient for the paired
estimand. The revised Results reports equal-group
CS-ECF-minus-visible differences of \(+0.00291\)
\([0.00000,+0.00582]\) for YOLO26n and \(+0.01281\)
\([0.00000,+0.02561]\) for YOLOv8n. Both intervals are conditional on two
frame groups.

Each M3FD evaluator in the original archive represented one training run. To
address run variation directly, we trained seeds 1 and 2 for both families
under the original 40-epoch recipe after the policy had been frozen. We then
jointly rescored all six checkpoints on the same 200-image test groups and
against the same four candidates. CS-ECF minus visible was positive in 6 of 6
runs. The pooled gain averaged \(+0.00852\) with an observed range from
\(+0.00179\) to \(+0.01809\). The equal-group gain averaged \(+0.00617\) with a
range from \(+0.00175\) to \(+0.01278\). Thus the direction is stable across the
six runs, while its magnitude varies substantially.

We do not treat these six runs as six independent model families, and the
interval remains conditional on two held-out frame groups. The run-level
values are reported in the supplementary M3FD seed-replication table and
`TableM3FDMultiseed_source.csv`. Because the seed-0 checkpoints were included
in the joint rescoring archive, its absolute AP50 values differ from the
earlier archive. We therefore compare only within-run contrasts and do not mix
absolute values across the two archives.
