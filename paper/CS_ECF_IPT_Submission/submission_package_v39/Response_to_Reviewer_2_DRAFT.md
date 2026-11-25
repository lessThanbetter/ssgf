# Response to Reviewer 2

Status: author review draft

## Comment 1: Novelty and heuristic framing

We agree that the original submission did not state the methodological
contribution beyond the empirical boundary study clearly enough. We have
retained CS-ECF as the four-candidate reference policy and added a budgeted
candidate-library selection procedure. The procedure keeps \(K=5\) and the
visible margin at 1.5, evaluates every candidate subset containing visible on
the same four development evaluator cells, imposes the existing \(-0.0005\)
per-evaluator loss bound, and uses an explicit minimum-detector-forward,
minimum-cost tie rule. At the frozen score and margin, this selected visible
plus PIAFusion.

After the exploratory 050--070 evaluation, we recorded a new temporal
protocol before scoring 900 frames from LLVIP sequences 080, 090, and 100. A
resource amendment recorded before candidate generation or detector scoring
fixed the sample. The unchanged policy gained \(+0.01560\)
\([+0.00249,+0.02874]\) AP50 and \(+0.00946\)
\([+0.00113,+0.01769]\) AP50--95 over visible, with positive gains in all
three sequences and all six checkpoints. Ungated PIAFusion was not stable: its
equal-sequence AP50 change was \(-0.01237\), including \(-0.15891\) on
sequence 100. The gate therefore added value beyond merely choosing PIAFusion.

We retained the cross-domain failures. The same subset was \(-0.00329\)
relative to visible on VEDAI and \(-0.00124\) on the independent DroneVehicle
runs under confidence-ordered OBB matching. The revision presents a
target-specific candidate-library reduction with a visible fallback, not a
universal fusion operator or cross-detector guarantee. The new confirmation is
bounded to three sequences and six fixed runs and was not externally
registered.

## Comment 2: Characterization of prior work

We agree that the original introduction could be read as a blanket criticism of
studies that evaluate a fused representation with a fixed deployed detector.
We have revised the paragraph to state that this reporting choice is valid for
a fixed deployment and that the present study addresses the different setting
in which the evaluator checkpoint or domain may change. The revision no longer
attributes a cross-detector generality assumption to the prior work.

## Comment 3: Practical value and computation

We agree that the full four-candidate policy is not an acceptable default when
it provides only a narrow AP50-specific benefit and requires four detector
forwards. We therefore measured the complete single-process pipeline rather
than relying on summed component times. With decoded images and loaded models,
six detector checkpoints, and batch size one, the equal-checkpoint median was
\(1.67\) s for visible plus PIAFusion and \(2.52\) s for the full library. The
selected pipeline reduced median latency by \(0.84\) s, or \(33.5\%\).
Visible-only inference had a median of \(15.6\) ms. The manuscript also reports
the limits of this reduction. The subset failed to improve on visible in VEDAI
and the independent DroneVehicle runs, so the operational procedure returns
visible input when target-domain validation fails.

We do not claim that this removes all validation burden. A checkpoint-specific
selection policy cannot be justified without target-checkpoint validation. The
revision states that this requirement is part of the method rather than a
guarantee transferred to the reader. The 102.7 and 706.3 ms values remain
available only as the deterministic selection diagnostic. They are no longer
used as deployment latency evidence. The measured pipelines remain too slow
for a real-time claim.

## Comment 4: Abstract, objectives, limitations, and supplemental results

We have revised the abstract to include the candidate-library result and its
cross-domain failure. The Introduction now specifies three research questions.
We added a consolidated Limitations subsection and moved the most important
negative results into the main text, including the direct no-shield contrast,
the no-detection sentinel audit, the M3FD equal-group estimates, and the
two-candidate transfer failures.

To make these boundaries easier to verify in one place, the revised abstract
states the null visible contrast, negative CDF comparison, AP50-only boundary,
external reversal, and absence of a mechanism. It also adds the lower-cost
subset result and its later temporal confirmation. We retain the full-policy
AP50--95 boundary and report the selected-policy AP50--95 result only for the
bounded 900-frame holdout.
