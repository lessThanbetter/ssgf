# Reviewer-driven revision notes for v20

Date: 2026-10-28

## Changes applied

1. Replaced the title-level risk-control claim with a conditional empirical
   mean-tail trade-off.
2. Rewrote the abstract and conclusions to distinguish a small, bounded
   dataset-balanced point estimate from a risk guarantee.
3. Replaced the unqualified “label-free” description with
   “ground-truth-box-free at inference” and disclosed that the margin was
   selected using labeled validation AP50.
4. Corrected the DroneVehicle grouping. Blocks 7, 32, and 71 are the three
   prespecified 100-frame evaluation units. Source prefixes 007/008, 032/033,
   and 071/072 are nested metadata.
5. Rebuilt checkpoint intervals using the three DroneVehicle blocks rather
   than six source-prefix groups.
6. Removed the independent exact sign test and all checkpoint-level p-value
   claims. Checkpoint results are now explicitly descriptive.
7. Narrowed the external-evaluator correction statement to the three
   visible-relative contrasts and marked CDF-relative contrasts descriptive.
8. Added a matched visible-versus-CDF gate using the same top-five score and
   1.5 margin as CS-ECF.
9. Corrected the incomplete Statistical analysis sentence.
10. Added AP50 point conversion text.
11. Renamed Figure 4 and Figure 9 outputs to match their evidence content.
12. Added a clean figure build driver, final-output mapping, local aggregate
    inputs, helper modules, and portable output paths.

## Still requiring new experiments

- Independent DroneVehicle training runs.
- More than two M3FD held-out frame groups.
- Evaluation on the remaining LLVIP test pairs or several prespecified
  stratified samples.
- A matched, validation-tuned image-quality selector with the same candidate
  set, visible fallback, and switching budget.
- Prospective evaluation at operational confidence and false-positive
  constraints.

These are stated as limitations rather than replaced by new claims.

## Reviewer issue disposition

| Reviewer issue | Revision response |
|---|---|
| R1-M1 and R2-M3, cross-run uncertainty | Replaced the nominal cross-run interpretation with a conditional interval and stated the single DroneVehicle run limitation. New runs are still required. |
| R1-M2 and R2-M3, DroneVehicle grouping | Recomputed all affected intervals with blocks 7, 32, and 71 and added a block manifest. |
| R1-M3, R2-M2, and R3-M2, risk definition | Removed formal risk-control language and reframed the result as an observed mean-tail trade-off. |
| R1-M4, image-quality controls | Added a clear mismatch limitation. A matched quality-selector experiment remains outstanding. |
| R2-M1 and R3-M1, novelty and strongest baseline | Reframed the contribution as an empirical study and added a matched visible-or-CDF gate. A current joint fusion-detection system comparison remains outstanding. |
| R2-M4, class and domain scope | Restricted the conclusion to pedestrian- and vehicle-focused evaluation and retained the people-class concentration. |
| R1-M5 and R2-M5, reproduction | Added local figure inputs, portable paths, a build driver, helper modules, and a figure-number map. |
| R1-M6, selection chronology | Kept the limitation explicit; a timestamped prospective protocol is still unavailable. |
| R1-m1, R2-m3, and R3-M3, sign test and incomplete statistics text | Removed the sign-test inference claim and repaired the Statistical analysis text. |
| R2-m1 and R3-M4, label-free wording | Replaced it with ground-truth-box-free at inference and disclosed labeled threshold selection. |
| R3-M6, corrected-contrast labeling | Restricted BH labels to visible-relative contrasts and marked CDF contrasts descriptive. |

## Second re-review corrections

1. Replaced the 0.0133 pooled-image point estimate with an explicit
   equal-group task-balanced estimand. The point, interval, and 20-run
   stability audit now use the same paired sampling structure.
2. Recomputed the equal-group AP50 result as `+0.0092 [+0.0026,+0.0166]`.
   The Always CDF contrast is `+0.0071 [-0.0019,+0.0167]`.
3. Recomputed the matched-gate and AP50--95 sensitivities and synchronized the
   manuscript, supplement, cover letter, figures, source tables, and evidence
   JSON.
4. Replaced the unexplained DroneVehicle interval `[0.0002,0.0406]` with the
   paired shared-block interval `[0.0002,0.0314]` and archived the source
   summary plus a metric-boundary note.
5. Removed the small-cluster Benjamini-Hochberg q-value claims. External
   evaluator contrasts are now equal-scene exploratory summaries without an
   adjusted p-value.
6. Added AP50--95 sensitivity, explicit one-to-one matching language, a
   non-benchmark-equivalence boundary, and a marked OBB metric limitation.
7. Added exact LLVIP train/validation/test manifests and the M3FD held-out
   group manifest. Third-party detector weights and raw datasets remain
   external to the package.
8. Replaced the unvalidated deployment matrix with provisional validation
   questions and relabeled the label-using oracle as a heuristic.

## Final presentation corrections

1. Defined the AP50 loss in Highlights explicitly as percentage points.
2. Rounded single-run latency and memory values to measurement-supported
   precision and retained unrounded source measurements.
3. Added a bounded leave-one-dataset-out statement and clarified why the
   heterogeneous label spaces do not support a pooled class-balanced claim.
4. Disclosed that a broader validation-only candidate search had no timestamped
   chronology relative to the frozen four-candidate policy. The manuscript
   treats this as a limitation rather than claiming prospective selection.
