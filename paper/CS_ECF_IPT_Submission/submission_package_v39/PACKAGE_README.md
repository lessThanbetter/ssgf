# CS-ECF IPT Submission Package v39

Prepared: 2026-11-26

Target: *Infrared Physics & Technology*

Article type: Research Paper

Decision: Moderate Revision (Editorial Manager)

Revision deadline: 2026-12-25

## Status

Version 39 completes the post-review budgeted candidate-library analysis while keeping
the original score depth and 1.5 margin fixed. It selects the smallest
development-feasible subset, evaluates that subset without reselection on the
held-out and external score arrays, and reports the resulting transfer
boundary. The main text now presents six figures. The supplement adds the full
development subset grid, a prospective temporal confirmation, and measured
end-to-end latency.

The supported claim remains deliberately bounded:

1. Fusion utility depends on the evaluator.
2. At AP50, the confidence shield reduces observed losses relative to
   unconstrained switching and a switch-budget-matched random allocation.
3. The shield does not establish a general gain over visible input and does not
   establish superiority over Always CDF.
4. The full-policy AP50 effect does not persist under AP50--95 and does not
   transfer uniformly to external evaluators.
5. A development-selected visible plus PIAFusion policy improves AP50 and
   AP50--95 on a 900-frame temporal LLVIP holdout. Its measured end-to-end
   median is 1.67 s versus 2.52 s for the full library, but it remains below
   visible input on VEDAI and independent DroneVehicle runs.

M3FD seed replication is now complete. Seeds 0, 1, and 2 were jointly rescored
for both model families on the fixed 200-image test split. CS-ECF minus visible
was positive in 6/6 runs, with pooled mean `+0.00852` and equal-group mean
`+0.00617`. The variation across seeds is retained as a limitation rather than
being converted into a population confidence claim.

R1.1--R1.4 and R2.1--R2.4 are complete at the evidence and text level. The
remaining inputs are author approval and the required point-by-point responses
entered in the Editorial Manager peer-review platform before resubmission.
Portal upload fields and final DOI or archive deposition also require
verification. The package remains `needs_author_input`, not
`ready_to_submit`, until those external items are confirmed.

## Submission files

- `Manuscript.pdf` and `main.tex`
- `Supplementary_Material.pdf` and `supplementary.tex`
- `Cover_Letter.pdf` and `cover_letter.tex`
- `revision_cover_letter_draft.pdf` and `revision_cover_letter_draft.tex`
- `Revision_Marked.pdf` and `Revision_Marked.tex`
- `Highlights.txt` and `Keywords.txt`
- `Response_to_Reviewer_1_DRAFT.md`, `Response_to_Reviewer_2_DRAFT.md`, and
  `INTERNAL_TRACKER_V39.md`
- `Response_to_Reviewer_1_DRAFT.pdf` and `Response_to_Reviewer_2_DRAFT.pdf`
- `REVISION_NOTES_V39.md`
- `DECISION_SUMMARY.md`
- `figures/` with six main and eight supplementary vector figures
- `source_data/` with source data for every main and supplementary figure and
  for the joint-baseline, M3FD seed-replication, prospective confirmation, and
  latency tables
- `evidence/` with protocols, standard-matching score arrays, matched-random
  control, complete M3FD split manifest, M3FD seed-replication runs and score
  arrays, the prospective visible-plus-PIA confirmation, measured latency,
  case studies, external-evaluator raw score arrays, and the joint-baseline
  run
- `reproduction/` with analysis, training, figure-generation, and archive
  scripts
- `SHA256SUMS.txt`

## Main evidence

- Held-out LLVIP standard-matching visible contrast:
  `+0.00012 [-0.00484,+0.00670]`.
- Held-out LLVIP shield-minus-no-shield AP50:
  `+0.00894 [+0.00096,+0.02155]`.
- Matched-random deficit relative to CS-ECF:
  `0.01392 [0.00964,0.02023]`.
- Observed LLVIP worst loss for no shield and CS-ECF:
  `0.03270` and `0.00451`.
- VEDAI shield-minus-no-shield AP50:
  `+0.0410 [+0.0097,+0.0782]`.
- Development-selected visible plus PIAFusion:
  mean validation gain `+0.00206`, worst-cell gain `+0.00039`, 4/4 positive.
- Frozen two-candidate LLVIP:
  visible `+0.00337 [-0.00016,+0.01267]` and
  no-shield `+0.01218 [+0.00136,+0.02915]`.
- Prospective visible plus PIAFusion on 900 new LLVIP frames:
  AP50 `+0.01560 [+0.00249,+0.02874]` and
  AP50--95 `+0.00946 [+0.00113,+0.01769]`, positive in 3/3 sequences and
  6/6 checkpoints.
- Ungated PIAFusion on the same holdout:
  AP50 `-0.01237`, with sequence 100 at `-0.15891`.
- Measured complete-pipeline median:
  visible `15.6 ms`, visible plus PIA `1672.6 ms`, and full library
  `2515.5 ms`.
- Frozen two-candidate transfer:
  VEDAI `-0.00329 [-0.00884,+0.00011]` and
  DroneVehicle `-0.00124 [-0.00578,+0.00280]` versus visible under
  confidence-ordered OBB matching.
- LLVIP-trained external evaluator change from visible: `-0.0142`.
- YOLOv11-RGBT M3FD AP50 under the study AP protocol: `0.0963` from one
  single-seed, resize-only implementation pilot.
- M3FD seed replication: CS-ECF minus visible positive in 6/6 runs, pooled
  mean `+0.00852` (range `+0.00179` to `+0.01809`) and equal-group mean
  `+0.00617` (range `+0.00175` to `+0.01278`).
- CS-ECF does not outperform visible input or Always CDF in the primary
  comparisons.
- Figure 1 uses an actual LLVIP frame and its recorded candidate scores to
  illustrate candidate generation, the 1.5 margin gate, and the visible
  fallback.

## Evidence boundary

- The local LLVIP protocol was not externally registered, and no shield was
  not a named primary endpoint.
- DroneVehicle contains dependent public best/last checkpoints and mixed
  independent-run outcomes.
- Matched image-statistic controls do not establish a confidence-specific
  mechanism.
- The YOLOv11-RGBT run changes the fusion representation and detector together
  and is therefore not a tuned performance ranking.
- Full-policy AP50--95 does not reproduce the AP50 tail-control direction;
  the bounded visible-plus-PIA holdout does improve AP50--95.
- The two-candidate subset was selected outside the original prospective
  protocol. Its later 900-frame temporal protocol and sample amendment were
  recorded before scoring but were not externally registered.
- CDF parameter sensitivity was evaluated one parameter at a time on M3FD
  validation and test. Fixed-CDF AP50 changed, but the full policy changed by
  at most 0.00018 and the selected visible-plus-PIA policy was unchanged.
- The M3FD seed archive is conditional on two fixed frame groups and two model
  families. Its jointly rescored seed-0 absolute AP50 values are not mixed with
  the earlier single-run archive.
- No claim is made for a formal risk guarantee, universal evaluator
  portability, CDF dominance, or low-latency deployment.
- The cover letter identifies the preceding manuscript
  `INFPHY-D-26-02136` as a substantially revised predecessor and summarizes
  the added validation and review-driven evidence without presenting the new
  manuscript as an unrelated study.

## Archive

The package is archived as
`CS_ECF_IPT_submission_2026-11-26_v39.zip` with a SHA-256 sidecar in the parent
directory. Final archive verification is recorded in
`FINAL_TECHNICAL_QA.md`.

The reusable concept DOI is `10.5281/zenodo.22984388`. A version-specific DOI
must be created after the final archive is deposited.
