# Revision notes for v40

Date: 2026-11-26

Decision: Moderate Revision (Editorial Manager)

Revision deadline: 2026-12-25

## Purpose

Version 38 started the response to the reviewer concerns about novelty and
practical cost. It keeps `K=5` and the visible margin at 1.5, then selects a
candidate library on the original development evaluator cells rather than
performing another margin search.

Version 39 completes the two remaining experiments. It records a later LLVIP
temporal protocol before scoring 900 frames from sequences 080--100 and adds a
six-checkpoint end-to-end latency benchmark.

Version 40 adds a post hoc leave-one-sequence-out sensitivity analysis to the
same fixed temporal holdout. It does not change the primary endpoint or policy.

## Added analysis

- All eight candidate subsets containing visible.
- A fixed development feasibility rule using the existing `-0.0005` AP50
  per-evaluator bound.
- A deterministic minimum-cost tie rule with no test-set selection.
- Selection of visible plus PIAFusion at the frozen margin.
- Frozen evaluation on standard-matched LLVIP, VEDAI, two independent
  DroneVehicle runs, and three external evaluators.
- Development and retrospective candidate-subset grids.
- Historical component-cost comparison retained only as a development
  selection diagnostic; v39 replaces it with measured complete-pipeline
  latency.
- A new temporal holdout on 900 LLVIP frames (three sequences, 300 frames each).
- Complete end-to-end latency measurement for visible-only, visible+PIA, and
  the full four-candidate pipeline on six detector checkpoints.
- Leave-one-sequence-out sensitivity: omitting 080, 090, or 100 leaves
  equal-sequence AP50 gains of `+0.01068`, `+0.01378`, and `+0.02233`,
  respectively, with all six checkpoints positive in every analysis.

## Main results

- Development: `+0.00206` mean AP50 and `+0.00039` worst-cell gain.
- LLVIP versus visible: `+0.00337 [-0.00016,+0.01267]`.
- LLVIP versus no shield: `+0.01218 [+0.00136,+0.02915]`.
- VEDAI versus visible: `-0.00329 [-0.00884,+0.00011]`.
- DroneVehicle versus visible: `-0.00136 [-0.00581,+0.00226]`.
- Prospective visible+PIA AP50: `+0.01560 [+0.00249,+0.02874]`.
- Prospective visible+PIA AP50--95: `+0.00946 [+0.00113,+0.01769]`.
- Prospective ungated PIAFusion AP50: `-0.01237`; sequence 100 was `-0.15891`.
- End-to-end median: visible `15.6 ms`, visible+PIA `1672.6 ms`, and full
  library `2515.5 ms`; the selected pipeline is 33.5% faster.

The original 050--070 evaluation is labelled exploratory because the subset
was not named in the original protocol. The later 080--100 evaluation uses a
new local protocol and a pre-scoring resource amendment. It is a bounded
temporal confirmation, not an externally registered trial. Its cross-domain
failures remain in the main text.

## Manuscript changes

- Added Methods subsection 3.4.
- Added Results subsection 4.3 and Figure 3.
- Added the frozen-transfer figure and source data.
- Added a consolidated Limitations subsection.
- Added the development margin alternatives, the sequence-040 audit values,
  the direct no-shield-minus-visible interval, and the no-detection sentinel
  audit.
- Completed confidence-ordered OBB rescoring for the two independent
  DroneVehicle runs and replaced the native values in the main comparison.
  Native maximum-cardinality results are retained as a supplementary
  sensitivity analysis.
- Added M3FD equal-group CS-ECF-minus-visible estimates and explicitly stated
  that one run per evaluator cannot estimate run variance.
- Added seeds 1 and 2 for both M3FD model families under the original 40-epoch
  recipe and jointly rescored all six checkpoints. CS-ECF minus visible was
  positive in 6/6 runs, with pooled mean `+0.00852` and equal-group mean
  `+0.00617`. Run-level values are in the supplementary M3FD
  seed-replication table.
- Added one-at-a-time CDF sensitivity for qTH, alpha, sigma, and minimum-detail
  magnitude on the M3FD validation and test partitions. Fixed-CDF AP50 varied,
  while the selected visible-plus-PIA policy was unchanged and the full policy
  changed by at most 0.00018 on test.
- Revised the abstract, introduction, Discussion, Conclusions, and Highlights
  to distinguish the full reference library from the budgeted subset.
- Added Supplementary Section 7 with the development grid and frozen transfer
  table.
- Added Supplementary Section 7.1 with the prospective temporal confirmation,
  run-level scores, and sample limitations.
- Added Supplementary Section 7.2 with per-checkpoint component and complete
  pipeline latency.
- Updated the abstract, Figure 3, Discussion, Conclusions, cover letter, and
  Reviewer 2 response to use measured end-to-end cost rather than the earlier
  heterogeneous component sum.
- Added the leave-one-sequence-out table and conservative post hoc wording to
  the supplement, Results, and Reviewer 2 response.

## Verification

- `analysis/cs_ecf/budgeted_candidate_selection.py` reruns the analysis.
- `analysis/cs_ecf/plot_budgeted_candidate_selection.py` rebuilds PDF, PNG, and
  SVG outputs.
- Unit tests cover subset enumeration, decision behavior, and cost reporting.
- Main and supplementary LaTeX sources compile locally with LaTeXmk.
- The M3FD seed replication is reproducible from the packaged score arrays
  with `reproduction/analyze_m3fd_multiseed.py`.
- The prospective confirmation is reproducible from the packaged score arrays
  with `reproduction/analyze_pia_confirmation.py`.
- The latency summary is reproducible from
  `reproduction/summarize_end_to_end.py`.
