# Final Technical QA

Date: 2026-11-26

Package candidate: v40

Editorial decision: Moderate Revision (Editorial Manager)

Revision deadline: 2026-12-25

## Scientific scope

The primary claim is bounded to evaluator-dependent fusion utility and the
observed AP50 advantage of the confidence shield over unconstrained selection.
The held-out LLVIP result under standard confidence-ordered matching is
`+0.00894 [+0.00096,+0.02155]` relative to no shield, with an observed
worst-checkpoint gain of `-0.00451`. The switch-budget-matched random median
is lower by `0.01392 [0.00964,0.02023]`. VEDAI gives
`+0.0410 [+0.0097,+0.0782]` relative to no shield.

CS-ECF does not outperform visible input or Always CDF in the primary
comparisons. Two zero-shot external evaluators reverse the no-shield ordering,
and the LLVIP-trained evaluator changes by `-0.0142` relative to visible. The
matched image-statistic audit does not establish a confidence-specific
mechanism. AP50--95 does not reproduce the no-shield benefit.

The recent YOLOv11-RGBT early-fusion baseline was trained on the same M3FD
split under a fixed 40-epoch budget. It obtained AP50 `0.0963` and AP50--95
`0.0467` under the study protocol. Its fusion representation and detector are
jointly trained, so it is reported as a system-level reference and not as an
architecture-matched estimate of the CS-ECF effect.

The post-review candidate-library analysis fixes `K=5` and margin 1.5, selects
visible plus PIAFusion from the original development evaluator cells, and
reports the selected subset without reselection. A later 900-frame temporal
LLVIP holdout was recorded before candidate or detector scoring. The policy
gained `+0.01560 [+0.00249,+0.02874]` AP50 and
`+0.00946 [+0.00113,+0.01769]` AP50--95, positive in 3/3 sequences and 6/6
checkpoints. It does not improve on visible in VEDAI or the independent
DroneVehicle runs. This boundary is stated throughout the manuscript.

A six-checkpoint complete-pipeline benchmark measured equal-checkpoint medians
of `15.6 ms` for visible-only inference, `1672.6 ms` for visible plus
PIAFusion, and `2515.5 ms` for the full library. The selected pipeline is
33.5% faster but is not presented as a low-latency method.

Two additional seeds were trained for each M3FD model family under the original
40-epoch recipe. All six checkpoints were jointly rescored on the fixed test
split. CS-ECF minus visible was positive in 6/6 runs. The pooled mean was
`+0.00852` (range `+0.00179` to `+0.01809`) and the equal-group mean was
`+0.00617` (range `+0.00175` to `+0.01278`). Run-level values are reported in
the supplement and in `TableM3FDMultiseed_source.csv`.

## Build and document checks

- Main manuscript: 21 A4 pages and an abstract within the 250-word limit.
- Supplementary material: 23 A4 pages.
- Cover letter: 1 A4 page.
- Revision cover letter draft: 1 A4 page.
- Marked manuscript: 21 A4 pages, generated from v39 and v40 flattened
  sources with latexdiff.
- Reviewer response drafts: 1 A4 page each.
- Main figures: 6.
- Supplementary figures: 8.
- Highlights: five lines, each at most 85 characters.
- The five Highlights are supplied only in `Highlights.txt`, not in the manuscript.
- References: 48 entries.
- LaTeX and BibTeX completed without unresolved citations or cross-references.
- No placeholders remain.
- The AI declaration states that AI was used only for writing assistance and
  explicitly excludes data collection, experiments, analysis, model training,
  figure generation, and result interpretation.
- Full-manuscript English polish applied without changing numerical results,
  claim boundaries, citations, or figures.
- Every main figure passes the rendered-text floor and has source data.
- The budgeted-selection figure is generated directly from the packaged JSON
  and CSV outputs of the candidate-subset analysis.
- The CDF sensitivity table is supported by packaged validation and test score
  arrays generated from the candidate images.
- The main and supplement pages were rendered and inspected through the new
  prospective-confirmation and latency tables.
- Figure 1 uses the recorded LLVIP frame 070142 candidate scores and passes
  the Python source preflight and the 5 pt rendered-text audit.

## Evidence checks

- Prospective LLVIP and VEDAI point estimates and intervals match canonical
  evidence.
- All DroneVehicle public best/last checkpoints remain one dependent run
  cluster.
- The independent DroneVehicle runs and VEDAI runs are archived separately.
- The joint baseline protocol, initialization, checkpoint, predictions,
  class-level AP, frame-group AP, and source table are archived.
- All JSON files parse, and all reproduction Python sources compile.
- The packaged M3FD multiseed analyzer reproduces `analysis.json` and
  `runs.csv` byte for byte from the packaged score arrays.
- Prospective source, candidate, score, protocol, and latency artifacts match
  their recorded counts and hashes.
- Leave-one-sequence-out values match the packaged score arrays and source
  table.

## Prior-submission disclosure

The cover letter identifies `INFPHY-D-26-02136` as a substantially revised
predecessor and summarizes the independent validation, matched random control,
external-evaluator tests, joint baseline, operational conditions, and complete
supplementary evidence added in the present manuscript. The manuscript itself
does not reproduce the predecessor's fidelity-ranking claims.

## Archive status

- `SHA256SUMS.txt` was rebuilt from the complete package payload.
- The complete preflight targets 21 main pages, 23 supplementary pages,
  six main figures, eight supplementary figures, and 48 references.
- The v40 ZIP and SHA-256 sidecar are generated from the final manifest.
- A fresh extraction of the v40 ZIP passes
  `reproduction/verify_cs_ecf_submission.py`.

All requested scientific experiments are complete. Remaining work is author
approval, entry of the point-by-point responses in the Editorial Manager
peer-review platform, upload-field verification, and final DOI or archive
administration.
