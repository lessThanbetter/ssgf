# IPT Figure Contract

Date: 2026-10-29  
Target: *Infrared Physics & Technology*  
Backend: Python and matplotlib  
Width: approximately 162 mm in the A4 manuscript text block
Typography: vector PDF text at a rendered minimum of 5 pt  
Source rule: every quantitative panel traces to a CSV in `source_data/`

## Final evidence chain

| Figure | Purpose | Required boundary |
|---|---|---|
| 1 | Define candidate generation and confidence-shielded arbitration. | Inference uses no ground-truth boxes; development labels selected the margin. |
| 2 | Present direct candidate-by-checkpoint dependence, held-out comparisons, and observed tail loss. | Panel a is retrospective; panels b and c use standard matching and conditional resampling summaries. |
| 3 | Test portability to three external evaluator checkpoints. | Preserve both visible-relative and no-shield reversals; avoid a universal-portability claim. |
| 4 | Show one real successful and one real failed switch. | Display matched true positives selected by a fixed rule; they are not average effects. |
| 5 | Report cost and deployment gates. | Timing is a component estimate and the gates are not measured guarantees. |
| S1 | Display controls and heuristic comparisons. | No confidence-specific mechanism is claimed. |
| S2 | Display all checkpoint- and group-level retrospective contrasts. | Keep the retrospective label and dependent-checkpoint structure. |
| S3 | Show calibration and switched-image diagnostics. | Associations are not causal. |
| S4 | Show class concentration and thermal/domain boundaries. | Thermal relationships are descriptive and domain dependent. |
| S5 | Show evaluator-dependent utility across all fusion candidates. | Development evidence only. |
| S6 | Show the observed mean-tail trade-off. | Empirical summary, not a future-risk guarantee. |
| S7 | Show the complete retrospective CS-ECF versus Always CDF comparison. | Held-out and external-domain intervals are reported in the tables. |
| S8 | Show development-margin selection and post-freeze sensitivity. | The later held-out audit did not reselect the policy after seeing test outcomes. |

## Final numbering

Main figures use `Figure1_workflow.pdf`, `Figure3_core_evidence.pdf`,
`Figure7_external_evaluators.pdf`, `Figure6_case_studies.pdf`, and
`Figure11_cost_deployment.pdf` in that order.

Supplementary figures use `FigureS1` through `FigureS8` in manuscript order.
The generator mapping is stored in `reproduction/FIGURE_BUILD_MAP.json`.

## Global statistical boundary

- The AP50 difference from the unconstrained no-shield selector is
  `+0.00894 [+0.00096,+0.02155]` on held-out LLVIP under standard matching and
  `+0.0410 [+0.0097,+0.0782]` on VEDAI.
- The switch-budget-matched random median is lower by
  `0.01392 [0.00964,0.02023]`.
- CS-ECF does not outperform visible input or Always CDF in the primary
  comparisons.
- The LLVIP-trained YOLOv9t change from visible is `-0.0142` and remains a
  negative portability boundary.
- AP50--95 does not reproduce the no-shield benefit.
- The recent YOLOv11-RGBT baseline is a single-seed implementation pilot, not a
  performance ranking.

## Export and QA

- Main and supplementary PDFs are retained as vector figures.
- All figures pass a rendered 5 pt text-floor audit and are inspected at final
  physical size.
- Figure 1 uses a nominal 182 mm source canvas with at least 6 pt text; LaTeX
  scales it to the 162 mm text block while retaining an effective 5 pt floor.
- Panel labels, zero/reference lines, intervals, legends, and image labels are
  checked for clipping and overlap.
