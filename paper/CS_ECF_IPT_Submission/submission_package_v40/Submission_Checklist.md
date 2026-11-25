# Infrared Physics & Technology Submission Checklist

Date: 2026-11-26

Package: v40

Target: *Infrared Physics & Technology* (Elsevier)

Decision: Moderate Revision (Editorial Manager)

Revision deadline: 2026-12-25

## Scientific revision

| Requested change | Status | Evidence |
|---|---|---|
| Add a recent joint fusion-detection baseline | complete | YOLOv11-RGBT early-fusion run on the same M3FD split |
| Compress the main text into one evidence chain | complete | abstract, Results, Discussion, Conclusions, Figure 2 |
| Move checkpoint/scene/auxiliary detail to the supplement | complete | eight supplementary figures and detailed tables |
| Replace a weak summary display with real switch cases | complete | Figure 4 and `evidence/case_studies/` |
| State operational guidance | complete | Results, Discussion, Conclusions, and Figure 5 |
| Prior-submission relationship | complete | `Cover_Letter.pdf` identifies `INFPHY-D-26-02136` and summarizes review-driven additions |
| Full-manuscript English polish | complete | `main.tex`, `results_compact.tex`, and `discussion_compact.tex`; no numerical changes |
| Algorithm framework figure | complete | two-stage Figure 1 with real LLVIP 070142 candidate images and recorded scores |
| Framework readability pass | complete | duplicate candidate scores removed; score handoff and decision branches clarified |
| Author-revised Figure 1 vector | complete | `Figure1_workflow.svg` rendered to PDF and integrated into the manuscript |
| AI-use declaration | complete | declaration limited to language, clarity, and readability assistance |
| Figure 5c decision layout | complete | four gates grouped as 2 × 2 checks leading to explicit PASS and fallback outcomes |
| Figure 5c scientific notation | complete | semantic gate icons and Boolean deployment expression |
| Figure 2 evidence redesign | complete | grouped dot plot, forest plot, and worst-loss dumbbell plot replace the heatmap/bar combination |
| Author-revised Figure 5 vector | complete | updated SVG rendered to PDF and integrated at full manuscript width |
| Highlights submission item | complete | five approved lines are retained in the separate `Highlights.txt` file |
| Budgeted candidate-library analysis | complete | development selection, exploratory transfer checks, main figure, supplement tables, and response |
| Prospective visible-plus-PIA confirmation | complete | 900 LLVIP frames from three new sequences; AP50 and AP50--95 positive in 3/3 sequences and 6/6 checkpoints |
| Leave-one-sequence-out robustness | complete | Omitting 080, 090, or 100 retained positive gains in all checkpoint means |
| End-to-end pipeline latency | complete | visible, visible+PIA, and full-library pipelines measured for six detector checkpoints |
| DroneVehicle independent-run table | complete | confidence-ordered OBB rescoring and native sensitivity both packaged |
| CDF hyperparameter sensitivity | complete | one-at-a-time qTH, alpha, sigma, and minimum-detail variants on M3FD validation/test |
| M3FD run-to-run variance | complete | seeds 0, 1, and 2 for both families under the original recipe; 6/6 positive CS-ECF-minus-visible contrasts and run-level source data |

## Documents

| Item | Output |
|---|---|
| Manuscript | `Manuscript.pdf` |
| Supplement | `Supplementary_Material.pdf` |
| Cover letter | `Cover_Letter.pdf` |
| Revision cover letter draft | `revision_cover_letter_draft.pdf` |
| Marked manuscript | `Revision_Marked.pdf` |
| Response to Reviewer 1 | `Response_to_Reviewer_1_DRAFT.pdf` |
| Response to Reviewer 2 | `Response_to_Reviewer_2_DRAFT.pdf` |
| Highlights | `Highlights.txt` |
| Keywords | `Keywords.txt` |
| Main figures | six final PDFs in `figures/` |
| Supplementary figures | eight final PDFs in `figures/` |
| Joint-baseline source data | `source_data/TableJointBaseline_source.csv` |
| M3FD seed-replication source data | `source_data/TableM3FDMultiseed_source.csv` |
| Prospective confirmation source data | `source_data/TablePIAConfirmation_source.csv` |
| Leave-one-sequence-out source data | `source_data/TablePIAConfirmationLOO_source.csv` |
| End-to-end latency source data | `source_data/TableEndToEndLatency_source.csv` |

## Technical gate

- [x] Current LaTeX compilation completed without undefined citations or
  unresolved cross-references.
- [x] Main and supplementary PDFs rendered and visually inspected.
- [x] Main figure captions and numbers match the manuscript order.
- [x] Five Highlights, each at most 85 characters.
- [x] Abstract within the journal limit.
- [x] `reproduction/verify_cs_ecf_submission.py` passes at the current draft
  scope.
- [x] `SHA256SUMS.txt` regenerated after final document build.
- [x] v40 ZIP created and tested after fresh extraction.

## Administrative gate

- [ ] Enter both reviewer responses in the Editorial Manager peer-review
  platform. The decision email states that this is required before submission.
- [ ] Confirm the live IPT portal article type and upload fields.
- [ ] Compare the portal-generated preview with `Manuscript.pdf`.
- [ ] Confirm whether the portal requires a separate prior-review response in addition to the cover-letter disclosure.
- [ ] Deposit the final archive and create a version-specific DOI.

## Claim boundary

The submission supports CS-ECF only as a checkpoint- and domain-specific AP50
safeguard after target validation. The separately selected visible-plus-PIA
policy improved AP50 and AP50--95 on a bounded 900-frame LLVIP holdout, while
ungated PIAFusion reversed on one sequence and external domains still failed.
The selected pipeline is 33.5% faster end to end than the full library but
remains above real time. No claim is made for universal visible-relative gain,
CDF superiority, a confidence-specific mechanism, or low-latency deployment.
