# Infrared Physics & Technology Submission Checklist

Date: 2026-10-28

Target: *Infrared Physics & Technology* (Elsevier)

## Submission readiness

**Author-confirmed; 11-figure redesign and language polish complete; Zenodo
v2.4.1 draft reserved; live portal check pending.**

The author confirmed that the inherited administrative facts and declaration
text remain unchanged. The manuscript, supplement, cover letter, Highlights,
keywords, 11 main figures, 11 figure-level source-data files, declarations,
statistical diagnostics, and reproducibility evidence are assembled.

The archival code revision is GitHub tag `v2.4.1`; the reserved Zenodo version
DOI is `10.5281/zenodo.23728385` under concept DOI
`10.5281/zenodo.22984388`.

## Technical gate

| Gate | Status | Evidence |
|---|---|---|
| LaTeX build | complete | `Manuscript.pdf`, `Supplementary_Material.pdf`, `Cover_Letter.pdf` |
| Main figure count | complete | 11 final figures, matching the target reference-paper display-item count |
| Figure rendering | complete | SVG, PDF, 600 dpi TIFF, and PNG for all 11 figures |
| Figure source data | complete | `Figure1_source.csv` through `Figure11_source.csv` |
| Figure typography | complete | all PDF text runs at least 5 pt; 0 below-minimum runs |
| Numerical/reference audit | complete | `FINAL_TECHNICAL_QA.md` |
| PDF visual audit | complete | 25 manuscript pages rendered and inspected |
| Related Work and citations | complete | 46 references, including 45 DOI-backed records |
| Multiplicity and sensitivity audit | complete | `MAJOR_REVISION_STATISTICS.md` |
| GitHub/Zenodo archive | draft reserved | version `v2.4.1`; DOI `10.5281/zenodo.23728385` |
| Author administrative facts | complete | author confirmed unchanged on 2026-10-28 |
| Live IPT portal prompts | pending | CAPTCHA/login blocks unattended check |

## Deliverable matrix

| Item | Status | Draft/output |
|---|---|---|
| Main manuscript | ready | `Manuscript.pdf`, `main.tex` |
| Title page | ready | compiled in `Manuscript.pdf` |
| Cover letter | ready | `Cover_Letter.pdf`, `cover_letter.tex` |
| Highlights | ready | `Highlights.txt` |
| Keywords | ready | `Keywords.txt` |
| Main figures | ready | `figures/Figure1_*.pdf` through `Figure11_*.pdf` |
| Figure captions and QA | ready | `IPT_FIGURE_CAPTIONS.md`, `IPT_FIGURE_QA.md` |
| Tables | ready | embedded in `Manuscript.pdf` |
| Supplementary information | ready | `Supplementary_Material.pdf`, `supplementary.tex` |
| External evaluator checks | ready | `External_Evaluator_Checks.md`, evidence JSON |
| Data and code availability | ready | compiled in `main.tex` |
| CRediT, interests, funding, AI use | ready | compiled in `main.tex` |
| Related/preprint disclosure | ready | compiled cover letter |
| Suggested reviewers | portal-dependent | omit unless requested |
| Graphical abstract | portal-dependent | submit only if requested |

## Figure evidence chain

1. Frozen CS-ECF workflow and visible fallback.
2. Pre-freeze margin selection and K/margin sensitivity.
3. Evaluator-dependent fusion utility.
4. Checkpoint contrasts with Holm/BH correction and sign test.
5. Direct CS-ECF versus Always CDF comparison.
6. Random/image-quality controls and oracle headroom.
7. External-evaluator portability, including the fine-tuned negative result.
8. Calibration and switched-image failure boundary.
9. Risk-return trade-off and shield effect.
10. M3FD class concentration and domain-dependent thermal evidence.
11. Latency, memory, and deployment decision boundaries.

## Canonical values

| Metric | Canonical value |
|---|---:|
| Confirmatory checkpoints | 12 |
| Positive checkpoints | 9 |
| Exact sign-test p value | 0.0730 |
| Checkpoints surviving Holm/BH at 0.05 | 0 |
| Dataset-balanced AP50 gain | +0.0133 |
| Training-run-by-group hierarchical CI | [+0.0026,+0.0261] |
| CS-ECF minus Always CDF, balanced | +0.0102 |
| Direct-comparison hierarchical CI | [-0.00057,+0.02980] |
| No-shield / CS-ECF worst AP50 loss | -0.0584 / -0.0040 |
| Zero-shot gain over visible | +0.0213 to +0.0250 AP50 |
| Fine-tuned YOLOv9t minus visible | -0.0108 AP50 |
| Fine-tuned shield gain over no-shield | +0.0153 AP50 |
| Uniform-random balanced gain | -0.0341 AP50 |
| Per-image oracle balanced upper bound | +0.0833 AP50 |
| Selected validation policy | K=5, margin=1.5 |
| Serial latency estimate | 706.3 ms |

## Author confirmation record

On 2026-10-28, the author confirmed retention of the original administrative
and declaration content without changes:

- author order, affiliations, emails, ORCIDs, CRediT roles, no-specific-grant
  funding, and no-competing-interest statements;
- the named OpenAI Codex AI tool and final AI declaration wording;
- the cover-letter “not previously published / not under consideration”
  assertion and related-manuscript disclosure;
- Research Paper as the intended article type.

Add reviewer suggestions only if the IPT portal requests them.

## Next actions

1. Open the live IPT submission portal with an interactive browser.
2. Confirm article type, portal labels, abstract/display limits, figure-upload
   order, and optional graphical-abstract fields.
3. Upload files from `submission_package_v18/`.
4. Compare the portal preview with `Manuscript.pdf` before final submission.
5. If the portal requires a scientific or administrative change, rebuild the
   PDFs and regenerate the SHA-256 manifest before upload.
