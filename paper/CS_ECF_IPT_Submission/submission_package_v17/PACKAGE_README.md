# CS-ECF IPT Submission Package v17

Prepared: 2026-10-28

Target: *Infrared Physics & Technology*

Article type: Research Paper

## Readiness

Status: **11-figure redesign complete; technical QA complete; Zenodo v2.4.0
draft reserved; live portal check pending.**

The author confirmed that the administrative facts, declarations, and
originality wording remain unchanged. The package contains the complete
manuscript, supplement, cover letter, Highlights, keywords, 11 main figures,
11 figure-level source-data files, declarations, evidence, and reproduction
scripts.

The version-specific DOI `10.5281/zenodo.23726338` is under the software concept
DOI `10.5281/zenodo.22984388`.

## Version v17 changes

Version v17 expands the main display set from four to 11 figures. The new
sequence makes the full scientific argument visible in the main article:
frozen inference, validation selection, evaluator dependence, multiplicity,
direct CDF comparison, label-free controls, external-evaluator portability,
calibration failure, risk control, thermal/class boundaries, and deployment
cost. It also renumbers source-data files to match the final figure order.

No experiment, threshold, or headline estimate was changed.

## Files

- `Manuscript.pdf`, `main.tex`, `references.bib`
- `Supplementary_Material.pdf`, `supplementary.tex`
- `Cover_Letter.pdf`, `cover_letter.tex`
- `Highlights.txt`, `Keywords.txt`
- `Submission_Checklist.md`
- `IPT_FIGURE_CONTRACT.md`, `IPT_FIGURE_CAPTIONS.md`, `IPT_FIGURE_QA.md`
- `External_Evaluator_Checks.md`
- `FINAL_TECHNICAL_QA.md`
- `REFERENCE_VERIFICATION.md`
- `DECLARATIONS_SOURCE.md`
- `RELATED_WORK_EVIDENCE.md`
- `RELATED_WORK_ADDITIONS.ris`, `.json`, `.tsv`, and `.html`
- `MAJOR_REVISION_STATISTICS.md`
- `figures/` with `Figure1_workflow.pdf` through
  `Figure11_cost_deployment.pdf`
- `source_data/` with `Figure1_source.csv` through
  `Figure11_source.csv`
- `evidence/` with policy, external-evaluator, multiplicity, baseline,
  calibration, class, registration, and reference-audit outputs
- `reproduction/` with the figure generators, external-evaluator workflows,
  statistical analyses, and submission preflight
- `SHA256SUMS.txt`

## Verified document state

- Main manuscript: 25 A4 pages, 240-word abstract, 11 figures.
- Supplementary material: 6 A4 pages.
- Cover letter: 1 A4 page, 491-word body.
- References: 46 verified entries, including 45 DOI-backed records.
- All 45 DOI records match Crossref title, author count, year, volume, and first
  page after normalization.
- The rendered manuscript contains every figure caption from `Figure 1:`
  through `Figure 11:`, every reference number from [1] through [46], one
  Discussion heading, and no unresolved text markers.
- Highlights: 5 lines, each no longer than 85 characters.
- No undefined citations, unresolved cross-references, overfull boxes, or
  author-input placeholders.
- All 11 figure PDFs pass the 5 pt rendered-text floor.
- All 25 manuscript pages, six supplement pages, and the cover letter were
  rendered and visually inspected.

## Portal upload set

Use the live portal labels rather than assuming a fixed upload order.

- Manuscript document: `Manuscript.pdf`; include the LaTeX source only if
  requested.
- Cover letter: `Cover_Letter.pdf`.
- Highlights: paste the five lines from `Highlights.txt` if requested.
- Keywords: enter the six terms from `Keywords.txt`.
- Supplementary material: `Supplementary_Material.pdf`.
- Figures: upload the 11 PDFs from `figures/` if the portal requests separate
  display items. The files are numbered in manuscript order.
- Source data and reproduction evidence: upload or expose separately only when
  requested.

## Evidence boundary

- CS-ECF versus visible: dataset-balanced AP50 gain `+0.0133`, 95% interval
  `[+0.0026,+0.0261]`.
- Checkpoint multiplicity: 9/12 positive, exact sign test `p=0.0730`, no
  checkpoint surviving Holm or BH at 0.05.
- CS-ECF versus Always CDF: `+0.0102`, interval
  `[-0.00057,+0.02980]`; CDF superiority is not established.
- Controls: uniform random `-0.0341`, maximum VIF `+0.0008`, maximum SSIM
  `-0.0946`, maximum entropy `-0.0599`, label-using oracle `+0.0833`.
- Zero-shot YOLOv9t/YOLOv10s gain `+0.0250`/`+0.0213` AP50 over visible, but
  neither separates from Always CDF.
- LLVIP-fine-tuned YOLOv9t is a negative portability boundary: CS-ECF is
  `-0.0108` AP50 relative to visible, while the shield improves the unshielded
  selector by `+0.0153`.

## Upload checks

1. Check the live IPT portal for article-type labels, mandatory fields,
   reviewer-suggestion requests, display-item order, and optional graphical
   abstract fields.
2. Confirm that all uploaded content comes from this v17 directory and that the
   portal-generated preview matches `Manuscript.pdf`.
3. The live portal remains the only non-technical gate.
