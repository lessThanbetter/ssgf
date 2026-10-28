# CS-ECF IPT Submission Package v18

Prepared: 2026-10-28

Target: *Infrared Physics & Technology*

Article type: Research Paper

## Readiness

Status: **11-figure design and language polish complete; technical QA complete;
Zenodo v2.4.1 draft reserved; live portal check pending.**

The author confirmed that the administrative facts, declarations, and
originality wording remain unchanged. The package contains the complete
manuscript, supplement, cover letter, Highlights, keywords, 11 main figures,
11 figure-level source-data files, declarations, evidence, and reproduction
scripts.

The version-specific DOI `10.5281/zenodo.23728385` is under the software concept
DOI `10.5281/zenodo.22984388`.

## Version v18 changes

Version v18 retains the 11-figure evidence sequence introduced in v17 and adds a
language-focused revision across the title, abstract, Introduction, Methods,
Results, Discussion, Conclusions, figure captions, supplement, and cover letter.
The revision shortens overloaded sentences, removes repetition, clarifies
dataset-balanced weighting, corrects the DroneVehicle mean-gain wording, and
standardizes evaluator terminology. No experiment, threshold, or headline
estimate was changed.

## Files

- `Manuscript.pdf`, `main.tex`, `references.bib`
- `Supplementary_Material.pdf`, `supplementary.tex`
- `Cover_Letter.pdf`, `cover_letter.tex`
- `Highlights.txt`, `Keywords.txt`
- `Submission_Checklist.md`
- `IPT_FIGURE_CONTRACT.md`, `IPT_FIGURE_CAPTIONS.md`, `IPT_FIGURE_QA.md`
- `LANGUAGE_REVISION_NOTES.md`
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
- `reproduction/` with the figure generators, statistical analyses, and
  submission preflight
- `SHA256SUMS.txt`

## Verified document state

- Main manuscript: 25 A4 pages, 250-word abstract, 11 figures.
- Supplementary material: 6 A4 pages.
- Cover letter: 1 A4 page, 490-word body.
- References: 46 entries, including 45 DOI-backed records.
- All 45 DOI records match Crossref title, author count, year, volume, and first
  page after normalization.
- The rendered manuscript contains every figure caption from `Figure 1:` to
  `Figure 11:`, every reference number from [1] through [46], and one
  Discussion heading.
- No undefined citations, unresolved cross-references, overfull boxes, or
  author-input placeholders.
- All 11 figure PDFs pass the 5 pt rendered-text floor.
- All 25 manuscript pages, six supplement pages, and the cover letter were
  rendered and visually inspected.

## Portal upload set

Use the live portal labels rather than assuming a fixed order.

- Manuscript document: `Manuscript.pdf`; include the LaTeX source only if
  requested.
- Cover letter: `Cover_Letter.pdf`.
- Highlights: paste the five lines from `Highlights.txt` if requested.
- Keywords: enter the six terms from `Keywords.txt`.
- Supplementary material: `Supplementary_Material.pdf`.
- Figures: upload the 11 PDFs from `figures/` if separate display items are
  requested.
- Source data and reproduction evidence: provide only when requested.

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
  `-0.0108` AP50 relative to visible, while shielding improves the no-shield
  selector by `+0.0153`.

## Upload checks

1. Check the live IPT portal for article-type labels, mandatory fields,
   reviewer-suggestion requests, display-item order, and optional graphical
   abstract fields.
2. Confirm that all uploaded content comes from this v18 directory.
3. Compare the portal preview with `Manuscript.pdf` before submission.
4. The live portal remains the only non-technical gate.
