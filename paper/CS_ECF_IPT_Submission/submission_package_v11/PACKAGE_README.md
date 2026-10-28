# CS-ECF IPT Submission Package v11

Prepared: 2026-10-28

Target: *Infrared Physics & Technology*

Article type: Research Paper

## Readiness

Status: **author-confirmed; live portal check pending**.

The manuscript, supplement, cover letter, Highlights, keywords, figures,
figure source data, declarations, reproducibility scripts, and external
evaluator evidence are included. The author has confirmed that the
administrative details and declaration text remain unchanged from the original.
The remaining task is to check the live submission portal prompts before upload.

Version v10 preserves the v9 scientific content and page rendering. It adds
document metadata, a final technical-QA record, the reference-verification
ledger, and corrected package guidance. Version v11 records the author's
confirmation that no administrative or declaration text should change.

## Files

- `Manuscript.pdf`, `main.tex`, `references.bib`
- `Supplementary_Material.pdf`, `supplementary.tex`
- `Cover_Letter.pdf`, `cover_letter.tex`
- `Highlights.txt`, `Keywords.txt`
- `Submission_Checklist.md`
- `External_Evaluator_Checks.md`
- `FINAL_TECHNICAL_QA.md`
- `REFERENCE_VERIFICATION.md`
- `figures/` with four vector PDFs
- `source_data/` with four figure-level CSV files
- `evidence/` with policy, multiplicity, score, and interval outputs
- `reproduction/` with the external-evaluator training, scoring, analysis, and
  preflight scripts
- `SHA256SUMS.txt`

## Verified document state

- Main manuscript: 13 A4 pages, 221-word abstract
- Supplementary material: 4 A4 pages
- Cover letter: 1 A4 page
- References: 27 verified entries
- Highlights: 5 lines, each no longer than 85 characters
- No undefined citations, unresolved cross-references, overfull boxes, or
  author-input placeholders
- All PDF fonts are embedded
- All 13 manuscript, 4 supplement, and 1 cover page renders are byte-identical
  to the inspected v9 page renders

## Portal upload set

Use the live portal labels rather than assuming a fixed upload order.

- Manuscript document: `Manuscript.pdf`; use the LaTeX sources only if the
  portal explicitly requests an editable source package.
- Cover letter: `Cover_Letter.pdf`.
- Highlights: paste the five lines from `Highlights.txt` if a text field or
  separate file is requested.
- Keywords: enter the six terms from `Keywords.txt`.
- Supplementary material: `Supplementary_Material.pdf`, if requested.
- Figures: upload from `figures/` only if the portal asks for separate display
  items. They are already embedded in the compiled manuscript.
- Source data and reproducibility evidence: upload or expose separately only
  when requested; do not replace the required manuscript with the internal
  package ZIP.

## Evidence boundary

- CS-ECF versus visible: dataset-balanced AP50 gain `+0.0133`, hierarchical
  95% interval `[+0.0026,+0.0261]`.
- CS-ECF versus Always CDF: `+0.0102`, interval
  `[-0.00057,+0.02980]`; CDF superiority is not established.
- Zero-shot YOLOv9t/YOLOv10s gain `+0.0250`/`+0.0213` AP50 over visible, but
  neither separates from Always CDF.
- LLVIP-fine-tuned YOLOv9t is a negative portability boundary: CS-ECF is
  `-0.0108` AP50 relative to visible, while the shield improves the unshielded
  selector by `+0.0153`.

## Upload checks

1. Check the live IPT portal for article-type labels, mandatory fields,
   reviewer-suggestion requests, and any optional graphical abstract.
2. Confirm that all uploaded content comes from this v11 directory and that
   the portal-generated preview matches `Manuscript.pdf`.

The author metadata, declarations, and originality wording are unchanged from
the original. The GitHub submission tag is already verified and resolves to
commit `2d53dd1`.
