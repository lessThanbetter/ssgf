# CS-ECF IPT Submission Package v15

Prepared: 2026-10-28

Target: *Infrared Physics & Technology*

Article type: Research Paper

## Readiness

Status: **minor-revision repair technically complete; live portal check pending**.

The manuscript, supplement, cover letter, Highlights, keywords, figures,
figure source data, declarations, reproducibility scripts, and external
evaluator evidence are included. The author has confirmed that the
administrative details and declaration text remain unchanged from the original.
The CS-ECF software release is archived in Zenodo as version DOI
`10.5281/zenodo.23720111` under concept DOI `10.5281/zenodo.22984388`. The
remaining task is to check the live submission
portal prompts before upload.

Version v14 responds to the major-revision review. It adds a 12-member Holm/BH
multiplicity family and exact sign test; validation and post-freeze K/margin
analyses; random, image-quality, and oracle controls; selected-fraction,
per-scene, per-image degradation, and ECE/Brier diagnostics; M3FD class counts
and intervals; and LLVIP registration-residual sensitivity. It preserves the
46-entry bibliography and the four-part Related Work synthesis introduced in
v13.

Version v15 resolves the follow-up factual and reproducibility review. It
clarifies that the LLVIP partitions are experiment subsets and records their
selection protocol; defines the CDF detail maps and top-hat threshold; defines
the validation, mean, dataset-balanced, and worst-column estimands; records that
DroneVehicle was not used for tuning; adds registration-scale context and Brier
scores; and includes a Crossref field-level reference audit.

## Files

- `Manuscript.pdf`, `main.tex`, `references.bib`
- `Supplementary_Material.pdf`, `supplementary.tex`
- `Cover_Letter.pdf`, `cover_letter.tex`
- `Highlights.txt`, `Keywords.txt`
- `Submission_Checklist.md`
- `External_Evaluator_Checks.md`
- `FINAL_TECHNICAL_QA.md`
- `REFERENCE_VERIFICATION.md`
- `DECLARATIONS_SOURCE.md`
- `RELATED_WORK_EVIDENCE.md`
- `RELATED_WORK_ADDITIONS.ris`
- `RELATED_WORK_ADDITIONS.json`, `.tsv`, and interactive `.html`
- `MAJOR_REVISION_STATISTICS.md`
- `figures/` with four vector PDFs
- `source_data/` with four figure-level CSV files
- `evidence/` with policy, external-evaluator, multiplicity, baseline,
  calibration, class, registration, and reference-audit outputs
- `reproduction/` with the external-evaluator training, scoring, analysis, and
  preflight scripts
- `SHA256SUMS.txt`

## Verified document state

- Main manuscript: 18 A4 pages, 240-word abstract
- Supplementary material: 6 A4 pages
- Cover letter: 1 A4 page, 491-word body
- References: 46 verified entries, including 45 DOI-backed records
- All 45 DOI records match Crossref title, author count, year, volume, and first
  page after normalization
- Highlights: 5 lines, each no longer than 85 characters
- No undefined citations, unresolved cross-references, overfull boxes, or
  author-input placeholders
- All PDF fonts are embedded
- All 18 manuscript pages, all six supplement pages, and the cover letter were
  visually inspected after the final major-revision build
- Code availability cites the version-specific Zenodo DOI
  `10.5281/zenodo.23720111`

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
- Checkpoint multiplicity: 9/12 positive, exact sign test `p=0.0730`, and no
  checkpoint survives Holm correction at 0.05.
- CS-ECF versus Always CDF: `+0.0102`, interval
  `[-0.00057,+0.02980]`; CDF superiority is not established.
- Controls: uniform random `-0.0341`, maximum VIF `+0.0008`, maximum SSIM
  `-0.0946`, maximum entropy `-0.0599`, and label-using oracle `+0.0833`.
- Zero-shot YOLOv9t/YOLOv10s gain `+0.0250`/`+0.0213` AP50 over visible, but
  neither separates from Always CDF.
- LLVIP-fine-tuned YOLOv9t is a negative portability boundary: CS-ECF is
  `-0.0108` AP50 relative to visible, while the shield improves the unshielded
  selector by `+0.0153`. Its ECE is 0.048 and 41.4% of switched images degrade.

## Upload checks

1. Check the live IPT portal for article-type labels, mandatory fields,
   reviewer-suggestion requests, and any optional graphical abstract.
2. Confirm that all uploaded content comes from this v15 directory and that
   the portal-generated preview matches `Manuscript.pdf`.

The author metadata, declarations, and originality wording are unchanged from
the original. The final archival revision is GitHub tag `v2.2.0`; its Zenodo
version DOI is `10.5281/zenodo.23720111`.
