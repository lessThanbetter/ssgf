# CS-ECF IPT Submission Package v9

Prepared: 2026-10-28

Target: *Infrared Physics & Technology*

Article type: Research Paper

## Readiness

Status: **ready with author checks**.

The manuscript, supplement, cover letter, Highlights, keywords, figures,
figure source data, declarations, reproducibility scripts, and external
evaluator evidence are included. The authors must confirm the inherited
administrative details and check the live submission portal prompts before
upload.

## Files

- `Manuscript.pdf`, `main.tex`, `references.bib`
- `Supplementary_Material.pdf`, `supplementary.tex`
- `Cover_Letter.pdf`, `cover_letter.tex`
- `Highlights.txt`, `Keywords.txt`
- `Submission_Checklist.md`
- `External_Evaluator_Checks.md`
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

1. Confirm author order, affiliation, corresponding author, emails, ORCIDs,
   CRediT roles, funding, competing interests, and the OpenAI Codex AI
   declaration.
2. Confirm the no-preprint and not-under-consideration statements.
3. Confirm that GitHub tag `cs-ecf-ipt-submission-2026-10-28` remains
   accessible; the documented pre-tag commit is `2d53dd1`.
4. Check the live IPT portal for article-type labels, mandatory fields,
   reviewer-suggestion requests, and any optional graphical abstract.
