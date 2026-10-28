# Final Technical QA

Date: 2026-10-28  
Target: *Infrared Physics & Technology*  
Package candidate: v16

## Status

The scientific content has changed materially from v9 in response to the
major-revision review. Version v14 added multiplicity control, selection and
sensitivity protocols, deployable controls and an oracle headroom bound,
external-evaluator calibration diagnostics, class-specific M3FD evidence, and
registration sensitivity. The manuscript now reports the negative and
non-separated findings as the primary claim boundary.

Version v15 resolves the follow-up factual and reproducibility review. It
identifies the LLVIP partitions as fixed subsets with explicit selection rules,
defines the CDF detail maps and top-hat quantile, clarifies the validation and
policy-table estimands, states that DroneVehicle was not used for margin
selection, adds registration-residual scale context and supplementary Brier
scores, and removes run-in heading punctuation artifacts.

Version v16 adds a technical-check gate for the rendered PDF itself. It
verifies all 46 numbered reference entries, the complete `Figure 1:` caption,
a single Discussion heading, correct `RT-DETR-l` naming, and the absence of
repeated-character corruption lines. It also records the complete pre-freeze
margin search and the exact LLVIP test sampling frame.

## Build and document checks

- Main manuscript: 18 A4 pages; abstract 240 words.
- Supplementary material: 6 A4 pages.
- Cover letter: 1 A4 page; 491 words in the letter body.
- Highlights: 5 lines, each no longer than 85 characters.
- Bibliography: 46 entries, including 45 DOI-backed records and one software
  repository entry.
- No author-input placeholders, stale interval strings, unresolved citations,
  unresolved cross-references, overfull boxes, or missing figures were found.
- All fonts shown by `pdffonts` were embedded.
- All manuscript, supplement, and cover-letter pages were rendered to PNG and
  inspected. No clipped text, overlapping elements, unreadable glyphs, or broken
  tables were found.
- The v14 analysis adds nine multiplicity-adjusted checkpoint rows, two
  sensitivity grids, a control/headroom table, an external-diagnostics table,
  a class-count table, and registration sensitivity. All 18 manuscript pages,
  all six supplement pages, and the cover letter were rendered and visually
  inspected.
- The five new analysis scripts pass `py_compile`. A bounded rerun of the main
  statistical script reproduces the sign test, the absence of Holm-surviving
  checkpoints, the balanced point estimate, and the oracle upper bound.
- The reference auditor compares DOI resolution and normalized Crossref title,
  author count, year, volume, and first page metadata against every BibTeX
  record.
- A forced clean rebuild produced an 18-page manuscript with SHA-256
  `FF97AC5FFB69AE6F64F73844A2DA26B49CEB7C3367044799CE1BD0F9AE7B78B1`.
  Page 16 contains references [1]--[9], and page 17 begins with [10]; both
  rendered text extractors found no repeated-character line.

## Reference checks

All 45 DOI records listed in `REFERENCE_VERIFICATION.md` resolved through
Crossref in a fresh field-level audit on 2026-10-28. The audit found zero
unresolved title, author-count, year, volume, or first-page differences. The
remaining bibliography entry is the Ultralytics software repository, which has
no DOI.

## Archive checks

The v16 ZIP and manifest are regenerated after the statistical, citation,
archival-DOI, compilation, and administrative updates. Every packaged file is
hashed, every manifest entry is present, and the ZIP-sidecar digest is checked.

## Repository check

The `v2.3.0` release packages the technical-check corrections and PDF gate. Its
Zenodo version DOI is `10.5281/zenodo.23721458` under concept DOI
`10.5281/zenodo.22984388`. The record title, authors, ORCIDs, MIT licence,
software resource type, and release relation are checked through the Zenodo API
before the version is published.

## Administrative confirmation

On 2026-10-28, the author confirmed that the author details, declarations, and
originality wording should remain exactly as in the original submission text.
No scientific or administrative content was changed in response.

## Remaining non-technical gate

- An author must inspect the live IPT portal because anonymous automated
  retrieval is blocked by CAPTCHA or transport restrictions.

