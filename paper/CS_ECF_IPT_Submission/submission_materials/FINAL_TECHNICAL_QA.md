# Final Technical QA

Date: 2026-10-28  
Target: *Infrared Physics & Technology*  
Package candidate: v14

## Status

The scientific content has changed materially from v9 in response to the
major-revision review. Version v14 adds multiplicity control, selection and
sensitivity protocols, deployable controls and an oracle headroom bound,
external-evaluator calibration diagnostics, class-specific M3FD evidence, and
registration sensitivity. The manuscript now reports the negative and
non-separated findings as the primary claim boundary.

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

## Reference checks

All 45 DOI records listed in `REFERENCE_VERIFICATION.md` resolved through
Crossref in a fresh audit on 2026-10-28. Transient SSL failures were retried.
The remaining bibliography entry is the Ultralytics software repository, which
has no DOI.

## Archive checks

The v14 ZIP and manifest are regenerated after the statistical, citation,
archival-DOI, compilation, and administrative updates. Every packaged file is
hashed, every manifest entry is present, and the ZIP-sidecar digest is checked.

## Repository check

The `v2.1.0` release packages the major-revision analyses and scripts. Its
Zenodo version DOI is `10.5281/zenodo.23719721` under concept DOI
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

