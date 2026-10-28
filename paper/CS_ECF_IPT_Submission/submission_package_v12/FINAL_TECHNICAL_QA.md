# Final Technical QA

Date: 2026-10-28  
Target: *Infrared Physics & Technology*  
Package candidate: v12

## Status

The scientific content is unchanged from v9. Version v10 added explicit PDF
metadata and final technical checks; version v11 records the author instruction
to retain the original administrative and declaration text unchanged. Version
v12 adds the published CS-ECF Zenodo version DOI and is ready for an interactive
check of the live submission portal.

## Build and document checks

- Main manuscript: 13 A4 pages; abstract 221 words.
- Supplementary material: 4 A4 pages.
- Cover letter: 1 A4 page; 470 words in the letter body.
- Highlights: 5 lines, each no longer than 85 characters.
- Bibliography: 27 entries, including 26 DOI-backed records and one software
  repository entry.
- No author-input placeholders, stale interval strings, unresolved citations,
  unresolved cross-references, overfull boxes, or missing figures were found.
- All fonts shown by `pdffonts` were embedded.
- All manuscript, supplement, and cover-letter pages were rendered to PNG and
  inspected. No clipped text, overlapping elements, unreadable glyphs, or broken
  tables were found.
- After the v10 metadata-only rebuild, all 13 manuscript, 4 supplement, and 1
  cover page rasters were byte-identical to the inspected v9 renders.
- The v12 DOI update changed only manuscript page 11. That page was re-rendered
  and inspected; the other 12 manuscript pages and all supplement/cover pages
  remained pixel-identical.

## Reference checks

All 26 DOI records listed in `REFERENCE_VERIFICATION.md` resolved through
Crossref on 2026-10-28. One request failed transiently with an SSL transport
error and succeeded on immediate retry. The remaining bibliography entry is the
Ultralytics software repository, which has no DOI.

## Archive checks

The v9 ZIP contained 35 files apart from its manifest. All 35 hashes matched
`SHA256SUMS.txt`, and no archive file was missing from the manifest. The v12 ZIP
and manifest are regenerated after the DOI, compilation, and administrative
updates.

## Repository check

GitHub release `v2.0.0` was published from commit
`18ea5ba6c35026a472fb14d57a2726cbce00258f`. Zenodo published the corresponding
software archive as version DOI `10.5281/zenodo.23718751` under concept DOI
`10.5281/zenodo.22984388`. The record title, authors, ORCIDs, MIT licence,
software resource type, and GitHub tag relation were checked through the Zenodo
API.

## Administrative confirmation

On 2026-10-28, the author confirmed that the author details, declarations, and
originality wording should remain exactly as in the original submission text.
No scientific or administrative content was changed in response.

## Remaining non-technical gate

- An author must inspect the live IPT portal because anonymous automated
  retrieval is blocked by CAPTCHA or transport restrictions.

