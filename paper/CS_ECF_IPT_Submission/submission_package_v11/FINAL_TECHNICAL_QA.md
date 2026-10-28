# Final Technical QA

Date: 2026-10-28  
Target: *Infrared Physics & Technology*  
Package candidate: v11

## Status

The scientific content is unchanged from v9. Version v10 added explicit PDF
metadata and final technical checks; version v11 records the author instruction
to retain the original administrative and declaration text unchanged. The
package is ready for an interactive check of the live submission portal.
The frozen GitHub tag continues to identify the same analysis code and data;
the v10 metadata and QA-document changes are packaging-only.

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
- After the metadata-only rebuild, all 13 manuscript, 4 supplement, and 1 cover
  page rasters were byte-identical to the inspected v9 renders.

## Reference checks

All 26 DOI records listed in `REFERENCE_VERIFICATION.md` resolved through
Crossref on 2026-10-28. One request failed transiently with an SSL transport
error and succeeded on immediate retry. The remaining bibliography entry is the
Ultralytics software repository, which has no DOI.

## Archive checks

The v9 ZIP contained 35 files apart from its manifest. All 35 hashes matched
`SHA256SUMS.txt`, and no archive file was missing from the manifest. The v11 ZIP
and manifest are regenerated after the metadata, QA, and administrative-status
updates.

## Repository check

The remote tag `cs-ecf-ipt-submission-2026-10-28` exists on
`https://github.com/lessThanbetter/ssgf`. Its annotated tag object is
`eb403a93ca7375d4e749571d1806be0124a24c86` and resolves to commit
`2d53dd164ba5b212188a3a547faeac7e0f343a8d`.

## Administrative confirmation

On 2026-10-28, the author confirmed that the author details, declarations, and
originality wording should remain exactly as in the original submission text.
No scientific or administrative content was changed in response.

## Remaining non-technical gate

- An author must inspect the live IPT portal because anonymous automated
  retrieval is blocked by CAPTCHA or transport restrictions.

