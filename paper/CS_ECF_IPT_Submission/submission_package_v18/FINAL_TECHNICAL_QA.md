# Final Technical QA

Date: 2026-10-28  
Target: *Infrared Physics & Technology*  
Package candidate: v18

## Status

Version v18 retains the 11-figure evidence sequence from v17 and adds a
language-focused revision of the title, abstract, Introduction, Methods,
Results, Discussion, Conclusions, figure captions, supplement, and cover letter.
The revision removes unnecessary repetition, shortens overloaded sentences, and
aligns terminology without changing the experiments or quantitative results.

The figure set replaces the earlier four-figure layout with an
11-figure evidence sequence
matching the target reference paper's display-item count. The redesign adds
explicit figures for the workflow, threshold sensitivity, multiplicity
correction, direct CDF comparison, control headroom, external-evaluator
portability, calibration failure, risk-return trade-off, thermal/class
boundaries, and deployment cost.

The scientific claims are unchanged. The new visuals preserve the principal
negative and non-separated results: no checkpoint survives Holm/BH correction,
the sign test is not significant, CS-ECF does not separate from Always CDF, and
the LLVIP-fine-tuned YOLOv9t reverses sign relative to visible.

## Build and document checks

- Main manuscript: 25 A4 pages; abstract 250 words.
- Supplementary material: 6 A4 pages.
- Cover letter: 1 A4 page; letter body 490 words.
- Main figures: exactly 11 figure environments, 11 unique labels, and 11
  rendered captions.
- Figure-level source data: 11 CSV files.
- Highlights: 5 lines, each no longer than 85 characters.
- Bibliography: 46 entries, including 45 DOI-backed records and one software
  repository entry.
- No author-input placeholders, stale interval strings, unresolved citations,
  unresolved cross-references, overfull boxes, or missing figures were found.
- Rendered PDF text contains every caption from `Figure 1:` through
  `Figure 11:`, every reference marker from [1] through [46], one Discussion
  heading, and no `??` unresolved-reference marker.
- All manuscript, supplement, and cover-letter pages were rendered to PNG and
  inspected after the final build.
- All 11 figures were inspected at final physical size. Panel titles, labels,
  legends, uncertainty extents, direct annotations, and captions were checked.
- All 11 figure PDFs pass the rendered-text audit: zero text runs below 5 pt.
- The minimum observed figure text size is 5.0 pt.

## Figure QA summary

| Figure | Evidence purpose | Automated result |
|---|---|---|
| 1 | Frozen workflow and visible fallback | static preflight pass; 5.4 pt minimum |
| 2 | Pre-freeze selection and K/margin sensitivity | static preflight pass; 5.1 pt minimum |
| 3 | Evaluator-dependent fusion utility | static preflight pass; 5.7 pt minimum |
| 4 | Multiplicity-corrected checkpoint evidence | static preflight pass; 5.0 pt minimum |
| 5 | Direct CS-ECF versus Always CDF | static preflight pass; 5.0 pt minimum |
| 6 | Label-free controls and oracle headroom | static preflight pass; 5.0 pt minimum |
| 7 | External evaluator portability | static preflight pass; 5.1 pt minimum |
| 8 | Calibration and switched-image boundary | static preflight pass; 5.0 pt minimum |
| 9 | Risk-return trade-off | static preflight pass; 5.0 pt minimum |
| 10 | Class and thermal boundaries | static preflight pass; 5.2 pt minimum |
| 11 | Cost and deployment boundaries | pass with documented log-axis warning; 5.0 pt minimum |

## Reference checks

All 45 DOI records listed in `REFERENCE_VERIFICATION.md` resolved through
Crossref in a field-level audit on 2026-10-28. The audit found zero unresolved
title, author-count, year, volume, or first-page differences. The remaining
bibliography entry is the Ultralytics software repository, which has no DOI.

## Archive checks

The v18 submission ZIP and standalone sidecar digest are regenerated after the
language revision, DOI update, compilation, and PDF checks. Every packaged file
is hashed, every manifest entry is present, and the sidecar digest is verified.

The version-specific manuscript DOI is `10.5281/zenodo.23728385` under concept
DOI `10.5281/zenodo.22984388`. The GitHub tag and Zenodo publication are
verified only after the final package upload.

## Administrative confirmation

On 2026-10-28, the author confirmed that the author details, declarations, and
originality wording should remain exactly as in the original submission text.
No administrative content was changed in response.

## Remaining non-technical gate

An author must inspect the live IPT portal because anonymous automated retrieval
is blocked by CAPTCHA or transport restrictions.
