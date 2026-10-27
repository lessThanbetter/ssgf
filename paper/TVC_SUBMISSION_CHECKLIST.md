# The Visual Computer submission checklist

## Positioning

- Title: `Fidelity Metrics Do Not Reliably Rank Visible--Infrared Fusion Rules
  for Pedestrian Detection`.
- Contribution: detector-dependent reliability of fidelity rankings for
  visible-infrared fusion.
- The manuscript does not claim a new fusion architecture.

## Files prepared

- `TVC_main.tex` and `TVC_BIBLIOGRAPHY.tex`: editable LaTeX source.
- `TVC_main.pdf`: A4, 10 pt Times, continuous page numbers, decimal headings.
- `TVC_supplementary.tex` and `TVC_supplementary.pdf`: supporting analyses.
- `TVC_COVER_LETTER.tex` and `TVC_COVER_LETTER.pdf`: initial cover letter.
- Vector PDF and editable SVG figure sets.

## Official-requirement checks

- Single-blind review: author identities are present in the manuscript.
- Abstract target: 150--250 words.
- Keywords: 4--6 terms.
- References: numbered in first-citation order, with DOI links where available.
- Tables and figures: Arabic numbering.
- Declarations: author contributions, competing interests, funding, data and code
  availability, and generative-AI assistance are grouped for inclusion.
- Source files: editable LaTeX and bibliography are included in the package.

## Author checks before portal submission

- Confirm the current article type and any optional highlights or graphical
  abstract requirements in the submission system.
- Provide suggested reviewers if requested.
- Confirm author order, affiliations, corresponding author and ORCIDs.
- Confirm the Zenodo DOI resolves publicly and the MIT licence is correct.
- Confirm the AI-assisted language-editing declaration against current Springer
  Nature policy.

## Validation

- Main manuscript compiles to 23 A4 pages, 11 figures and 14 tables.
- Supplementary material compiles separately.
- No undefined citations or references.
- No overfull horizontal boxes.
- Numeric audit against frozen data returns zero failures.
- `scripts/verify_clean_r2.py --root .` returns `CLEAN_R2_OK`.
