# IEEE TIM submission checklist

## Ready files

- `TIM_main.tex` and `TIM_BIBLIOGRAPHY_V1.tex`: IEEEtran manuscript source.
- `TIM_main.pdf`: compiled regular-paper manuscript.
- `TIM_supplementary.tex` and `TIM_supplementary.pdf`: supporting material.
- `TIM_COVER_LETTER.tex` and `TIM_COVER_LETTER.pdf`: initial-submission letter.
- Existing vector figure PDFs and editable SVG files.

## Positioning check

- The title identifies validity and uncertainty as the contribution.
- The abstract identifies fidelity metrics as indirect measurements.
- Section II defines the measurand, reference measurement, proxy measurement and
  measurement context.
- The uncertainty audit separates sampling, training, evaluator,
  metric-construction, source-convention and alignment effects.
- The manuscript states that the bootstrap interval is not a complete GUM uncertainty
  budget.
- The Conclusion is bounded to the measured pedestrian-detection populations.
- The paper does not claim a new fusion network.

## Author checks before portal submission

- Confirm the IEEE TIM article type and any current overlength or supplementary
  material rules in the submission portal.
- Confirm whether the portal requires a separate title page or anonymized manuscript.
- Provide suggested reviewers if requested by the portal.
- Confirm the final author order, affiliations, ORCID links and corresponding author.
- Confirm that the Zenodo DOI is public and that the MIT code licence is correct.
- Confirm the AI-assisted language-editing declaration against IEEE policy.

## Validation

- Main manuscript compiles with IEEEtran to 21 pages, 11 figures, 15 tables and
  49 references.
- Supplementary material compiles to 29 pages.
- No undefined citations or references.
- No overfull horizontal boxes in the final compile.
- Figures and tables are reviewed from rendered PDF pages.
- `scripts/verify_clean_r2.py --root .` returns `CLEAN_R2_OK`.
- The package manifest is regenerated after final edits.
