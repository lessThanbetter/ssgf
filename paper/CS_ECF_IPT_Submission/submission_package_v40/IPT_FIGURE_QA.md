# IPT Figure QA

Date: 2026-10-29

Package: v37

## Final display set

- Main figures: 5.
- Supplementary figures: 8.
- Figure PDFs: vector output.
- Source data: one CSV per figure.
- Captions: `Figure 1:` through `Figure 5:` in the manuscript and
  `Figure S1:` through `Figure S8:` in the supplement.

## Typography and layout

- Every source figure was rendered at final physical width.
- All exported PDF text runs meet the 5 pt minimum.
- Figure 1 uses a 6 pt source-text floor so that scaling to the 162 mm
  manuscript text block preserves an effective floor above 5 pt.
- Panel labels, legends, axis labels, direct annotations, intervals, and
  reference lines were checked for clipping and overlap.
- The successful/failed case figure displays only matched true-positive
  predictions and ground truth, so the visual evidence agrees with the AP
  calculation.
- The compact core-evidence figure replaces the separate retrospective
  summaries in the main text.
- Figure 1 was rebuilt as a two-stage framework under one unified visual
  grammar. Candidate images come from LLVIP frame 070142, and its displayed
  scores are read from the archived YOLO26n seed-0 arrays.
- Figure 1 text was reduced from 141 to 92 rendered words. Candidate-level
  score labels were removed because the score panel already reports them.
  A single labeled score path now connects the two stages, and YES/NO branches
  terminate at distinct selection cards.
- The author-revised editable Figure 1 SVG is the source of the packaged PDF,
  PNG, and TIFF. The manuscript embeds the re-rendered vector PDF.
- Figure 5c groups all four deployment checks into an even 2 × 2 block,
  followed by a single "ALL FOUR PASS?" decision and separate YES/NO outcomes.
- Figure 5c adds vector icons for the held-out split, matched switching budget,
  AP50/AP50-95 relevance, and latency, together with the deployment expression
  `D = 1 ⇔ g1 ∧ g2 ∧ g3 ∧ g4`.
- Figure 2 now uses a grouped dot plot for fixed-fusion sensitivity, a forest
  plot for held-out contrasts, and a dumbbell plot for worst-case AP50 loss.
- The author-revised Figure 5 SVG is the source of the packaged PDF, PNG, and
  TIFF. Unsupported logic glyphs were replaced by equivalent readable text so
  the final PDF contains no empty boxes.

## Scientific presentation

- Figure 2 and Table 3 use the same held-out LLVIP and VEDAI estimands.
- The external-evaluator figure retains both visible-relative and no-shield
  reversals.
- Figure 4 uses a fixed case-selection rule and labels the examples as
  illustrative endpoints.
- Figure 5 does not imply a deployment threshold or latency guarantee.
- Supplementary figures retain retrospective, calibration, class, thermal,
  and tail analyses with their stated boundaries.

## Source-data checks

- Each main figure maps to its published CSV in `source_data/`.
- Figure 1 additionally records its example scores in
  `Figure1_example_scores.csv` and its display-size image provenance in
  `Figure1_example_images.csv`.
- Each supplementary figure maps to its published CSV in `source_data/`.
- The joint-baseline table has its own source CSV.
- Figure generation mappings are recorded in
  `reproduction/FIGURE_BUILD_MAP.json`.
