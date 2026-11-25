# Revision notes for v28

Date: 2026-10-29

## Figure readability

Figure 1 was simplified in response to the visual review.

- Removed duplicate candidate-level score labels; candidate thumbnails now show
  only the four input names.
- Shortened stage subtitles, candidate labels, detector labels, branch labels,
  and the final-output label.
- Replaced the interrupted downward score arrow with one labeled path from the
  score panel to the Stage 2 top-candidate card.
- Kept YES and NO paths spatially separate and directed each to its own
  selection card before convergence at the final detections.
- Preserved the real LLVIP 070142 images and recorded candidate scores.

The rendered word count in the figure decreased from 141 to 92 while retaining
the algorithm, data example, and observed selection composition.

## Verification

- Figure-source preflight: 20 pass, 0 warn, 0 fail.
- PDF text audit: minimum 6 pt, no glyphs below the 5 pt requirement.
- Figure width remains compatible with the manuscript text block.
- Main manuscript remains 18 pages and five figures.
