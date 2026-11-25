# Revision notes for v27

Date: 2026-10-29

## Algorithm framework figure

Figure 1 was rebuilt as a two-stage workflow with a consistent visual grammar
inspired by the provided framework reference. The scientific content remains
specific to CS-ECF:

- Stage 1 shows the registered visible-infrared input pair, four frozen
  candidates, target-detector scoring, and the absence of per-image ground
  truth.
- Stage 2 shows score preselection, the visible-relative 1.5 margin gate, the
  accepted-fusion and visible-fallback branches, and the final selected input.
- Panel b retains the observed selection composition for LLVIP, M3FD, and
  DroneVehicle.

## Data and presentation

- The example uses LLVIP frame 070142.
- Candidate scores are computed from the archived YOLO26n seed-0 score arrays:
  Visible 0.068, CDF 0.100, PIAFusion 0.133, and SPGFusion 0.170.
- Candidate images are resized only for display. The frame identity and display
  dimensions are recorded in `Figure1_example_images.csv`.
- The nominal source canvas is approximately 182 mm wide and is scaled to the
  162 mm manuscript text block. Figure text is at least 6 pt before scaling
  and remains at least 5 pt after placement; SVG and PDF text are editable.

## Verification

- Python figure-source preflight: 20 pass, 0 warn, 0 fail.
- PDF text audit: minimum 5 pt.
- Main manuscript: 18 pages and five figures.
- Supplementary material: 19 pages and eight figures.
