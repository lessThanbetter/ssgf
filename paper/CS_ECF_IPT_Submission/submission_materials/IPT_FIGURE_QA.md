# IPT Figure QA

Date: 2026-10-28  
Backend: Python / matplotlib  
Final width: approximately 160 mm  
Exports: SVG, PDF, 600 dpi TIFF, PNG preview

## Automated checks

| Figure | Static preflight | PDF glyph audit |
|---|---|---|
| Fig. 1 | 20/20 pass | PASS, min about 6.1 pt |
| Fig. 2 | 20/20 pass | PASS, min about 6.8 pt |
| Fig. 3 | 20/20 pass | PASS, min about 7.6 pt |
| Fig. 4 | 20/20 pass | PASS, min about 8.1 pt |

Manual visual QA was performed at full rendered size after the automated checks.

The final manuscript build contains 16 pages, the supplementary material
contains 4 pages, and the cover letter contains 1 page. There are no undefined
citations, unresolved cross-references, overfull LaTeX boxes, or author-input
placeholders after the final recompilation.

## Panel audit

| Figure | Panel | Unique claim | Statistics | Collision check | Pass |
|---|---|---|---|---|---|
| 1 | a | Fixed-policy utility changes sign across checkpoints | Point AP50 contrasts; no significance claimed per cell | Long y labels and colorbar clear | yes |
| 1 | b | Fusion ranking reverses across datasets | Dataset-level checkpoint contrast | Legend separated above bars | yes |
| 2 | a | The target detector arbitrates four input candidates with visible fallback | No statistical estimate | Arrows, equations, and blocks clear | yes |
| 2 | b | Most frozen decisions retain visible input | Counts divided by checkpoint–image trials | Stacked labels and legend clear | yes |
| 3 | a | CS-ECF occupies a different risk–return position from fixed/no-shield policies | Dataset-balanced contrast and CS-ECF run-cluster/group CI on x | Labels remain inside panel | yes |
| 3 | b | The shield reduces worst losses while sacrificing some DroneVehicle gain | Paired checkpoint contrast; zero line shown | Long labels shortened; no panel crossing | yes |
| 4 | a | M3FD gain is concentrated in people | Point AP50 contrasts | Dominate class effect remains intentional | yes |
| 4 | b | Thermal mask modestly exceeds global injection | Fixed-input AP50 contrast | Legend clears bars | yes |
| 4 | c | Thermal association is not universal | Spearman rho; p values in source/caption | Legend below panel; zero line visible | yes |

## Statistical definitions

- Fig. 1: signed AP50 difference from the visible input. Individual cells are
  point estimates, not independent significance tests.
- Fig. 2: decision fractions use all checkpoint–image decisions within each
  dataset.
- Fig. 3a: x is the equal-weight mean of LLVIP, M3FD, and DroneVehicle; y is the
  worst checkpoint gain. The CS-ECF error bar is the training-run-by-group hierarchical
  interval.
- Fig. 3b: lines connect the same checkpoint under no-shield and CS-ECF.
- Fig. 4: class, mask, and thermal panels report descriptive contrasts. Thermal
  Spearman p values are provided in source data and the caption.

## Source data

- `figure1_source.csv`
- `figure2_source.csv`
- `figure3_source.csv`
- `figure4_source.csv`

## Remaining production notes

- Verify the current IPT artwork instruction for TIFF color mode, compression,
  and maximum figure count in the live submission portal.
- Figure captions must state that LLVIP intervals cross zero and DroneVehicle
  uses two checkpoints from the same training run.
- No representative IR image plate is included yet. IPT may benefit from one
  additional qualitative panel, but it should not replace the quantitative
  boundary evidence.

