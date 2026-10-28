# IPT Figure Contract

Date: 2026-10-28  
Target: Infrared Physics & Technology  
Backend: Python / matplotlib  
Final width: approximately 161 mm to match the A4 manuscript text block  
Typography: source text at least 5 pt; exported PDF text floor 5 pt  
Formats: editable SVG and PDF, 600 dpi TIFF, PNG preview  
Reference language: flat editorial diagrams, pale outlined containers, thin
arrows, restrained blue/teal/orange/purple encoding, direct labels, compact
multi-panel plots, and `(a)`, `(b)`, `(c)` panel labels

## Final evidence chain

| Figure | Core conclusion | Archetype | Reviewer boundary |
|---|---|---|---|
| 1 | CS-ECF admits a fused input only when the target detector supports a frozen visible-relative margin; otherwise it retains visible. | schematic-led composite | The flow must show detector reuse and visible fallback, not a label oracle. |
| 2 | Margin 1.5 was selected on validation before confirmation, and post-freeze K/margin cells are descriptive. | quantitative grid | Do not imply that test results selected the threshold. |
| 3 | The same candidate library changes sign across evaluator checkpoints and datasets. | hero heatmap plus dataset summary | Cells are point contrasts, not independent significance tests. |
| 4 | Nine of 12 checkpoint point estimates are positive, but no individual contrast survives Holm or BH correction and the sign test is not significant. | forest plot plus evidence summary | Preserve `p=0.073`, `0/12` corrected survivors, and the positive aggregate interval. |
| 5 | CS-ECF is not established as superior to Always CDF; the balanced interval crosses zero. | multi-panel paired comparison | Keep the balanced interval and six-versus-six checkpoint count visible. |
| 6 | Detector confidence adds information beyond random and image-quality controls, while substantial oracle headroom remains. | forest plot plus linear headroom comparison | The oracle uses labels and is an upper bound only. |
| 7 | Portability holds for two zero-shot evaluators but reverses for an LLVIP-fine-tuned evaluator. | asymmetric paired comparison | Preserve the negative result and BH exploratory status. |
| 8 | The fine-tuned failure is associated with calibration error, more degraded switches, and negative scene groups; causality is unresolved. | quantitative triptych | Do not attribute the reversal solely to calibration without an intervention. |
| 9 | The shield trades a smaller average return for a much smaller worst-case loss. | hero risk-return plane plus paired panels | Axes must retain the large-regression region and zero lines. |
| 10 | The M3FD gain is concentrated in people; thermal association reverses by domain. | asymmetric mixed quantitative grid | Two test groups make class intervals descriptive; thermal evidence is not a universal switch score. |
| 11 | Candidate generation is dominated by SPGFusion, total serial cost is about 706 ms, and deployment depends on evaluator stability, calibration, and latency. | cost panels plus decision matrix | No pruning or early termination was implemented. |

## Source-data map

| Final figure | Generator | Source data |
|---|---|---|
| 1 | `scripts/make_ipt_fig2.py` | `Figure1_source.csv` |
| 2 | `scripts/make_ipt_fig5.py` | `Figure2_source.csv` |
| 3 | `scripts/make_ipt_fig1.py` | `Figure3_source.csv` |
| 4 | `scripts/make_ipt_fig6.py` | `Figure4_source.csv` |
| 5 | `scripts/make_ipt_fig10.py` | `Figure5_source.csv` |
| 6 | `scripts/make_ipt_fig7.py` | `Figure6_source.csv` |
| 7 | `scripts/make_ipt_fig8.py` | `Figure7_source.csv` |
| 8 | `scripts/make_ipt_fig9.py` | `Figure8_source.csv` |
| 9 | `scripts/make_ipt_fig3.py` | `Figure9_source.csv` |
| 10 | `scripts/make_ipt_fig4.py` | `Figure10_source.csv` |
| 11 | `scripts/make_ipt_fig11.py` | `Figure11_source.csv` |

## Global statistical contract

- Confirmatory family: 12 prespecified CS-ECF-minus-visible checkpoint
  contrasts.
- Paired intervals resample scene or consecutive frame groups; combined
  intervals also account for training-run clusters.
- No individual checkpoint survives Holm or BH correction at 0.05.
- The dataset-balanced gain is +0.0133 with 95% interval [0.0026, 0.0261].
- CS-ECF minus Always CDF is +0.0102 with 95% interval [-0.00057, 0.02980].
- The fine-tuned YOLOv9t result is a negative portability boundary, not a
  successful replication.
- DroneVehicle best/last checkpoints come from one training run and one model
  cluster.

## Export and QA contract

- Every figure exports SVG, PDF, 600 dpi TIFF, and PNG.
- PDF text remains editable and every rendered glyph is at least 5 pt.
- Every quantitative panel traces to a clean CSV.
- Whole figures are inspected at final physical size, including panel labels,
  legends, uncertainty extents, zero lines, and direct annotations.
- Final manuscript figures are named `Figure1_*.pdf` through
  `Figure11_*.pdf` so upload filenames match their displayed numbers.
