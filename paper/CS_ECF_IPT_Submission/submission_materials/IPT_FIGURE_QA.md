# IPT Figure QA

Date: 2026-10-28  
Backend: Python / matplotlib  
Manuscript figure count: 11  
Final width: approximately 161 mm (A4 text block)  
Exports: SVG, PDF, 600 dpi TIFF, PNG preview

## Automated checks

| Final figure | Generator | Static preflight | PDF glyph audit |
|---|---|---|---|
| 1 | `make_ipt_fig2.py` | 20/20 pass | pass; minimum 5.4 pt |
| 2 | `make_ipt_fig5.py` | 20/20 pass | pass; minimum 5.1 pt |
| 3 | `make_ipt_fig1.py` | 20/20 pass | pass; minimum 5.7 pt |
| 4 | `make_ipt_fig6.py` | 20/20 pass after removing sub-5-pt footer text | pass; minimum 5.0 pt |
| 5 | `make_ipt_fig10.py` | 20/20 pass | pass; minimum 5.0 pt |
| 6 | `make_ipt_fig7.py` | 20/20 pass | pass; minimum 5.0 pt |
| 7 | `make_ipt_fig8.py` | 20/20 pass | pass; minimum 5.1 pt |
| 8 | `make_ipt_fig9.py` | 20/20 pass | pass; minimum 5.0 pt |
| 9 | `make_ipt_fig3.py` | 20/20 pass | pass; minimum 5.0 pt |
| 10 | `make_ipt_fig4.py` | 20/20 pass | pass; minimum 5.2 pt |
| 11 | `make_ipt_fig11.py` | 19/20 pass plus one documented log-axis warning | pass; minimum 5.0 pt |

The 161 mm width warning is expected because the manuscript uses a 162 mm
text block rather than Nature's 89/183 mm defaults. The Figure 11 warning
concerns a positive log-scaled latency axis used only for the cost display.

## Manual panel-by-panel audit

| Figure | Unique claim | Statistics / interval | Collision and hierarchy check | Pass |
|---|---|---|---|---|
| 1 | A fused input is used only after a large detector-supported margin; visible remains the fallback. | Decision counts over checkpoint-image trials; no labels used for selection. | Panel labels, arrows, detector block, margin gate, fallback path, and stacked percentages are legible. | yes |
| 2 | The threshold was frozen before confirmation and the K/margin grid is sensitivity analysis. | Validation objective; post-freeze descriptive cells. | Margin ticks reduced to avoid collision; selected cell outlined once. | yes |
| 3 | Candidate utility changes sign across evaluator checkpoints and datasets. | Signed AP50 point contrasts and dataset means; no per-cell significance. | Matrix labels, selected-policy outline, colorbar, and dataset panel are clear. | yes |
| 4 | Positive point estimates do not survive multiplicity correction. | Scene/frame-group intervals, sign test, Holm/BH survivors, aggregate interval. | Forest labels and summary card checked; no raw-p columns crowd the plot. | yes |
| 5 | CS-ECF is not separated from Always CDF. | Dataset/group-aware intervals and paired checkpoint differences. | The zero-crossing interval, equality line, dataset colors, and six/six split remain visible. | yes |
| 6 | Confidence selection beats label-free controls but remains below the oracle. | Random-policy mean, checkpoint extrema, image-bootstrap or released aggregate controls as documented. | Positive-count column separated from gain labels; oracle identified as label-using upper bound. | yes |
| 7 | Portability reverses after LLVIP fine-tuning. | Scene-group intervals and three-check BH q-values. | Absolute performance and paired contrasts are separated; negative result is prominent. | yes |
| 8 | Calibration error and switched-image degradation accompany the fine-tuned failure. | Detection ECE/Brier, switched-outcome fractions, scene contrasts. | ECE markers, stacked bars, scene heatmap, and legends checked. | yes |
| 9 | The shield reduces worst-case risk at a cost in mean return. | Run-cluster-by-group interval; paired checkpoint changes. | Low-tail guard and large-regression region remain visible; labels clear the markers. | yes |
| 10 | The M3FD gain is people-specific; thermal association is domain-dependent. | Class image-bootstrap intervals, annotation counts, mask ablation, Spearman rho. | Previously clipped people-class intervals are now fully visible on the percentage-point axis. | yes |
| 11 | Cost is dominated by SPGFusion and deployment depends on stability, calibration, and latency. | Stage timings, peak memory, parameter counts; no additive-memory claim. | Log ticks corrected; decision matrix avoids crossing arrows; all card text at least 5 pt. | yes |

## Manuscript-level checks

- The compiled manuscript contains 11 figure environments and 11 unique figure
  references.
- Figure numbers follow first appearance through Methods, Results, and
  Discussion.
- The main PDF renders as 25 A4 pages after three LaTeX/BibTeX passes, with no
  undefined citations, unresolved cross-references, overfull boxes, or duplicate
  Discussion heading.
- Figure captions are attached to their figures and include panel definitions,
  statistical meaning, and source-data filenames.
- Every final figure PDF is present in
  `CS_ECF_IPT_Submission/figures/` under a filename matching its manuscript
  number.

## Source data

`Figure1_source.csv` through `Figure11_source.csv` are stored in
`CS_ECF_IPT_Submission/source_data/`.

## Remaining production note

The live IPT portal still determines whether separate figure files, a graphical
abstract, or a specific display-item order is requested. No additional
experiment is needed for the figure set.
