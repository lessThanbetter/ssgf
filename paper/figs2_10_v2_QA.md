# Figures 2-10 quality and reproducibility report

## Scope and contract

- Figures 1 and 11 are intentionally unchanged.
- Target: Measurement, double-column width.
- Master format: vector PDF with embedded Arial fonts.
- Editable format: SVG with text retained as text.
- Preview: 300 dpi PNG.
- Visual system: common evaluator colours and markers, direct labels, thin axes,
  light guide grids and no decorative chartjunk.

## Redrawn files

| Figure | Data question | Main encoding |
|---|---|---|
| Fig. 2 | Is the negative result stable and does the evaluator reverse it? | Clean-split intervals and pooled sensitivity intervals |
| Fig. 3 | Where is the deficit? | Fourteen-input status matrix, all-input contrast curves and segmentation intervals |
| Fig. 4 | What survives seeds and index construction? | Seed intervals, RT-DETR-l intervals and VIF rank paths |
| Fig. 5 | Does fidelity transfer across evaluators? | VIF-contrast scatter and rank-correlation dot plot |
| Fig. 6 | Does the detection criterion change the conclusion? | IoU sweep for four representative rules |
| Fig. 7 | Is the deficit in recall or precision? | Maximum-F1 recall and precision dot plots |
| Fig. 8 | Is the rule robust to temperature? | Difference from mean fusion over a logarithmic temperature sweep |
| Fig. 9 | Do backbone and inference scale change the sign? | Paired scale comparison and resolution sweep |
| Fig. 10 | Does degradation change the conclusion? | Condition-specific contrasts and fifteen-input status matrices |

## Data validation

The script checks the frozen inputs before drawing. It fails if any of the following
changes:

- all 42 clean visible-trained evaluator-input contrasts remain negative;
- the clean modality-agnostic stable gains are exactly probe, top-hat, ensemble and
  PIAFusion;
- the same four inputs remain above zero in the seed-range file;
- SPGFusion remains first under the reported VIF and last under the reference VIF.

Source files are under `_deposit/ssgf-llvip-fusion-detection-v1/data/aggregate/` and
`results/`. The clean primary results are read directly from
`results/clean_r2/clean_r2_full.json`.

## Statistical traceability

- Fig. 2: 300 held-out LLVIP images; YOLO11n and modality-agnostic means span four
  training runs, while YOLO26n and RT-DETR-l use paired image intervals for one run.
- Fig. 3: pooled sensitivity audit; matrix intervals are uncorrected image intervals,
  with BH-adjusted counts reported in Supplementary Table S16.
- Fig. 4: clean modality-agnostic intervals resample seeds and then images; panel (c)
  uses ranks from the same eight fused variants across VIF constructions.
- Fig. 5: thirteen inputs in the original fidelity export; correlations are descriptive
  Spearman coefficients, not estimates over a population of methods.
- Fig. 6: paired mAP differences at the same ten IoU criteria under two evaluators.
- Fig. 7: maximum-F1 recall and precision at IoU 0.50 for five inputs and four
  evaluators.
- Fig. 8: absolute mAP50 is transformed to the difference from mean fusion so both
  evaluator panels share a zero baseline.
- Fig. 9: YOLO11s uses a 95% interval; resolution-sweep panels use 400 image-bootstrap
  draws.
- Fig. 10: every input is degraded independently and identically; intervals are paired
  image intervals and panel (c) uses the 15-input resolution classification.

## QA

- Academic Figure Skill validator: exit code 0.
- All nine PDFs contain only embedded Arial font subsets.
- All PNG previews are 300 dpi.
- Output widths are approximately 178 mm, within the 183 mm double-column limit.
- No figure text is clipped; all panel labels use one position and type size.
- Colour is redundant with marker shape, line style or direct labels.
- The manuscript compiles to 23 pages with no undefined references or citations.
