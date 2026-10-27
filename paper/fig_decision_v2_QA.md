# Figure 11 quality and reproducibility report

## Figure contract

- Target: Measurement, single-column layout.
- Core conclusion: the ranking of fixed fused inputs changes with the evaluator and
  the fidelity-index construction.
- Archetype: schematic-led composite containing one source example and two evidence
  plots.
- Output: vector PDF master, editable SVG, and 300 dpi PNG preview.

## Visual encoding

- (a) LLVIP test frame 190002 shows the visible source, infrared source and mean-fusion
  output under the same crop.
- (b) Each marker is one of 14 fused inputs. Horizontal ranges show all training runs;
  the vertical zero line is the colour-scored visible source. The three blue rows are
  clean visible-only evaluators. The lower purple row is the clean modality-agnostic
  evaluator, with stable positive inputs in red.
- (c) Each line is the same fusion input connected across its rank under the reported
  and reference VIF. Grey lines are the other 13 inputs; the red line highlights
  SPGFusion from rank 1 to rank 14.
- The footer states the conditional relationship as an equation rather than a prose
  conclusion. Experimental detail remains in the caption.

## Evidence assertions

The plotting script reads the result JSON files and fails if any assertion changes:

- 42 of 42 clean visible-only evaluator-input contrasts are negative.
- Four fused inputs are positive in every clean modality-agnostic run.
- SPGFusion ranks first under the reported VIF.
- SPGFusion ranks last among 14 inputs under the reference pixel VIF.

## Statistical traceability

- Evaluator: clean YOLO11n and modality-agnostic models use four training runs;
  clean YOLO26n and RT-DETR-l use one run each. Markers show the run mean and ranges
  show the full observed run spread.
- Test data: the fixed 300-image LLVIP test split, never used for training, validation
  or checkpoint selection.
- Index: reported and reference pixel VIF measured on the same 14 inputs and the same
  300-image set. Ranks are descending; larger VIF is better.
- Source files: `results/clean_r2/clean_r2_full.json` and
  `results/reference_vif/llvip_14_full.json`.

## Visual QA

- Automated Academic Figure Skill validator: exit code 0.
- PDF page size: 263.85 by 286.53 points, approximately 93.1 by 101.1 mm.
- PNG preview: 1099 by 1193 pixels at 300 dpi.
- All type is at least 5.3 pt at the final 88 mm width.
- PDF fonts are embedded; the PNG is exported at 300 dpi.
- No labels, ranges or rank lines are clipped or obscured.
- Colour is redundant with row position, direct labels and line weight.
- The rendered figure contains no prose cards, dense paragraphs, decorative grid or
  hidden legend.
