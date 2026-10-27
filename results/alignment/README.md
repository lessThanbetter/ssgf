# LLVIP alignment sensitivity

The source pairs come from the public LLVIP ZIP and exactly match the 300 stems in
`../../_deposit/ssgf-llvip-fusion-detection-v1/data/aggregate/regime_definitions.csv`.
All 300 source pairs produced a finite estimate.

## Estimator

- Convert both images to grayscale and resize the longest side to 320 px.
- Gaussian-blur by 1.2 px, take Sobel-gradient magnitudes, clip at the 99th
  percentile, and z-normalize.
- Run OpenCV ECC with a translational warp and refine to subpixel precision.
- Multiply the estimated translation by the resize factor and record its Euclidean
  norm as `residual_px`.

The median residual is 2.533 px and the 90th percentile is 4.587 px. The best
quartile contains 75 images with residuals at or below 1.613 px. Selection uses
only the source images, never labels, fusion scores, or detector results.

## Results

Values are mAP50 contrasts against the colour-scored visible source. Brackets for
the four-run clean split are observed minima and maxima across independent detector
seeds; other brackets are image-level 95% paired bootstrap intervals.

| Evaluator and subset | mean | probe | PIAFusion | SPGFusion |
|---|---:|---:|---:|---:|
| pooled visible, best quartile | +.0563 [-.0123,+.1112] | +.0508 [-.0181,+.1088] | +.0300 [-.0468,+.0978] | -.0035 [-.0863,+.0719] |
| pooled visible, worst quartile | -.1135 [-.1515,-.0747] | -.1087 [-.1434,-.0707] | -.1679 [-.2238,-.1091] | -.2031 [-.2631,-.1362] |
| clean visible, best quartile | +.0239 [+.0109,+.0296] | +.0318 [+.0230,+.0451] | +.0287 [+.0019,+.0386] | -.0210 [-.0396,+.0047] |
| clean visible, worst quartile | -.0482 [-.0882,-.0273] | -.0425 [-.0524,-.0170] | -.1419 [-.1556,-.1078] | -.1654 [-.1901,-.1378] |
| clean modality-agnostic, best quartile | +.0765 [+.0626,+.1023] | +.0831 [+.0656,+.1041] | +.0810 [+.0523,+.1165] | +.0389 [+.0072,+.0724] |
| clean modality-agnostic, worst quartile | +.0021 [-.0441,+.0298] | +.0134 [-.0321,+.0446] | +.0434 [-.0122,+.0724] | -.0044 [-.0549,+.0219] |

The subset is not a controlled experiment. Relative to the all-image medians, the
best-aligned quartile has lower visible RMS contrast (.0846 versus .1212), fewer
annotated pedestrians on average (2.12 versus 2.48), and a larger median box area
(31,557 versus 26,903 px). The worst quartile has more pedestrians (2.52) and a
smaller median box area (21,646 px). Registration quality therefore co-varies with
scene content.

The result changes the interpretation: residual alignment strongly amplifies the
negative contrast, and on the best-aligned clean-split quartile the mean, top-hat,
and probe rules have positive point estimates in all four detector seeds. The pooled
all-image deficit remains, but the claim cannot be stated as a universal effect
independent of source alignment or scene composition.

## Reproduction

From the `ssgf/` directory:

```bash
python -m analysis.extract_llvip_sources \
  --stems _deposit/ssgf-llvip-fusion-detection-v1/data/aggregate/regime_definitions.csv \
  --out <llvip-source-subset> --workers 8
python -m analysis.estimate_alignment \
  --sources <llvip-source-subset> --out results/alignment/alignment.json
python -m analysis.alignment_sensitivity \
  --alignment results/alignment/alignment.csv \
  --regime-csv _deposit/ssgf-llvip-fusion-detection-v1/data/aggregate/regime_definitions.csv \
  --per-image _deposit/ssgf-llvip-fusion-detection-v1/data/per_image \
  --out results/alignment/alignment_sensitivity.json \
  --evaluators yolo11n_vis agnostic --bootstrap 1000
```
