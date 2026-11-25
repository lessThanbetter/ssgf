# Revision Notes for v22

Date: 2026-10-29

## Added evidence

1. Added official VEDAI 512 visible--infrared data using fold01 as test,
   fold02 as validation, and folds03--10 as training.
2. Trained two visible-only YOLO11n-OBB detectors with fixed final checkpoints.
3. Generated visible, CDF, PIAFusion, and SPGFusion candidates at registered
   512-pixel geometry.
4. Scored all five retained vehicle classes with native rotated-box IoU and
   one-to-one maximum-cardinality matching.
5. Reported block-level and run-level results for CS-ECF, visible, Always CDF,
   and the no-shield selector.

## Result

VEDAI did not support a visible-relative gain. CS-ECF was `-0.0134`
`[-0.0231,-0.0058]` relative to visible, and neither run had a positive
equal-block mean.

VEDAI did support the narrowed shield claim. CS-ECF exceeded no shield by
`+0.0411` `[+0.0097,+0.0782]`; both independent runs were positive. The
no-shield policy lost 0.0545 AP50 on average and had a worst block loss of
0.2073, whereas CS-ECF lost 0.0134 on average and had a worst block loss of
0.0751.

The paper therefore treats VEDAI as external support for limiting harmful
switching, not as evidence of general accuracy improvement.

## Integrity controls

- The dataset and test fold were selected from metadata before CS-ECF scoring.
- No class, fold, checkpoint, or margin was changed after observing results.
- Negative visible-relative outcomes are retained in the main and
  supplementary tables.
- The VEDAI protocol, data manifest, component hashes, detector checkpoints,
  per-image score arrays, and analysis are archived.
