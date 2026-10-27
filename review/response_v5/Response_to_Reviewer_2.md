# Response to Reviewer Comments

**Previous manuscript:** Evaluator-Dependent Validity of Fidelity Metrics for
Visible-Infrared Image Fusion in Pedestrian Detection

**Previous submission:** INFPHY-D-26-02136

**Prior editorial decision:** Reject

We thank the reviewer for the focused comments on figure interpretation, independent
validation, reference-index recalculation and AP computation. The current manuscript
addresses these points directly. The most important change is that the clean split
with all 300 test images held out is now the primary result for every major
evaluator, while the earlier pooled audit is retained only as an explicitly
optimistic sensitivity analysis. Table numbers below refer to
`PAPER_MEASUREMENT_v5.pdf`.

## Comment 1

> The results in Fig. 5(b) appear inconsistent with the figure caption. The caption
> states that none of the metrics maintains the same direction of correlation across
> different evaluators. However, the correlation coefficients for
> SSIM\(_{\mathrm{vis}}\) are positive for all four evaluators, as also reported in
> Table 7. Please check the caption and describe more accurately the relationship
> between the fidelity metrics and downstream performance.

**Response.** We agree that the caption did not match the displayed correlations and
have corrected it. The caption now states that SSIM against the visible source
remains positive under all four evaluators, with a threefold range. The reported VIF
changes sign across evaluators, and entropy remains between -0.37 and +0.06.

The Results text also distinguishes the behaviour of each metric. It no longer says
that every fidelity metric reverses direction. The claim is that the sign and
ranking are evaluator-dependent, not that every individual metric has the same
pattern under every evaluator.

**Manuscript change.** Fig. 5 caption, p. 13; Sec. 6.

## Comment 2

> For LLVIP, some detectors use the same 300 images evaluated in the study as their
> validation set. Although the manuscript reports experiments with a partially
> revised data split, this protocol has not been applied to all major evaluators and
> comparisons. Please use results from a test set that is fully independent of the
> training and validation sets as the primary results.

**Response.** We agree and have rebuilt the primary detection comparison around a
fixed split with zero overlap. The split contains 1,350 training, 150 validation and
300 test images. All 300 test images are held out from training, validation,
checkpoint selection and calibration.

We trained or retained YOLO11n visible-only with four seeds, YOLO26n visible-only,
RT-DETR-l visible-only, and YOLO11n modality-agnostic with four seeds. Each
evaluator scores all 14 fused inputs and both sources. All 42 visible-trained
evaluator-input comparisons are negative, with deficits from 0.046 to 0.246.
Single-run YOLO26n and RT-DETR-l have separate 14-test BH corrections and resolve
all 14 inputs. The pooled, validation-overlapping audit is now labelled as a
sensitivity analysis and never described as the primary result.

**Manuscript change.** Sections 2 and 4; Table 1; Table 5; Figs. 2 and 4.

## Comment 3

> The authors note that the “reported VIF” and the reference implementation produce
> opposite rankings because of differences in noise-parameter calibration. However,
> Table 1 presents the comparison simply as “VIF.” Please clearly distinguish between
> the two implementations and report results recalculated with appropriately
> calibrated noise parameters using the reference implementation for the complete set
> of methods.

**Response.** We agree. The table now labels the two constructions separately as
reported-index VIF and reference pixel VIF. The Methods define both constructions,
including the scale widths, decimation rule and noise variance. Every fused input is
resocred with both constructions on LLVIP, and all eight methods are resocred on the
full TNO, MSRS-test and RoadScene sets.

The complete reference VIF calculation is in Table 2 and Table 11. SPGFusion ranks
first under the reported construction and last under the reference construction on
all three fidelity benchmarks. The same reversal appears across all 14 LLVIP fused
inputs.

**Manuscript change.** Section 3; Table 2; Table 11; Fig. 5; pp. 5 and 12 to 13.

## Comment 4

> The description of the detection evaluation procedure is insufficient. The
> manuscript states that one-to-one maximum-cardinality matching is used and that AP
> is computed using a detection library, but it does not clarify whether the matching
> is performed according to the confidence scores of the predicted boxes or how the
> matching results are used to calculate AP. Please provide these details.

**Response.** The evaluation protocol now states the complete procedure. For each
image, detections are sorted by confidence. A one-to-one maximum-cardinality
assignment matches predictions to ground-truth boxes at the specified IoU threshold.
Matched boxes above the confidence threshold are labelled true positives, unmatched
predictions are false positives, and unmatched ground-truth boxes are false
negatives. True-positive flags and confidence scores are then pooled over the 300
images to construct the precision-recall curve. AP uses the detector library's
101-point interpolation.

The manuscript names Ultralytics 8.4.189 and states the confidence threshold of
0.001. Interval construction and multiplicity correction are described separately.

**Manuscript change.** Section 2, evaluation protocol, pp. 3 to 4.
