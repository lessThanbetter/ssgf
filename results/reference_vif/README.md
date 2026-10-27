# Full reference-VIF results

All scores use the visible source as the common spatial grid and cap its longest
side at 512 px. The infrared source and the fused output are resampled to that same
grid before VIF is computed. This convention is necessary for RoadScene, whose
crops have different native dimensions; taking a common crop would discard part of
the scene.

## Three fidelity benchmarks, eight inputs

Both columns are measured on the same generated fusion outputs and the same image
grid. The reported index uses widths `2**(s-1)`, no decimation, and noise variance
2.0 in `[0,1]`; the reference construction uses canonical widths, decimation by
two, and noise variance `2/255^2`.

| Dataset | Method | reported | reference |
|---|---|---:|---:|
| TNO (74) | mean | 0.2683 | 0.2226 |
| TNO | top-hat | 0.3560 | 0.2307 |
| TNO | probe | 0.3297 | 0.2309 |
| TNO | ensemble | 0.3431 | 0.2327 |
| TNO | IFCNN | 0.4440 | 0.2896 |
| TNO | DenseFuse | 0.3697 | 0.2729 |
| TNO | PIAFusion | 0.5485 | **0.3256** |
| TNO | SPGFusion | **0.7855** | 0.1607 |
| MSRS (361) | mean | 0.2914 | 0.2558 |
| MSRS | top-hat | 0.3914 | 0.2573 |
| MSRS | probe | 0.4015 | 0.2562 |
| MSRS | ensemble | 0.3955 | 0.2583 |
| MSRS | IFCNN | 0.3757 | 0.2517 |
| MSRS | DenseFuse | 0.3004 | 0.2722 |
| MSRS | PIAFusion | 0.7507 | **0.4468** |
| MSRS | SPGFusion | **0.8314** | 0.2308 |
| RoadScene (221) | mean | 0.3126 | 0.2661 |
| RoadScene | top-hat | 0.4240 | 0.2617 |
| RoadScene | probe | 0.4229 | 0.2602 |
| RoadScene | ensemble | 0.4234 | 0.2637 |
| RoadScene | IFCNN | 0.4988 | **0.3082** |
| RoadScene | DenseFuse | 0.3828 | 0.3052 |
| RoadScene | PIAFusion | 0.6315 | 0.2855 |
| RoadScene | SPGFusion | **0.8293** | 0.2166 |

SPGFusion is first under the reported index and last under the reference
construction on all three datasets. The result is therefore not a one-benchmark
accident.

## Full LLVIP table and downstream rank association

The full LLVIP table contains all 14 fused inputs and 300 images
(`llvip_14_full.csv`). The endpoints reverse: SPGFusion goes from 0.8115 under the
reported index to 0.1699 under the reference construction, while PIAFusion moves
from 0.7056 to 0.3719 and from second to first.

Spearman correlations between each index and the downstream mAP50 contrast are:

| Evaluator | reported | reference |
|---|---:|---:|
| YOLO11n, pooled | -0.367 | -0.037 |
| YOLO26n | +0.156 | +0.473 |
| RT-DETR-l | -0.178 | -0.024 |
| modality-agnostic, pooled | -0.086 | +0.626 |
| YOLO11n visible-only, clean, four-run mean | -0.134 | +0.182 |
| YOLO26n visible-only, clean | -0.108 | -0.209 |
| RT-DETR-l visible-only, clean | -0.002 | -0.200 |
| modality-agnostic, clean, four-run mean | +0.073 | +0.591 |

The reference construction recovers a positive association for the four-run
modality-agnostic evaluator but remains near zero or negative for every visible-trained
architecture. This qualifies the stronger claim that no fidelity score carries
downstream information: the specific reported construction does not transfer,
while its corrected counterpart partially does when the evaluator is
modality-agnostic. The clean-evaluator values are frozen in
`../clean_r2/clean_r2_correlations.json`.

## Reproduction

`analysis/run_reference_vif_parallel.py` accepts a job JSON and uses one recoverable
subprocess per variant. The frozen outputs used here are in the JSON/CSV files in
this directory; the source-pair extraction and alignment check are documented in
`../alignment/README.md`.
