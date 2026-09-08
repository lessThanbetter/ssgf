# SOTA baseline results (uniform protocol)

> **Note on numbers.** The tables below were produced by an **early training run**
> of the backbone/probe. They are kept in this repository for transparency, but the
> **official numbers of the submitted manuscript** are those of the stabilized
> seed-2026 model (`data/msrs_ae.pt` + `data/msrs_probe.pt`, 30 epochs, `lr=1e-3`),
> which are reproduced by `sota_runners/recompute_tabfusion.py`,
> `sota_runners/recompute_quant_table.py`, `sota_runners/det_eval_fused.py` and
> `sota_runners/seg_eval_fused.py`. If any number here differs from the manuscript
> tables, the manuscript (stable seed-2026) is authoritative. Run the scripts above
> with the official checkpoints to regenerate a single consistent table.

All fused at `max_side=512` (rounded to a multiple of 16 for model
compatibility). Metrics via `ssgf/utils/metrics.py`; sources resized to the fused
size before comparison. Detection = fixed YOLO11n on LLVIP test (300 images,
`imgsz=640`); segmentation = fixed U-Net (MSRS-train) on MSRS-test (361 images).

## Fusion VIF (mean over dataset)

| Dataset | IFCNN | DenseFuse | CS-saliency | Ours (probe) | PIAFusion | SPGFusion |
|---|---|---|---|---|---|---|
| TNO (74) | 0.444 | 0.370 | 0.289 | 0.372 | 0.625 | **0.820** |
| MSRS (361) | 0.376 | 0.301 | 0.310 | 0.416 | 0.759 | **0.834** |
| RoadScene (221) | 0.458 | 0.365 | 0.323 | 0.422 | 0.630 | **0.830** |

Ours (top-hat) = 0.364 / 0.400 / 0.418 on the three sets.

## Downstream detection mAP50 (LLVIP test)

### Visible-trained detector (YOLO11n, `data/det_runs/llvip/weights/best.pt`)

| Input | mAP50 |
|---|---|
| visible | 0.816 |
| infrared | 0.697 |
| mean fusion | 0.828 |
| top-hat | 0.837 |
| **Ours (probe)** | **0.847** |
| ensemble | 0.842 |
| PIAFusion | 0.742 |
| SPGFusion | 0.736 |

### Modality-agnostic detector (`runs/detect/data/det_agnostic/agn/weights/best.pt`)

| Input | mAP50 |
|---|---|
| visible | 0.774 |
| infrared | 0.948 |
| **Ours (probe)** | **0.893** |
| PIAFusion | 0.899 |
| SPGFusion | 0.884 |

## Downstream segmentation mIoU (MSRS-test, U-Net, 8 classes)

| Input / method | mIoU |
|---|---|
| visible | 0.130 |
| infrared | 0.134 |
| mean | 0.124 |
| top-hat | 0.127 |
| probe | 0.127 |
| PIAFusion | 0.130 |
| SPGFusion | 0.133 |

## Takeaway

SPGFusion / PIAFusion dominate on pixel-level VIF but their high-fidelity fusion
does **not** transfer to downstream detection: for a visible-trained detector they
are *worse* than the raw visible source (0.736 / 0.742 vs 0.816) and clearly worse
than the proposed targetness guide (0.847). Under a modality-agnostic detector they
are roughly on par with the proposed method (0.899 / 0.884 vs 0.893) and all are
below the raw infrared source (0.948). Segmentation is saturated at ~0.13 for every
method. => Image-fidelity (VIF) is a poor proxy for target-centric downstream
performance, which is the core claim of the paper.

## Environment notes

- SPGFusion: `open-clip-torch==2.24.0`, `mmseg` shim (only `resize` used), DINO
  `dino_vitbase16_pretrain.pth` (327 MB) at `~/.cache/torch/hub/checkpoints/`.
- PIAFusion: pure PyTorch; weights bundled in repo (`pretrained/*.pth`).
