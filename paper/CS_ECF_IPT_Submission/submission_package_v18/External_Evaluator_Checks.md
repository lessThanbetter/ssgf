# External evaluator stress tests

Date: 2026-10-28

## Purpose

This note records three evaluators that were absent from policy construction,
threshold selection, and the primary confirmatory result. Two are zero-shot
COCO models; the third is fine-tuned on the LLVIP training split to test a
domain-adapted failure mode.

## Evaluators

| Model | Initialization | LLVIP fine-tuning | SHA-256 |
|---|---|---|---|
| YOLOv9t | COCO-pretrained Ultralytics weights | none | `410E3CACA789AA8FEC19971C05577616CBB963A416BCD0228171B2DFF7F718F7` |
| YOLOv10s | COCO-pretrained Ultralytics weights | none | `65FA8332C38A7189597A268897BCE6C1026F9499711B9761FB0C2C639B91D4D6` |
| YOLOv9t, LLVIP-trained | COCO-pretrained Ultralytics weights | 1,350 visible training images | `AE0A44359C489DEEAE3F7443173588B9F68E264B6BE1EA455E961EB2DED0797D` |

Only person-class detections were retained. The frozen policy used the same
four candidate images, top-five confidence score, and 1.5 visible margin as the
primary experiment.

The LLVIP-trained checkpoint was produced for a pre-specified 40 epochs with
AdamW, initial learning rate 0.002, three-epoch warm-up, cosine decay, weight
decay `5e-4`, batch size 16, 640-pixel inputs, horizontal flipping, and
brightness/contrast jitter. The last checkpoint was frozen after epoch 40; no
epoch was selected using test results. It reached validation mAP50 `0.8456`.

## Candidate generation

The visible, PIAFusion, and SPGFusion images were taken from the clean LLVIP
test archive. Pixel-mean and CDF were regenerated from the released source
pairs with:

```powershell
python scripts/generate_pixel_mean.py `
  --sources D:\Code\_ssgf_external\r2_clean\test `
  --out D:\Code\_ssgf_external\r2_clean\test_candidates\pixel_mean

python scripts/generate_consensus_detail.py `
  --sources D:\Code\_ssgf_external\r2_clean\test `
  --candidates D:\Code\_ssgf_external\r2_clean\test_candidates `
  --out D:\Code\_ssgf_external\r2_clean\test_candidates `
  --threshold 0.95 --alpha 0.20 --sigma 5 `
  --policy-names consensus_intersection_pixel_mean_ifcnn
```

## Results

| Model | Visible | CDF | No shield | CS-ECF |
|---|---:|---:|---:|---:|
| YOLOv9t, zero-shot | 0.7698 | 0.7888 | 0.8216 | 0.7948 |
| YOLOv10s, zero-shot | 0.8344 | 0.8617 | 0.8699 | 0.8557 |
| YOLOv9t, LLVIP-trained | 0.6666 | 0.6598 | 0.6405 | 0.6558 |

CS-ECF minus visible:

- YOLOv9t: `+0.0250`, scene-group 95% CI `[+0.0053,+0.0523]`
- YOLOv10s: `+0.0213`, scene-group 95% CI `[+0.0041,+0.0439]`
- YOLOv9t, LLVIP-trained: `-0.0108`, scene-group 95% CI
  `[-0.0342,+0.0050]`

CS-ECF minus Always CDF:

- YOLOv9t: `+0.0060`, scene-group 95% CI `[-0.0277,+0.0374]`
- YOLOv10s: `-0.0059`, scene-group 95% CI `[-0.0266,+0.0180]`
- YOLOv9t, LLVIP-trained: `-0.0040`, scene-group 95% CI
  `[-0.0206,+0.0108]`

The two zero-shot evaluators support improvement over visible but not dominance
over Always CDF. The fine-tuned evaluator reverses the sign relative to visible.
The shield still improves the unshielded selector by `+0.0153` AP50, with a
scene-group CI `[+0.0072,+0.0267]`, and reduces the worst scene loss from
`0.1208` to `0.0724`. It does not eliminate the loss.

## Calibration and switching diagnostics

| Model | Visible/fused | ECE visible | ECE CS-ECF | Switched improved | Switched degraded | Positive/negative scenes |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv9t, zero-shot | 77.7%/22.3% | 0.0094 | 0.0091 | 46.3% | 20.9% | 5/2 |
| YOLOv10s, zero-shot | 81.7%/18.3% | 0.0103 | 0.0116 | 38.2% | 9.1% | 4/3 |
| YOLOv9t, LLVIP-trained | 76.7%/23.3% | 0.0475 | 0.0459 | 25.7% | 41.4% | 2/5 |

ECE is a descriptive detection-level calibration proxy with ten fixed
confidence bins. The fine-tuned model has a similar fused-selection fraction
but much worse calibration and a larger fraction of harmful switches.

The three external-evaluator contrasts form a separate exploratory family.
Benjamini-Hochberg q values from 100,000 group-bootstrap resamples are `0.043`
for zero-shot YOLOv9t, `0.043` for zero-shot YOLOv10s, and `0.119` for the
fine-tuned YOLOv9t. Holm-adjusted values are `0.057`, `0.050`, and `0.119`.

## Reproduction

The raw model outputs are in `per_image/`; fixed-candidate scores are in
`scores.json` and `multiplicity.json`; policy decisions and intervals are in
`policy_summary.json`. The LLVIP-trained result is in
`results/independent_evaluators/domain_yolov9t_v3/`. Regenerate the zero-shot
selection comparison with:

```powershell
python scripts/analyze_independent_evaluators.py
python analysis/cs_ecf/external_evaluator_diagnostics.py `
  --data-root D:\Code\_server_handoff_v2\ssgf `
  --out results/selective_fusion/external_evaluator_diagnostics.json
```
