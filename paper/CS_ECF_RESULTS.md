# CS-ECF Results

> Generated from `confidence_shield*.json`; do not edit tables by hand.

## Frozen policy

- Top-k confidence: 5
- Visible margin: 1.5
- Confirmatory datasets: LLVIP test, M3FD grouped test, DroneVehicle official test subset
- DroneVehicle checkpoints: best and last from the same training run

## Per-evaluator test results

| Dataset | Evaluator | Visible AP50 | CS-ECF AP50 | Gain | Image CI95 |
|---|---|---:|---:|---:|---:|
| LLVIP | yolo11 | 0.8811 | 0.8845 | +0.0034 | `[-0.0026,+0.0107]` |
| LLVIP | agnostic | 0.8550 | 0.8611 | +0.0061 | `[+0.0011,+0.0148]` |
| LLVIP | yolo26_s0 | 0.8122 | 0.8088 | -0.0034 | `[-0.0091,+0.0030]` |
| LLVIP | rtdetr | 0.8095 | 0.8136 | +0.0040 | `[+0.0001,+0.0079]` |
| LLVIP | yolo26_s1 | 0.7937 | 0.7911 | -0.0027 | `[-0.0110,+0.0082]` |
| LLVIP | yolo26_s2 | 0.8061 | 0.8087 | +0.0026 | `[-0.0028,+0.0086]` |
| LLVIP | yolo8_s0 | 0.7801 | 0.7761 | -0.0040 | `[-0.0089,+0.0021]` |
| LLVIP | yolo8_s1 | 0.8055 | 0.8119 | +0.0064 | `[+0.0011,+0.0130]` |
| M3FD | yolo26_m3fd6 | 0.3717 | 0.3757 | +0.0041 | `[-0.0008,+0.0080]` |
| M3FD | yolo8n_m3fd6 | 0.3790 | 0.3972 | +0.0182 | `[+0.0068,+0.0310]` |
| DRONEVEHICLE | best | 0.6563 | 0.6814 | +0.0251 | - |
| DRONEVEHICLE | last | 0.6474 | 0.6767 | +0.0293 | - |

## Profile and dataset-balanced summaries

| Scope | Models | Positive | Mean | Worst | Model CI95 | Image CI95 | Group/Hierarchical CI95 |
|---|---:|---:|---:|---:|---|---|---|
| LLVIP | 8 | 5/8 | +0.0016 | -0.0040 | `[-0.0014,+0.0042]` | `[-0.0010,+0.0051]` | `[-0.0008,+0.0052]` |
| M3FD | 2 | 2/2 | +0.0111 | +0.0041 | `[+0.0041,+0.0182]` | `[+0.0036,+0.0187]` | `[+0.0000,+0.0157]` |
| DRONEVEHICLE_TEST | 2 | 2/2 | +0.0272 | +0.0251 | `[+0.0251,+0.0293]` | `[+0.0002,+0.0611]` | `[+0.0002,+0.0406]` |
| Dataset-balanced | 12 | 9/12 | +0.0133 | -0.0040 | `[+0.0101,+0.0167]` | `[+0.0032,+0.0248]` | `[+0.0028,+0.0213]` |

## DroneVehicle independent test

| Checkpoint | Visible AP50 | CS-ECF AP50 | Shield gain | No-shield gain | Selected images |
|---|---:|---:|---:|---:|---:|
| best | 0.6563 | 0.6814 | +0.0251 | +0.1064 | 59/300 |
| last | 0.6474 | 0.6767 | +0.0293 | +0.1016 | 67/300 |
| Combined | - | - | +0.0272 | +0.1040 | - |

- Image bootstrap 95% CI: `[-0.0028,+0.0682]`
- Group bootstrap 95% CI: `[+0.0002,+0.0406]`


## Policy comparison

| Policy | Positive checkpoints | Mean gain | Dataset-balanced | Worst |
|---|---:|---:|---:|---:|
| Always CDF | 10/12 | +0.0052 | +0.0031 | -0.0059 |
| Always PIAFusion | 3/12 | -0.1087 | -0.0710 | -0.2910 |
| Always SPGFusion | 2/12 | -0.1136 | -0.0602 | -0.2501 |
| Top-5 no shield | 7/12 | +0.0132 | +0.0332 | -0.0584 |
| CS-ECF | 9/12 | +0.0074 | +0.0133 | -0.0040 |

## Shield ablation

| Scope | No-shield positive | No-shield mean | No-shield worst | Shield positive | Shield mean | Shield worst |
|---|---:|---:|---:|---:|---:|---:|
| LLVIP | 4/8 | -0.0069 | -0.0584 | 5/8 | +0.0016 | -0.0040 |
| M3FD | 1/2 | +0.0026 | -0.0057 | 2/2 | +0.0111 | +0.0040 |
| DRONEVEHICLE_TEST | 2/2 | +0.1040 | +0.1016 | 2/2 | +0.0272 | +0.0251 |
| Overall | 7/12 | +0.0132 | -0.0584 | 9/12 | +0.0074 | -0.0040 |

## M3FD class effects

| Evaluator | Class | Visible AP50 | CS-ECF AP50 | Gain | Image CI95 |
|---|---|---:|---:|---:|---:|
| yolo26_m3fd6 | people | 0.5120 | 0.5334 | +0.0214 | `[-0.0070,+0.0498]` |
| yolo26_m3fd6 | car | 0.6214 | 0.6214 | +0.0000 | `[+0.0000,+0.0000]` |
| yolo26_m3fd6 | bus | 0.3399 | 0.3399 | +0.0000 | `[+0.0000,+0.0000]` |
| yolo26_m3fd6 | motorcycle | 0.4780 | 0.4780 | +0.0000 | `[+0.0000,+0.0000]` |
| yolo26_m3fd6 | lamp | 0.2365 | 0.2395 | +0.0030 | `[+0.0000,+0.0102]` |
| yolo26_m3fd6 | truck | 0.0423 | 0.0423 | +0.0000 | `[+0.0000,+0.0000]` |
| yolo8n_m3fd6 | people | 0.4364 | 0.5454 | +0.1090 | `[+0.0440,+0.1913]` |
| yolo8n_m3fd6 | car | 0.5637 | 0.5636 | -0.0001 | `[-0.0003,+0.0000]` |
| yolo8n_m3fd6 | bus | 0.6380 | 0.6380 | +0.0000 | `[+0.0000,+0.0000]` |
| yolo8n_m3fd6 | motorcycle | 0.4355 | 0.4355 | +0.0000 | `[-0.0001,+0.0000]` |
| yolo8n_m3fd6 | lamp | 0.1549 | 0.1550 | +0.0001 | `[+0.0000,+0.0002]` |
| yolo8n_m3fd6 | truck | 0.0456 | 0.0459 | +0.0003 | `[+0.0000,+0.0010]` |

## Thermal association with switching

| Dataset | Thermal rho (p) | Visible-luma rho (p) | CDF-mask rho (p) |
|---|---:|---:|---:|
| llvip | +0.1505 (0.0090) | -0.1044 (0.0711) | -0.0293 (0.6135) |
| m3fd | +0.3198 (0.0000) | -0.0656 (0.3561) | -0.3315 (0.0000) |
| dronevehicle_test | -0.4027 (0.0000) | -0.4413 (0.0000) | +0.0640 (0.2689) |

## Thermal-mask ablation on DroneVehicle

| Checkpoint | Visible | Masked CDF | Global CDF | Mask minus global |
|---|---:|---:|---:|---:|
| best | 0.6563 | 0.6624 | 0.6620 | +0.0003 |
| last | 0.6474 | 0.6524 | 0.6512 | +0.0012 |

## Rejected validation-selected variants

| Variant | Candidates | Score | Margin | Test positive | Test mean | Dataset-balanced | Hierarchical CI95 |
|---|---|---|---:|---:|---:|---:|---|
| CS-ECF-v2 | vis, cdf, pia | top10_mean_detection_confidence | 1.30 | 5/10 | +0.0021 | +0.0039 | `[-0.0029,+0.0127]` |
| CS-ECF-v3 | vis, spg | mean_detection_confidence | 1.50 | 6/10 | -0.0039 | -0.0034 | `[-0.0085,+0.0013]` |

## Claim boundary

- LLVIP intervals cross zero; do not describe the overall method as uniformly robust.
- M3FD gain is concentrated in `people`; do not claim uniform improvement across six classes.
- Always CDF remains a strong baseline (10/12 positive).
- DroneVehicle image-level intervals cross zero; its three fixed groups are non-negative, but only two are materially positive.
- Group-aware dataset-balanced CI is positive, but M3FD contributes only two groups and LLVIP seven scenes.
- The two DroneVehicle checkpoints come from the same training run; they are not independent architectures.
