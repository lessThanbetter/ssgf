# Domain-fine-tuned YOLOv9t check

Date: 2026-10-28

## Checkpoint

- Weights:
  `D:\Code\_ssgf_external\r2_clean\runs_independent\yolov9t_clean_manual40_v3\weights\last.pt`
- SHA-256:
  `AE0A44359C489DEEAE3F7443173588B9F68E264B6BE1EA455E961EB2DED0797D`
- Training protocol: 40 epochs, 1,350 visible-image training pairs, AdamW,
  batch 16, image size 640, seed 0, PIL loader/manual Ultralytics loss loop.
- Training loss: 64.69 at epoch 1; 29.94 at epoch 40.
- Validation mAP50: 0.8456. Validation mAP50-95: 0.4706.

The checkpoint was frozen after the pre-specified 40 epochs. No epoch was
selected using test results.

## Test results

The frozen policy and the checkpoint were evaluated on the same 300 LLVIP test
pairs used by the independent zero-shot checks.

| Candidate or policy | AP50 | AP50-95 |
|---|---:|---:|
| Visible | 0.6666 | 0.3019 |
| Always CDF | 0.6598 | 0.3023 |
| No-shield | 0.6405 | - |
| CS-ECF | 0.6558 | - |

Paired differences:

- CS-ECF minus visible: `-0.0108`; image CI
  `[-0.0267,+0.0046]`; scene-group CI `[-0.0342,+0.0050]`.
- CS-ECF minus Always CDF: `-0.0040`; image CI
  `[-0.0303,+0.0220]`; scene-group CI `[-0.0206,+0.0108]`.
- CS-ECF minus no-shield: `+0.0153`; scene-group CI `[+0.0072,+0.0267]`.

The shield selected visible for 230 of 300 images and selected CDF, PIAFusion,
and SPGFusion for 11, 26, and 33 images, respectively. Scene 240 was the worst
case: CS-ECF lost 0.0724 AP50 relative to visible, while no-shield lost 0.1208.

## Claim boundary

This fine-tuned evaluator does not reproduce the visible-baseline gain found for
the two COCO-pretrained zero-shot evaluators. It therefore blocks a claim that
CS-ECF is universally portable across unused detector checkpoints. It does
still show that the confidence shield reduces the loss of the unshielded
selector, but it does not reduce that loss to zero.

The 0.8456 validation mAP50 and 0.6666 test-set visible AP50 indicate a large
scene-shift effect. The negative result is retained as a boundary case rather
than treated as a scoring failure.
