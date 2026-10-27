# Clean-split detector checkpoints

These weights were trained for the fixed LLVIP 1,350/150/300 split used in the
Reviewer 2, comment 2 revision. Both runs use 40 epochs, 640-pixel inputs and seed 0.

| Model | Batch | Best validation epoch | Validation mAP50 | Validation mAP50-95 | SHA-256 |
|---|---:|---:|---:|---:|---|
| YOLO26n | 16 | 38 | 0.94402 | 0.60823 | `426ca4ef453b7333b1e3e53161b0b6410da9c26e790ec1ff9764ca490ebedb70` |
| RT-DETR-l | 4 | 38 | 0.94748 | 0.62448 | `9c02a3b610f91448697d2d418019c324d45aed6cb5e004f9dc145175df002f17` |

The complete recipes are in `results/clean_r2/yolo26n_clean_training.json` and
`results/clean_r2/rtdetr_l_clean_training.json`.
