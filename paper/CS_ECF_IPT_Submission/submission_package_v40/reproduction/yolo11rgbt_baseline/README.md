# YOLOv11-RGBT controlled M3FD baseline

Source repository: `https://github.com/wandahangFY/YOLOv11-RGBT`

Source commit: `9cc2e208a3d3e8452e15b5ef47e1c536766aa1f7`

The controlled baseline is the early-fusion YOLOv11-RGBT model. It receives a
four-channel image ordered as visible BGR plus one grayscale infrared channel.
The fusion and detection paths are trained together.

The local run used the upstream four-channel dataset loader and added the patch
in `ultralytics_data_augment_4ch.patch`. The patch keeps four-channel affine
augmentation and box transforms on the current Windows stack. It does not alter
the network, loss, class definition, split, or inference precision.

Training initialization uses the public `yolo11n.pt` COCO checkpoint. The
three-channel stem is copied into the first three channels of the four-channel
stem, and the infrared channel is initialized to the channel mean. The
six-class detection head is reinitialized for M3FD.

The final run used 40 epochs, batch size 16, 640-pixel inputs, a fixed seed,
AdamW selected by the upstream `auto` optimizer logic, and no mosaic, mixup, or
photometric augmentation. The validation checkpoint was evaluated once on the
fixed 200-image M3FD test split.

Relevant commands:

```text
python train_yolo11rgbt_m3fd.py \
  --repo PATH_TO_YOLO11_RGBT \
  --model PATH_TO_YOLO11_RGBT/ultralytics/cfg/models/11-RGBT/yolo11-RGBT-earlyfusion.yaml \
  --initial-weights yolo11n.pt \
  --data m3fd.yaml \
  --project ../evidence/yolo11rgbt_baseline/runs \
  --name yolo11n_rgbt_earlyfusion_e40_resize_auto \
  --epochs 40 --batch 16 --workers 4 --device 0 \
  --mosaic 0 --optimizer auto --no-augment
```

The training script writes the test `predictions.json`. Re-score those
predictions with the study's maximum-cardinality matching and 101-point AP
protocol:

```text
python evaluate_joint_predictions.py \
  --predictions PATH_TO_RUN_TEST/predictions.json \
  --annotations PATH_TO_M3FD/annotations/test.json \
  --groups-manifest ../evidence/major_revision/m3fd_primary_group_manifest.json \
  --output ../evidence/yolo11rgbt_baseline/metrics.json
```

Export the main-table source values with:

```text
python export_joint_baseline_table.py \
  --joint ../evidence/yolo11rgbt_baseline/metrics.json \
  --statistics ../evidence/major_revision/major_revision_statistics.json \
  --out ../source_data/TableJointBaseline_source.csv
```

Exact run arguments, metrics, checkpoint hashes, and held-out predictions are
stored under `evidence/yolo11rgbt_baseline/`.
