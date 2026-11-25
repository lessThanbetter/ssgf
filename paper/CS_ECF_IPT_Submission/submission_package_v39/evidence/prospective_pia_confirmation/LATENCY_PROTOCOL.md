# End-to-end pipeline latency protocol

Recorded: 2026-11-26T01:03:45+08:00

This protocol was recorded before the end-to-end latency benchmark was run.

## Scope

- Device: NVIDIA GeForce RTX 2080 Ti, 11 GiB.
- Images: six decoded LLVIP pairs from sequence 080 at native
  \(1280\times1024\) resolution.
- Batch size: one image pair.
- All network weights are loaded before timing.
- JPEG decoding and file-system I/O are excluded.
- Tensor transfer, fusion, detector preprocessing and postprocessing, scoring,
  candidate selection, and synchronization are included.
- Each setting receives two warm-up executions followed by 12 timed
  executions.

## Compared pipelines

1. `visible`: one detector forward on visible input.
2. `budgeted`: one PIAFusion forward, detector forwards on visible and
   PIAFusion, top-five score comparison at margin 1.5, and selected-output
   return.
3. `full`: PIAFusion, SPGFusion at a maximum side of 512 pixels, and the
   frozen CDF construction, followed by detector forwards on visible, CDF,
   PIAFusion, and SPGFusion, top-five score comparison at margin 1.5, and
   selected-output return.

The full pipeline reproduces the four-candidate reference. The budgeted
pipeline reproduces the development-selected visible-plus-PIAFusion policy.

## Checkpoints

The benchmark uses the same six deterministic final detector checkpoints as
the confirmation analysis:

- YOLO26n seeds 0, 1, and 2;
- YOLOv8n seeds 0 and 1;
- RT-DETR-l.

For each checkpoint, the report includes the median and 95th percentile of
the complete wall-clock pipeline time and medians for the fusion and detector
components.

## Reporting

- Report every checkpoint separately and the equal-checkpoint median.
- Report observed p95 latency as well as the median.
- Keep the result as a single-GPU steady-state benchmark.
- Do not convert it into a general real-time guarantee or a multi-GPU
  deployment claim.
