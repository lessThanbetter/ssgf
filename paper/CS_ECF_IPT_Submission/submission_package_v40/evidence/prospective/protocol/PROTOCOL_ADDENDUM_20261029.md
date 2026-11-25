# Pre-confirmation sensitivity amendment

Recorded: 2026-10-29T01:10:00+08:00

This addendum was recorded before calculating any CS-ECF policy outcome on
LLVIP sequences 050, 060, or 070.

The primary confirmation remains the preregistered six-run analysis based only
on deterministic `last.pt` checkpoints.

As a prespecified secondary sensitivity analysis, evaluate the same unchanged
margin 1.5 on the eight original LLVIP comparison checkpoints distributed with
the submission:

- YOLO11n visible-trained
- modality-agnostic YOLO11n
- YOLO26n seeds 0, 1, and 2
- RT-DETR-l
- YOLOv8n seeds 0 and 1

The original checkpoint packages used finalized archived weights rather than
newly selected replacements. They are tested on sequences 050, 060, and 070
without any threshold change. Results from this secondary set are not allowed
to replace a negative primary result.

The analysis will report each checkpoint separately, the equal-sequence mean,
the worst checkpoint, the image-level win rate, and a deterministic
bootstrap interval over sequences, checkpoints, and within-sequence blocks.
