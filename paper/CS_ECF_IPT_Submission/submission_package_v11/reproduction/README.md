# External evaluator reproduction

The scripts in this directory reproduce the three external evaluator checks
reported in the manuscript.

## Inputs

- A clean LLVIP archive with 1,350 training pairs and 300 test pairs.
- COCO-pretrained `yolov9t.pt` and `yolov10s.pt`.
- The four test candidates: visible, CDF, PIAFusion, and SPGFusion.
- The fixed 300-image test manifest used by the primary experiment.

## LLVIP-fine-tuned YOLOv9t

The final checkpoint was trained for a pre-specified 40 epochs:

```text
python train_independent_manual.py
  --weights yolov9t.pt
  --data data.yaml
  --project runs_independent
  --name yolov9t_clean_manual40_v3
  --imgsz 640 --batch 16 --workers 0 --epochs 40 --seed 0 --lr 0.002
```

The expected checkpoint SHA-256 is:

```text
AE0A44359C489DEEAE3F7443173588B9F68E264B6BE1EA455E961EB2DED0797D
```

## Scoring and policy analysis

Score all four candidate inputs with `score_clean_evaluators.py`, retain
person-class outputs, and use the fixed 300-image stem manifest. Then run:

```text
python analyze_independent_evaluators.py
```

This command regenerates the policy decisions and scene-group bootstrap
intervals for the two zero-shot evaluators and the LLVIP-fine-tuned boundary
case. Model-specific bootstrap seeds make each result independent of the model
iteration order.

The final preflight check is:

```text
python verify_cs_ecf_submission.py
```

It checks abstract and Highlights limits, placeholders, canonical numerical
claims, stale intervals, reference count, PDF page counts, LaTeX warnings, and
figure presence.
