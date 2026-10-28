# Reproducibility guide

The scripts in this directory reproduce the confirmatory statistics, baseline
controls, external-evaluator diagnostics, class summaries, and registration
sensitivity reported in the manuscript.

## Primary and external-evaluator inputs

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

## Major-revision analyses

The following commands write directly into the expected evidence directory.
Placeholders denote the released data/result roots and intermediate CSVs.

```text
python major_revision_statistics.py
  --out ../evidence/major_revision/major_revision_statistics.json

python image_quality_baselines.py
  --data-root DATA_ROOT
  --external-root EXTERNAL_ROOT
  --out ../evidence/major_revision/image_quality_baselines.json
  --features-out ../evidence/major_revision/image_quality_features.json

python external_evaluator_diagnostics.py
  --out ../evidence/major_revision/external_evaluator_diagnostics.json

python alignment_csecf_sensitivity.py
  --alignment-csv ALIGNMENT_CSV
  --out ../evidence/major_revision/alignment_csecf_sensitivity.json

python summarize_m3fd_classes.py
  --diagnostics DIAGNOSTICS
  --intervals INTERVALS
  --out ../evidence/major_revision/m3fd_class_summary.json
```

The primary script uses 2,000 grouped bootstrap draws per checkpoint, 500
uniform-random allocations, and 10,000 run-cluster-by-group aggregate draws.
`--bootstrap-draws` and `--random-draws` are available for bounded smoke tests;
reported values must be regenerated with the full settings.

`image_quality_features.json` and `major_revision_statistics_raw.json` retain
intermediate outputs used to audit table values. The scripts require the
released per-image policy scores, the fixed split/group manifests, and, for the
registration audit, LLVIP pair geometry. Dataset images and third-party model
weights are not redistributed.

## Final preflight

```text
python verify_cs_ecf_submission.py
```

It checks abstract and Highlights limits, placeholders, canonical numerical
claims, stale intervals and DOIs, the 46-entry reference count, 18/6/1 page
counts, LaTeX warnings, and figure presence.
