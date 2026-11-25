# Reproducibility guide

The scripts in this directory reproduce the retrospective and held-out
statistics, confidence-ordered matching sensitivity, switch-budget-matched
control, external-evaluator diagnostics, native DroneVehicle OBB scoring,
class summaries, registration sensitivity, and the recent YOLOv11-RGBT
implementation pilot.

## Primary and external-evaluator inputs

- The official LLVIP archive and its annotation files. The experiment uses a
  fixed 300-pair test subset (uniform shuffle, seed 0, non-empty labels) and a
  1,500-pair training subset (lexicographically first non-empty labels), split
  into 1,350 training and 150 validation pairs.
- COCO-pretrained `yolov9t.pt` and `yolov10s.pt`.
- The four test candidates: visible, CDF, PIAFusion, and SPGFusion.
- The fixed 300-image test manifest used by the primary experiment.

## LLVIP-trained YOLOv9t

The final checkpoint was trained for a pre-specified 40 epochs:

```text
python train_independent_manual.py `
  --weights yolov9t.pt `
  --data data.yaml `
  --project runs_independent `
  --name yolov9t_clean_manual40_v3 `
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
intervals for the two zero-shot evaluators and the LLVIP-trained boundary
case. Model-specific bootstrap seeds make each result independent of the model
iteration order.

## Analysis scripts

The following commands write directly into the expected evidence directory.
Placeholders denote the released data/result roots and intermediate CSVs.

```text
python analyze_csecf_vs_cdf.py

python equal_group_statistics.py `
  --out ../evidence/major_revision/equal_group_statistics.json

python major_revision_statistics.py `
  --out ../evidence/major_revision/major_revision_statistics.json

python image_quality_baselines.py `
  --data-root DATA_ROOT `
  --external-root EXTERNAL_ROOT `
  --out ../evidence/major_revision/image_quality_baselines.json `
  --features-out ../evidence/major_revision/image_quality_features.json

python external_evaluator_diagnostics.py `
  --out ../evidence/major_revision/external_evaluator_diagnostics.json

python alignment_csecf_sensitivity.py `
  --alignment-csv ALIGNMENT_CSV `
  --out ../evidence/major_revision/alignment_csecf_sensitivity.json

python summarize_m3fd_classes.py `
  --diagnostics DIAGNOSTICS `
  --intervals INTERVALS `
  --out ../evidence/major_revision/m3fd_class_summary.json
```

`analyze_csecf_vs_cdf.py` reconstructs the primary policy and the matched
visible-or-CDF gate from the released per-image arrays. Its pooled-image
quantities are descriptive. `equal_group_statistics.py` is the canonical
source for the main paired effect: groups are weighted equally within
checkpoint, checkpoints are averaged within training-run cluster, and datasets
are weighted equally. `merge_equal_group_evidence.py` maps the equal-group
point and interval into the canonical evidence files while retaining the
historical pooled values under explicit `pooled_` keys.

The equal-group script uses 2,000 grouped resampling draws per checkpoint,
5,000 aggregate draws, and 20 independent 5,000-draw stability reruns.

`image_quality_features.json` and `major_revision_statistics_raw.json` retain
intermediate outputs used to audit table values. The scripts require the
released per-image policy scores, the fixed split/group manifests, and, for the
registration audit, LLVIP pair geometry. Dataset images and third-party model
weights are not redistributed.

`prefreeze_validation_summary.json` records the complete original development
margin search. The later audit did not reselect margin 1.5.
`llvip_test_sampling_summary.json` and
`llvip_test_manifest_3463.json` document the 3,463-pair non-empty official test
frame and the 300-pair seed-0 subset.

## Prospective LLVIP and DroneVehicle

The prospective score arrays for LLVIP sequences 040--070 are included under
`../evidence/prospective/`. The analysis commands are:

```text
python prospective_llvip_analysis.py `
  --root ../evidence/prospective `
  --data-root "$env:SSGF_DATA_ROOT" `
  --out ../evidence/prospective/llvip_primary_analysis.json

python analyze_llvip_original_checkpoint_confirmation.py `
  --root ../evidence/prospective `
  --data-root "$env:SSGF_DATA_ROOT" `
  --out ../evidence/prospective/llvip_original8_analysis.json

python assemble_prospective_evidence.py
```

The manuscript headline LLVIP result uses a second, standard
confidence-ordered matching analysis:

```text
python rescore_prospective_standard_matching.py --batch 8 --device 0
python analyze_standard_matching_sensitivity.py
python analyze_matched_switch_control.py `
  --evidence-root ../evidence/prospective_standard_matching `
  --out ../evidence/prospective_standard_matching/matched_switch_target_control.json
```

The original all-at-once maximum-cardinality arrays remain as a sensitivity
analysis under `../evidence/prospective/`.

Native DroneVehicle scoring and independent training use:

```text
python prepare_dronevehicle_obb_runs.py
python train_dronevehicle_obb_independent.py
python score_dronevehicle_native_obb.py
python analyze_dronevehicle_native_obb.py
```

Required data and weight paths are command-line arguments in those scripts.
The package includes the resulting per-image score arrays, run manifests,
canonical evidence, and protocol hashes.

## VEDAI external domain

VEDAI preparation, independent training, native OBB scoring, analysis, and
canonical assembly use `prepare_vedai_obb.py`, `train_vedai_obb.py`,
`score_vedai_native_obb.py`, `analyze_vedai_native_obb.py`, and
`assemble_vedai_evidence.py`. The manuscript table is exported with
`export_vedai_table.py`.

`rescore_vedai_standard_matching.py` recomputes the true-positive flags with
confidence-ordered matching. The resulting analysis is under
`../evidence/vedai/standard_matching/` and is the source of the main VEDAI
table.

## Recent joint fusion-detection baseline

The YOLOv11-RGBT run and its input-stem initialization are implemented in
`train_yolo11rgbt_m3fd.py`. Apply
`yolo11rgbt_baseline/ultralytics_data_augment_4ch.patch` to the upstream commit
before training. Exact arguments and evidence paths are documented in
`yolo11rgbt_baseline/README.md`.

Re-score the exported test predictions with the study's AP protocol:

```text
python evaluate_joint_predictions.py `
  --predictions PATH_TO_RUN_TEST/predictions.json `
  --annotations PATH_TO_M3FD/annotations/test.json `
  --groups-manifest ../evidence/major_revision/m3fd_primary_group_manifest.json `
  --output ../evidence/yolo11rgbt_baseline/metrics.json
```

The main-table source CSV is generated by `export_joint_baseline_table.py`.

## Reference field audit

The reference audit checks DOI resolution and normalized Crossref title,
author count, year, volume, and first-page metadata:

```text
python verify_references_crossref.py ../references.bib --jobs 6 --fail-on-issues
```

## Figure regeneration

`inputs/` contains the released aggregate files needed for figure generation.
Run the complete figure build from this directory with:

```text
python build_figures.py
```

The driver writes final-numbered PDFs to `../figures` and CSV files to
`../source_data`. `FIGURE_BUILD_MAP.json` records the mapping from the
historical generator filenames to the final manuscript order. The shared
generator can also export SVG, 600 dpi TIFF, and PNG by setting
`SSGF_FIGURE_SUFFIXES`; the submission package stores PDF figures.

## Final preflight

```text
python build_manifest.py
python verify_cs_ecf_submission.py
python build_submission_archive.py
```

It checks abstract and Highlights limits, placeholders, LLVIP provenance,
CDF parameter definitions, equal-group point and interval alignment,
standard-matching estimates and intervals, protocol hashes, score-array counts,
canonical evidence, full split manifests, VEDAI protocol and detector hashes,
stale values and DOIs, the 48-entry reference count, 19/19/1 page counts,
LaTeX warnings, all 13 figure and source-data pairs, and the 5/8
main/supplement figure declarations. The PDF-text gate verifies references
[1]--[48] and main caption continuity. The archive builder includes only the
manifest-listed payload and writes a sidecar SHA-256 digest beside the ZIP.

## Runtime

The figure build was verified with Python 3.14.7, NumPy 2.5.3, and Matplotlib
3.11.2. Full experiment scoring additionally requires the dependencies listed
in the public repository. Raw datasets and third-party model weights are not
redistributed.
