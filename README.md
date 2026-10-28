# SSGF / CS-ECF Research Code

This repository hosts the code and frozen numerical results for two related
visible--infrared studies.

## 1. Evaluator-conditioned fusion input arbitration (CS-ECF)

The current release accompanies the working manuscript
**“Confidence-Shielded Evaluator-Conditioned Fusion for Risk-Controlled
Infrared--Visible Object Detection.”**

The version-specific archival release is **v2.4.1**, DOI
[`10.5281/zenodo.23728385`](https://doi.org/10.5281/zenodo.23728385). The
shared software-series concept DOI is
[`10.5281/zenodo.22984388`](https://doi.org/10.5281/zenodo.22984388).

CS-ECF evaluates visible, CDF, PIAFusion, and SPGFusion candidates with the
target detector. A fused candidate is selected only when its top-five mean
confidence reaches 1.5 times the visible-image score; otherwise the visible
input is retained. The policy uses no test labels.

Key frozen results:

- 12 confirmatory detector checkpoints; 9 positive;
- dataset-balanced AP50 gain `+0.0133`;
- training-run-by-group hierarchical 95% CI `[+0.0026, +0.0261]`;
- direct dataset-balanced difference from Always CDF `+0.0102`, with a
  training-run-by-group interval `[-0.00057, +0.02980]` that includes zero;
- two unused COCO-pretrained evaluators improved over visible by `+0.021` to
  `+0.025` AP50, while their differences from Always CDF still crossed zero;
- an LLVIP-fine-tuned YOLOv9t exposed a negative portability boundary:
  CS-ECF was `-0.0108` AP50 relative to visible, while the shield improved the
  unshielded selector by `+0.0153`;
- no-shield worst loss `-0.0584`, CS-ECF worst loss `-0.0040`;
- DroneVehicle independent-test gains `+0.0251 / +0.0293`;
- thermal-mask advantage over global injection `+0.0007 AP50`;
- standardized serial cost estimate `706.3 ms`.

Artifacts:

- `ssgf/fusion/confidence_shield.py` — frozen selector;
- `scripts/generate_consensus_detail.py` — CDF candidate generation;
- `scripts/make_ipt_fig*.py` — publication figures;
- `scripts/analyze_csecf_vs_cdf.py` — reconstruction and paired bootstrap for
  the strongest fixed CDF baseline;
- `scripts/analyze_independent_evaluators.py` — frozen-policy check on unused
  YOLOv9t and YOLOv10s evaluators plus the LLVIP-fine-tuned boundary case;
- `scripts/train_independent_manual.py` — reproducible PIL-loader training of
  the LLVIP-fine-tuned YOLOv9t evaluator;
- `scripts/verify_cs_ecf_submission.py` — abstract, claim, PDF, reference, and
  placeholder preflight checks;
- `analysis/cs_ecf/` — analysis and interval-reproduction scripts;
- `results/selective_fusion/` — aggregate JSON, source-data CSV and per-image
  compressed score files;
- `paper/CS_ECF_IPT_Submission/` — Elsevier LaTeX source, compiled 25-page PDF,
  cover letter, 11 numbered figures, 11 figure-level source-data files, and the
  versioned v18 submission package;
- `paper/CS_ECF_IPT_DRAFT.md`, `paper/CS_ECF_RESULTS.md`,
  `paper/CS_ECF_2ZONE_MATERIALS.md` — manuscript and evidence notes.

The repository does not contain LLVIP, M3FD, or DroneVehicle images. LLVIP is
distributed at <https://github.com/bupt-ai-cz/LLVIP>; M3FD with the TarDAL
benchmark at <https://github.com/JinyuanLiu-CV/TarDAL>; DroneVehicle at
<https://github.com/VisDrone/DroneVehicle>. The 300-pair DroneVehicle test
manifest and source-archive SHA-256 are included at
`results/selective_fusion/dronevehicle_test_manifest.json`.

## 2. Evaluator-dependent fidelity-metric validity

The preceding part of the repository is the official code for
“Evaluator-Dependent Validity of Fidelity Metrics for Visible-Infrared Image
Fusion in Pedestrian Detection.”

The paper is a measurement-validity study of image-fidelity scores used to select
visible-infrared fusion rules. It contains a target-aware fusion family based on a
morphological top-hat prior and a distilled probe, but that family is one of the
fourteen evaluated inputs rather than the paper's claim of a superior fused image.

## Datasets

Publicly available (download manually and point the scripts to the paths):
- **TNO** (74 registered pairs): https://doi.org/10.6084/m9.figshare.1008029
- **MSRS** (1083 train / 361 test): https://github.com/Linfeng-Tang/Image-Fusion
- **RoadScene** (221 test pairs): https://github.com/hanna-xu/RoadScene
- **LLVIP** (detection subset): https://github.com/bupt-ai-cz/LLVIP

Run `bash setup_datasets.sh ./datasets` to download the smaller archives and create the
paired `vis`/`ir` folders the scripts expect. For the large MSRS/LLVIP archives, download
them from the links above (they are too large for a Git repository), then run the
`prepare_data.py` exemplar shown in `setup_datasets.sh`.

## Install

```bash
conda create -n ssgf python=3.10 -y
conda activate ssgf
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

For the complete reference-VIF and SPGFusion reproduction, install
`requirements-experiments.txt` after matching `torchvision` to the server's PyTorch
build.

## Repository layout

- `ssgf/`  package (config, data, fusion rules, models, metrics, checkpoints utils)
- `evaluate.py`  cross-dataset VIF / EN / SSIM / PSNR evaluation
- `train_fusion_e2e.py`  end-to-end fusion-aware training (stronger backbone option)
- `run_baselines.py`  train / evaluate IFCNN and DenseFuse on MSRS
- `train_classifier.py`  train the CSF classification-saliency baseline classifier
- `seg_downstream.py`  downstream semantic segmentation (UNet mIoU)
- `det_eval.py`  downstream detection (LLVIP YOLO) mAP evaluation
- `analysis/`  the scripts that produced the detection tables, the paired intervals
  the VIF schedule and full-set comparisons, and the alignment sensitivity; see
  `analysis/README.md` for the mapping
- `sota_runners/`  helpers to run PIAFusion / SPGFusion and reproduce every table/figure
- `results/reference_vif/`  frozen full-set reported/reference VIF tables
- `results/alignment/`  LLVIP registration estimates and best/worst-quartile results
- `data/`  trained checkpoints (small)

## Reproduce the paper numbers

Prepare masks and data first:
```bash
python gen_masks.py --root <MSRS>/train --vis_dir vi --ir_dir ir \
    --prior tophat --out data/msrs_masks_tophat
```
The `--prior` argument selects the teacher map. The paper distils the probe against
the morphological top-hat targetness prior, so it is `tophat` here; the flag also
accepts a path to an ONNX saliency model, which the earlier design used and the
paper does not.
Train the reconstruction auto-encoder and distil the probe (the official model):
```bash
python -m ssgf.train --mode ae   --root <MSRS>/train --vis_dir vi --ir_dir ir \
    --out data/msrs_ae.pt --cfg seed=2026 epochs=30 lr=2e-3
python -m ssgf.train --mode probe --root <MSRS>/train --vis_dir vi --ir_dir ir \
    --mask_vis_dir data/msrs_masks_tophat/vis --mask_ir_dir data/msrs_masks_tophat/ir \
    --ae_ckpt data/msrs_ae.pt --out data/msrs_probe.pt --cfg seed=2026 epochs=30 lr=2e-3
```
The default in `ssgf/config/defaults.py` is also `lr=2e-3`, matching the Methods
section of the paper.
Fusion metrics on the three test sets:
```bash
python evaluate.py --ae data/msrs_ae.pt --probe data/msrs_probe.pt --prior tophat \
    --max_side 512 \
    --datasets 'TNO::<tno>/tno_trial::vis::ir' \
    --datasets 'MSRS::<MSRS>/test::vi::ir' \
    --datasets 'RoadScene::<RoadScene>::crop_HR_visible::infrared'
```
Multi-seed stability, downstream detection/segmentation, and the SOTA baselines
(PIAFusion / SPGFusion) are reproduced by `sota_runners/multiseed_vif.py`,
`sota_runners/det_eval_fused.py`, `sota_runners/seg_eval_fused.py`, and
`sota_runners/{run_piafusion,run_spgfusion}.py`. The detection tables, the paired
bootstrap intervals, the architecture and seed comparisons and the VIF schedule
comparison are produced by the scripts in `analysis/`; `analysis/README.md` maps
each script to the paper table it fills. Note that `sota_runners/det_ci.py` is a
legacy routine that was not used for the paper's intervals. See `SOTA_RESULTS.md`
for the aggregated numbers.

The independent-test detection protocol is reproduced by
`scripts/prepare_clean_evaluator_dataset.py`, `scripts/train_clean_r2.py`,
`scripts/score_clean_evaluators.py` and `analysis/regime_bh.py`. It uses a fixed
1,350/150/300 LLVIP split, trains YOLO11n, YOLO26n and RT-DETR-l visible-only
evaluators plus the modality-agnostic YOLO11n control, and scores all fourteen fused
inputs and both sources. `data/splits/clean_split_manifest.json` records every stem.
The frozen outputs and training manifests are under `results/clean_r2/`.

Figure 11 is rebuilt by `scripts/make_fig11_decision_v2.py`. The script reads the
released clean-split and VIF aggregates, asserts the 42/42 and endpoint counts, and
exports editable PDF, SVG and 300 dpi PNG versions.

The three-way comparison of the reported fidelity index against the reference pixel
VIF is produced by `analysis/vif_reference_rescore.py`, which takes one fused-image
directory per method. The reference construction is implemented as
`ssgf.utils.metrics.vifp_reference`, and `report()` now carries both `VIF` and
`VIF_ref`.

For a full parallel run, use `analysis/run_reference_vif_parallel.py` with
`deploy/reference_jobs.example.json`; completed methods are skipped on restart. The
frozen results in `results/reference_vif/` cover all 14 LLVIP inputs on 300 images
and all eight benchmark methods on TNO (74), MSRS-test (361) and RoadScene (221).
SPGFusion is first under the reported index and last under the reference
construction on all three benchmarks.

The alignment check recovers the exact 300 LLVIP source stems from the public ZIP,
downloads only those JPEGs, and estimates a translational ECC residual on normalized
Sobel edges. See `results/alignment/README.md`. The best-aligned clean-split quartile
reverses several pooled signs, so the paper reports the negative result as a
population-level and stratum-dependent finding rather than a universal property.

## Note on baselines and checkpoints

The small checkpoints needed to reproduce the main tables (`msrs_ae.pt`,
`msrs_probe.pt`, `msrs_csf.pt`) are included in `data/`. The detection weights for
the LLVIP and FLIR runs of the paper, the adaptation checkpoints of Section 5, the
fused-image outputs and the per-image detection scores are archived with the frozen
runs rather than in this repository, because of their size; the scripts in
`analysis/` state which inputs each of them reads. The segmentation weights are
trainable with `seg_downstream.py` and the IFCNN / DenseFuse weights with
`run_baselines.py`.

## Citation

If you use this code, please cite the paper (add the BibTeX on acceptance).

## License

MIT (see `LICENSE`).
