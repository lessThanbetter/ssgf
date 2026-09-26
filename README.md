# Fidelity Metrics Do Not Predict Target-Centric Detection

Official code for "Fidelity Metrics Do Not Predict Target-Centric Detection: The
Evaluator Determines the Ranking of Visible-Infrared Fusion Rules". The method the
paper introduces is a target-aware semantic saliency fusion rule, instantiated with
a morphological top-hat targetness prior and a distilled probe.

We argue that the semantic that should drive visible--infrared fusion is
**targetness** (whether a pixel belongs to a small bright or dark thermal target),
not generic saliency or classification-based importance. We instantiate it with a
morphological top-hat prior and a small distilled probe, and use it to modulate
feature activity before a temperature softmax fuses the two sources.

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

## Repository layout

- `ssgf/`  package (config, data, fusion rules, models, metrics, checkpoints utils)
- `evaluate.py`  cross-dataset VIF / EN / SSIM / PSNR evaluation
- `train_fusion_e2e.py`  end-to-end fusion-aware training (stronger backbone option)
- `run_baselines.py`  train / evaluate IFCNN and DenseFuse on MSRS
- `train_classifier.py`  train the CSF classification-saliency baseline classifier
- `seg_downstream.py`  downstream semantic segmentation (UNet mIoU)
- `det_eval.py`  downstream detection (LLVIP YOLO) mAP evaluation
- `analysis/`  the scripts that produced the detection tables, the paired intervals
  and the VIF schedule comparison; see `analysis/README.md` for the mapping
- `sota_runners/`  helpers to run PIAFusion / SPGFusion and reproduce every table/figure
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

The three-way comparison of the reported fidelity index against the reference pixel
VIF is produced by `analysis/vif_reference_rescore.py`, which takes one fused-image
directory per method. The reference construction is implemented as
`ssgf.utils.metrics.vifp_reference`, and `report()` now carries both `VIF` and
`VIF_ref`.

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
