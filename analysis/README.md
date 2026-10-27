# Analysis code for the evaluation tables

These are the scripts that produced the detection, segmentation and paired-interval
numbers of the paper. They were run on a single-GPU server with the datasets at
`/root/autodl-tmp`; adjust the `BASE` paths at the top of each file, or set them
through the command line, before rerunning. The model checkpoints they consume are
the small ones in `data/` plus the detector and segmenter weights trained by
`det_eval.py` and `seg_downstream.py`.

## What produces what

| Paper element | Script |
|---|---|
| Detection point values of the main tables | `j1_eval2.py` |
| Detection matching and the pooled metric | `j1_eval2.py` |
| Paired bootstrap intervals for LLVIP and FLIR | `paired_eval.py` |
| Architecture family, one training set, three architectures | `j6_det_family.py` |
| Four-seed replication of the two YOLO architectures | `j3_det_seeds.py`, `j9_seeds.py` |
| Second benchmark, FLIR-ADAS aligned | `j7_flir.py` |
| Adaptation variants, fusion and evaluation | `robust_eval.py`, `robust_eval_common.py` |
| Interval for the change in the average across the evaluator set | `net_ci.py` |
| Ranking robustness of VIF to the Gaussian schedule | `vif_schedule_robustness.py` |
| Rescoring every variant under the reference pixel VIF | `vif_reference_rescore.py` |
| Parallel, resumable full-set reference VIF jobs | `run_reference_vif_parallel.py` |
| Rebuilding reported/reference VIF tables | `summarize_reference_vif.py` |
| LLVIP source extraction and alignment residual | `extract_llvip_sources.py`, `estimate_alignment.py` |
| Best/worst-aligned subset detection check | `alignment_sensitivity.py`, `summarize_alignment_seeds.py` |
| Fixed clean LLVIP split and training manifests | `../scripts/prepare_clean_evaluator_dataset.py`, `../scripts/train_clean_r2.py`, `../scripts/collect_clean_training.py` |
| Independent-test scoring of all 16 inputs | `../scripts/score_clean_evaluators.py`, `../scripts/merge_clean_score_runs.py` |
| Clean-split BH counts within scene regimes | `regime_bh.py` |
| Clean-split fidelity-to-detection rank associations | `../scripts/compute_clean_correlations.py` |
| Figure 11 schematic from verified aggregate counts | `../scripts/make_fig11_decision_v2.py` |
| Activity-only control rule used by the FLIR and detection scripts | `k1_controls.py` |

`det_eval_any.py` is the older single-directory scorer; the values printed in the
main detection tables come from `j1_eval2.py`, which is also the module whose
matcher and pooled metric the intervals use. Its docstring describes the matcher as
the greedy IoU matcher of the training library; the implementation matches with
`scipy.optimize.linear_sum_assignment` on a binary cost, which maximises the number
of valid pairs rather than the total IoU, and on the published inputs the two agree
to about 0.0001. That difference is stated here rather than silently.

`vif_reference_rescore.py` is the script behind Supplementary Table S6 extended to
the published baselines: give it one `--fused name=directory` per method and it
reports the reported index, a width-only variant, and the reference construction
(`ssgf.utils.metrics.vifp_reference`). The full runs include SPGFusion, PIAFusion,
IFCNN and DenseFuse on every benchmark. The scorer resizes all three images of a pair
to one common grid defined by the visible source, and `--target_hw` fixes that grid
when needed. `--fields` permits a reference-only run, and `--limit 0` means the
complete dataset. Source images with a different aspect ratio are resized rather
than cropped; the RoadScene check depends on this convention.

For a full multi-method run, `run_reference_vif_parallel.py` launches one subprocess
per fused variant, logs progress independently, resumes completed variants, and
merges their rows into one JSON/CSV file. This is the path used for the 14-variant
LLVIP table and the three-benchmark reported/reference matrix.

`extract_llvip_sources.py` reads the remote LLVIP ZIP directory and downloads only
the source JPEGs named by the 300 analysis stems, verifying each member's CRC.
`estimate_alignment.py` estimates a translational residual with ECC on normalized
Sobel-edge maps. `alignment_sensitivity.py` uses only those source-pair estimates to
construct best and worst quartiles; it never selects on labels or detector scores.

`k1_controls.py` is here because `j7_flir.py` imports its `ActivityOnlyRule`. The
other scripts read fused-image directories and checkpoints that are archived with
the frozen runs rather than in the repository, and each of them names the inputs it
expects at the top of the file.

`sota_runners/det_ci.py` is a legacy routine that works at 512 px with a confidence
threshold of 0.25 and 200 resamples. It is not the routine that produced the
intervals in the paper. The paper's intervals come from `paired_eval.py` and
`net_ci.py`, which score at native resolution, use a confidence threshold of 0.001
and 1000 resamples, and recompute the same pooled metric on each resample.

## Per-image scores

`paired_eval.py` and `net_ci.py` read the per-image matching statistics that the
detection runs write next to each fused-image directory. Those per-image files,
together with the fused-image outputs, are archived with the frozen runs of the
paper and are available from the authors on request.
