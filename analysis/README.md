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
(`ssgf.utils.metrics.vifp_reference`). Rescoring SPGFusion, PIAFusion, IFCNN and
DenseFuse under the reference construction is the first item of future work stated
in the paper, and this script is what does it. It resizes all three images of a pair
to one common grid, and `--target_hw` fixes that grid when the fused outputs survive
only at a size different from the sources.

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
