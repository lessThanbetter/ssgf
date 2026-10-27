# CS-ECF Manuscript Draft for Infrared Physics & Technology

Status: evidence-grounded drafting skeleton. Citations, figures, and the
IR-specific analyses listed at the end remain author inputs.

## 1. Working title

Recommended:

> **Confidence-Shielded Evaluator-Conditioned Fusion for Risk-Controlled
> Infrared–Visible Object Detection**

Alternatives:

1. **When Fusion Helps: Detector-Conditioned Input Arbitration for
   Infrared–Visible Detection**
2. **Risk-Controlled Thermal Fusion Selection for Infrared–Visible Pedestrian
   and Vehicle Detection**
3. **Evaluator-Conditioned Fusion Arbitration with Confidence Shielding for
   Multimodal Detection**

The first title is the safest: it names the system, task, and benefit without
claiming uniform improvement.

## 2. Terminology ledger

| Canonical term | First-use definition | Decision |
|---|---|---|
| infrared–visible (IR–visible) | thermal infrared and visible-light image pair | use en dash |
| CS-ECF | Confidence-Shielded Evaluator-Conditioned Fusion | define once |
| CDF | Consensus Detail Fusion | define once |
| detector evaluator | an object detector together with its trained checkpoint and inference protocol | use this when gains depend on detector weights |
| visible input | the visible-light image used as the safety fallback | do not call it “RGB fusion” |
| fused candidate | one of CDF, PIAFusion, or SPGFusion | candidate set excludes visible |
| shield margin | the factor \(\tau=1.5\) applied to visible confidence | use one term only |
| no-shield selector | the same top-k rule with \(\tau=1\) | do not call it the method |
| AP50 | mean average precision at IoU 0.50 | report as AP50, not mAP50, when single class |
| AP50–95 | mean AP across IoU 0.50:0.05:0.95 | use en dash |
| image CI | bootstrap interval over images | within-checkpoint uncertainty |
| group CI | bootstrap interval over scenes or consecutive frame groups | primary cluster-aware uncertainty |
| dataset-balanced gain | equal weight for LLVIP, M3FD, and DroneVehicle | do not use model-weighted mean as the headline |

## 3. One-sentence argument

In IR–visible object detection, fusion utility is evaluator-conditioned; a
validation-frozen confidence shield therefore converts a volatile per-image
fusion selector into a risk-controlled input arbitration policy that improves
dataset-balanced AP50 across three test datasets while reducing the worst
observed degradation.

## 4. Abstract draft

Infrared–visible image fusion is commonly treated as a preprocessing step whose
output should benefit any downstream detector. In practice, the same fused
image may improve one detector, leave another unchanged, and degrade a third,
because detection utility depends on the detector's learned representation and
confidence calibration. We introduce Confidence-Shielded
Evaluator-Conditioned Fusion (CS-ECF), a label-free input-arbitration policy
that evaluates visible, CDF, PIAFusion, and SPGFusion candidates with the target
detector itself. For each candidate, CS-ECF averages its five highest detection
confidences and permits a fused input only when its confidence exceeds that of
the visible image by a factor of 1.5; otherwise, it falls back to the visible
image. The margin was fixed on validation data before confirmatory testing. We
evaluated the frozen policy on LLVIP, a grouped six-class M3FD test set, and
three consecutive groups from the official DroneVehicle test split, comprising
12 detector checkpoints. Nine checkpoints improved, the dataset-balanced AP50
gain was 0.0133, and the model-by-group hierarchical 95% interval was
[0.0028, 0.0213]. Removing the shield introduced a worst AP50 loss of 0.0584,
whereas the frozen policy limited the worst loss to 0.0040. The two
DroneVehicle checkpoints improved by 0.0251 and 0.0293, although they belong to
the same training run. Gains on the six-class M3FD benchmark were concentrated
in the people class, and the LLVIP intervals crossed zero. These results support
CS-ECF as a risk-controlled, evaluator-aware fusion-input system for
pedestrian-centric and mixed IR–visible detection; they do not support a claim
of uniform improvement across all object classes. Under a standardized serial
benchmark, candidate generation and four detector evaluations required
approximately 0.71 s per image pair, dominated by SPGFusion.

## 5. Highlights

1. Fusion utility reverses across detector checkpoints.
2. A confidence shield gates image-level fusion-input selection.
3. Nine of 12 checkpoints improve AP50.
4. The worst AP50 drop falls from 5.84 to 0.40 points.
5. Gains concentrate in people; six-class uniformity is not claimed.

## 6. Introduction

### Paragraph 1 — field stake

Infrared–visible imaging combines the spatial detail and color of visible
sensors with thermal evidence that can remain available when illumination is
poor, smoke reduces contrast, or targets have weak visual texture. This
complementarity has motivated a broad family of pixel-, feature-, and
network-based fusion methods for surveillance, autonomous perception, and
thermal imaging systems. In detection-oriented applications, however, the value
of a fused image is not determined by visual appeal alone; it is the change in
detector outputs under a specified operating condition. [Cite: IR–visible
fusion foundations and representative fusion networks.]

### Paragraph 2 — bottleneck

Current practice often selects one fused representation and evaluates it with
one detector on one benchmark. This design assumes that a globally preferred
fusion result remains preferable across detector architectures and training
seeds. The assumption is fragile: detectors differ in receptive fields,
confidence calibration, training data, and sensitivity to high-frequency
detail. Consequently, a fusion rule that improves one evaluator may reduce the
AP of another on the same image. [Cite: fusion evaluation and detector-aware
fusion studies.]

### Paragraph 3 — why simple selection is unsafe

Per-image selection offers a natural response: run several fusion candidates
and choose the one with the strongest detector response. Yet unconstrained
selection converts small calibration differences into frequent input switches
and can produce severe losses when a spurious high-confidence response is
mistaken for improved detection. The problem is therefore not only to predict
which fused image looks best, but to decide when the evidence is strong enough
to depart from the visible-image safety baseline. [Cite: adaptive fusion,
uncertainty-aware fusion, and detector confidence calibration.]

### Paragraph 4 — gap

Existing work provides fusion operators, selection heuristics, or
detector-specific training procedures. It does not establish a label-free
arbitration rule whose switching threshold is fixed independently of the
confirmatory tests and whose behavior is reported at the level of detector
checkpoint, image, and scene/frame group. This distinction matters because
image-level resampling can hide temporal dependence, while pooling
checkpoints without accounting for dataset composition can make a headline gain
look more general than the evidence permits.

### Paragraph 5 — present study

We formulate IR–visible fusion selection as evaluator-conditioned input
arbitration. The proposed CS-ECF evaluates four candidates with the target
detector, computes a top-five confidence score, and accepts a fused candidate
only when its score reaches 1.5 times that of the visible input. The threshold
is selected on validation data and frozen before testing. We evaluate the policy
on LLVIP, M3FD, and DroneVehicle with model-, image-, and group-aware
uncertainty, compare it with fixed fusion and an unshielded selector, and
report the cost and failure boundaries. The contributions are:

1. an evaluator-conditioned formulation of fusion-input selection that does not
   use test labels;
2. a confidence shield that reduces catastrophic switching while retaining
   positive dataset-balanced gains;
3. a three-dataset confirmation that separates image-level uncertainty,
   scene/group uncertainty, and checkpoint variability, including explicit
   evidence that the M3FD gain is concentrated in the people class.

## 7. Methods

### 7.1 Problem formulation

Let \(x_v\) and \(x_i\) denote a registered visible–infrared pair, and let
\(\mathcal{D}\) denote the target detector checkpoint. A candidate
\(c\in\mathcal{C}\) produces an input image \(x_c\), where
\(\mathcal{C}=\{\mathrm{vis},\mathrm{cdf},\mathrm{pia},\mathrm{spg}\}\).
The policy returns one candidate per pair without access to ground-truth boxes:

\[
g(x_v,x_i;\mathcal{D})\in\mathcal{C}.
\]

The final detection output is
\(\mathcal{D}(x_{g(x_v,x_i;\mathcal{D})})\). The detector is therefore reused
for candidate evaluation and final inference; it is not an external oracle.

### 7.2 Fusion candidates

The visible input is always included as a safety candidate. CDF combines a
pixel-mean image and an IFCNN image through sign-consistent high-frequency
details. A thermal-validity mask is formed from the infrared top-hat response,
infrared intensity, visible luminance, connected components, and smoothing.
The shared detail is

\[
d(x)=
\begin{cases}
\operatorname{sign}(d_m+d_i)\min(|d_m|,|d_i|), & d_m d_i>0,\\
0, & \text{otherwise},
\end{cases}
\]

and is injected into the LAB luminance channel with
\(L'=L+\alpha M d\). The frozen settings are \(q=0.95\), \(\alpha=0.20\), and
\(\sigma=5\). PIAFusion and SPGFusion are generated using their official model
definitions and pretrained weights under the same image-pair protocol. [Cite:
pixel-mean, IFCNN, PIAFusion, and SPGFusion sources.]

### 7.3 Confidence-shielded arbitration

For candidate \(c\), let
\(s_{c,1}\ge s_{c,2}\ge\cdots\) be detector confidences. We define

\[
q_c=\frac{1}{\min(K,n_c)}\sum_{j=1}^{\min(K,n_c)}s_{c,j},
\qquad K=5,
\]

where \(n_c\) is the number of detections. Candidates with no detections
receive \(q_c=-1\). The highest-scoring candidate is
\(c^\*=\arg\max_c q_c\), with ties resolved in favor of visible. The selected
input is

\[
c=
\begin{cases}
c^\*, & c^\*\ne\mathrm{vis}\ \text{and}\ q_{c^\*}\ge1.5q_{\mathrm{vis}},\\
\mathrm{vis}, & \text{otherwise}.
\end{cases}
\]

The margin \(1.5\) was selected on validation data before the confirmatory
tests. The selector uses no test labels and makes no class-specific decision.

### 7.4 Datasets and evaluators

LLVIP supplies a 300-image pedestrian test set spanning seven scenes. M3FD
supplies a grouped six-class test set of 200 images, split by consecutive frame
groups to reduce adjacent-frame leakage; the detector families are YOLO26n and
YOLOv8n. DroneVehicle contributes 300 image pairs from three fixed consecutive
100-frame groups of the official test archive. The groups were selected with a
predefined seed before evaluation, and the YOLO11x-OBB best and last checkpoints
were both evaluated. All model outputs use a confidence floor of 0.001 and an
NMS IoU of 0.7 for the OBB detector; exact settings for each detector appear in
Supplementary Table S1.

### 7.5 Statistical analysis

Point estimates use pooled AP50 after concatenating detections across the test
images within a checkpoint. Image bootstrap intervals resample images within a
checkpoint. Group bootstrap intervals resample scenes or consecutive frame
groups and are the primary clustered intervals. The dataset-balanced estimand
assigns equal weight to LLVIP, M3FD, and DroneVehicle, then resamples
checkpoints and groups. M3FD has only two test groups and LLVIP has seven
scenes, so the group bootstrap is deliberately reported as a finite-cluster
analysis rather than a high-precision population estimate. [Author input:
confirm the exact bootstrap seed, resample count, and software version in the
finalized run.]

### 7.6 Implementation and cost

At a standardized fusion input of \(384\times512\), FP32, batch size one, on an
NVIDIA RTX 2080 Ti, CDF required 22.3 ms, PIAFusion 37.7 ms, and SPGFusion
516.3 ms. Four serial YOLO11x-OBB evaluations at an input size of 640 required
130.0 ms, giving an estimated serial total of 706.3 ms. This estimate excludes
model loading or swapping, data loading, and final detection post-processing.
SPGFusion's additional peak activation memory was 1453.5 MiB; per-stage memory
values must not be summed as an end-to-end peak.

## 8. Results

### 8.1 Fusion utility is detector-dependent

Across the 12 confirmatory checkpoints, the same candidate library produced
both positive and negative gains. CS-ECF improved nine checkpoints; its
dataset-balanced AP50 gain was 0.0133, with a model-by-group hierarchical 95%
interval of [0.0028, 0.0213]. The unshielded selector improved only seven of 12
checkpoints, had a larger dataset-balanced mean of 0.0332, and produced a worst
loss of 0.0584. The shield therefore traded part of the high-end gain for a
substantial reduction in tail risk.

[Place Table 3 and Figure 1 here.]

### 8.2 Dataset-specific behavior

On LLVIP, the mean gain was 0.0016 and five of eight checkpoints improved; the
model, image, and group intervals crossed or approached zero depending on the
resampling unit. On M3FD, both six-class checkpoints improved, with a mean gain
of 0.0111. On the independent DroneVehicle test subset, the best and last OBB
checkpoints improved by 0.0251 and 0.0293; the combined group interval was
[0.0002, 0.0406]. All three DroneVehicle groups were non-negative, but one
gained only 0.0002. The two OBB checkpoints originated from the same training
run and are not independent architectures.

[Place Table 2 and Figure 2 here.]

### 8.3 The shield is a risk-control mechanism

Without the shield, the policy selected fused candidates on most LLVIP and
DroneVehicle images and occasionally produced large confidence gains that did
not translate into AP. The 1.5 margin reduced switches and lowered the worst
AP50 loss from 0.0584 to 0.0040. On DroneVehicle, however, the shield also
reduced the mean gain from 0.1040 to 0.0272. This trade-off is the central
quantitative meaning of “risk-controlled” in the manuscript.

[Place Figure 3 here.]

### 8.4 Fixed baselines and class concentration

Always CDF improved 10 of 12 checkpoints and had a dataset-balanced gain of
0.0031, but remained negative on both M3FD detectors. Always PIAFusion and
Always SPGFusion had negative dataset-balanced gains because their strong
DroneVehicle performance did not transfer to the other datasets. In the M3FD
six-class evaluation, the CS-ECF gain was concentrated in the people class:
YOLO26n improved by 0.0214 and YOLOv8n by 0.1090, while the other five classes
changed by at most 0.0003 in absolute AP50. The evidence therefore supports a
pedestrian/people-focused interpretation, not uniform six-class improvement.

[Place Table 4 and Figure 4 here.]

### 8.5 Thermal feature association

To test whether switching follows simple thermal evidence, we related the
per-image fraction of fused selections to infrared top-hat contrast, visible
luminance, and CDF mask activity. On LLVIP, thermal contrast showed a weak
positive association with switching (Spearman \(\rho=0.151\), \(p=0.009\)),
whereas visible luminance was only weakly negative (\(\rho=-0.104\),
\(p=0.071\)). On M3FD, thermal contrast showed a moderate positive association
(\(\rho=0.320\), \(p<10^{-5}\)). On the DroneVehicle test subset, however, both
thermal contrast (\(\rho=-0.403\)) and visible luminance
(\(\rho=-0.441\)) were negatively associated with switching. The
high-thermal/low-visible median split also did not produce a consistently
larger AP gain than the remaining images.

These results do not support a universal rule in which higher thermal contrast
causes a switch or guarantees a benefit. Instead, they show that raw thermal
saliency is not a sufficient proxy for detector-conditioned fusion utility.
CS-ECF therefore remains an evaluator-conditioned policy; its thermal candidate
CDF is one option, not a fixed thermal-dominance rule. This negative
stratification result should remain in the manuscript because it defines the
physical interpretation more honestly.

A fixed-input mask ablation provided a smaller but more consistent thermal
effect on DroneVehicle. Replacing the thermal-validity mask with global detail
injection changed the CDF gain from +0.0060 to +0.0057 for the best checkpoint
and from +0.0050 to +0.0038 for the last checkpoint. The thermal mask therefore
added +0.0007 AP50 on average relative to global injection. This result supports
a modest benefit from spatially constraining thermal-detail insertion, but it
does not establish a general physical law.

## 9. Discussion

The central finding is that fusion utility is not an intrinsic property of a
fused image alone. A candidate can improve one detector's confidence ordering
while harming another's, and a policy that ignores this dependence can select
inputs that appear confident but do not improve AP. CS-ECF responds by using
the target detector only as an input arbiter and by requiring a large
visible-relative margin before switching.

The shield's value is clearest in the tail rather than in the mean. The
unshielded selector achieved a higher dataset-balanced average on DroneVehicle,
where both PIAFusion and SPGFusion were strong, but it also produced severe
LLVIP losses. The 1.5 margin deliberately favors visible fallback under
ambiguous evidence. This is consistent with a deployment rule in which a small
probability of a large regression is more costly than a missed opportunity for
a moderate gain.

Two boundaries constrain the interpretation. First, the strongest M3FD effect
is class-specific: people detection improves substantially while the remaining
classes are nearly unchanged. This suggests that the selected thermal evidence
is most useful for a subset of target appearances, not that fusion improves all
six vehicle classes. Second, the DroneVehicle confirmation uses two checkpoints
from the same training run. The official test groups are independent of model
selection, but the checkpoints share training data and are correlated.

The cost analysis also matters. SPGFusion dominates latency, and the four
detector evaluations are not free. CS-ECF is therefore best viewed as a
risk-controlled system for settings where the target detector is already run
and the cost of a catastrophic input choice is high. It is not a low-latency
replacement for a single fusion operator.

The thermal stratification adds a second limitation. Thermal contrast may
increase the likelihood of selecting fused evidence in LLVIP and M3FD, but the
association reverses on DroneVehicle. This suggests that sensor-level thermal
evidence and detector confidence capture different information; neither should
be interpreted as a universal utility score. The IPT framing should therefore
be “thermal fusion utility under a detector-conditioned decision rule,” not a
claim that thermal contrast alone identifies the best fused image.

## 10. Conclusion

We showed that IR–visible fusion utility is evaluator-conditioned and proposed
CS-ECF, a confidence-shielded input-arbitration policy that falls back to the
visible image unless a detector-supported fusion candidate clears a fixed
visible-relative margin. Across three test datasets and 12 checkpoints, the
policy improved nine checkpoints, produced a dataset-balanced AP50 gain of
0.0133, and reduced the worst observed degradation from 0.0584 to 0.0040. The
evidence supports risk-controlled, pedestrian-centric IR–visible detection; it
does not support uniform six-class improvement, and the additional cost of
SPGFusion and four detector evaluations must be considered in deployment.

## 11. Claim–evidence map

| Claim | Evidence | Status |
|---|---|---|
| Fusion utility depends on the detector checkpoint | Positive and negative gains across 12 checkpoints for the same four candidates | supported |
| CS-ECF reduces tail degradation | Worst gain -0.0040 vs -0.0584 for no-shield | supported |
| CS-ECF improves dataset-balanced AP50 | +0.0133; model×group CI [0.0028,0.0213] | supported, conditional |
| DroneVehicle test confirms transfer | best +0.0251; last +0.0293; group CI [0.0002,0.0406] | supported, same training family |
| Gains generalize uniformly across six M3FD classes | Other five classes change by ≤0.0003 | rejected |
| Always CDF is fully dominated | Always CDF positive on 10/12 checkpoints | rejected |
| The system is cheap | Serial estimate 706.3 ms, dominated by SPGFusion | rejected |

## 12. IPT-specific analyses still required

These analyses are needed to strengthen the thermal-imaging contribution
before submission:

1. **Thermal contrast stratification**: compute infrared local contrast or
   top-hat energy for selected and non-selected images; test whether the
   probability of switching increases when visible contrast is low and thermal
   evidence is high.
2. **Thermal-mask ablation (first pass completed)**: DroneVehicle fixed-input
   AP50 favors the mask by +0.0007 on average; still need switch-rate and
   false-switch examples.
3. **Day/night or illumination split**: report gains separately for LLVIP
   night scenes and M3FD/DroneVehicle conditions, without changing the frozen
   policy.
4. **Thermal qualitative panels**: show visible, infrared, CDF, PIAFusion,
   SPGFusion, detected boxes, and the shield decision for matched success and
   failure examples.
5. **Compute statement at the actual sensor resolution**: repeat the cost
   benchmark for the target IR–visible resolution, or state explicitly that
   inference is standardized to \(384\times512\) before fusion and detection.

## 13. Assumptions and missing inputs

- No verified reference list has been attached to this draft. All citation
  markers are placeholders and must be replaced from inspected papers.
- The exact journal length limit, reference style, and Highlights character
  limit must be checked against the current IPT author instructions.
- The final figures and table numbers do not yet exist.
- The 500-draw group bootstrap should be repeated with the final analysis
  settings if the journal expects a specific resampling convention.
- The detector checkpoint hashes, data licenses, code repository, and model
  release statement remain author inputs.

