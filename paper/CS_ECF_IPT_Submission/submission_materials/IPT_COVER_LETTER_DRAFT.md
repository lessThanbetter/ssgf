# Initial Submission Cover Letter Draft

Target journal: Infrared Physics & Technology  
Article type: Research Paper

---

Dear Editors,

Please consider our manuscript, **“Confidence-Shielded Evaluator-Conditioned
Fusion for Risk-Controlled Infrared–Visible Object Detection”**, for publication
in *Infrared Physics & Technology*.

Infrared–visible fusion can make thermal evidence available to object detectors,
but the same fused image does not benefit every detector. We show that fusion
utility changes across detector checkpoints and formulate selection as a
label-free input-arbitration problem. The proposed Confidence-Shielded
Evaluator-Conditioned Fusion (CS-ECF) evaluates visible, CDF, PIAFusion, and
SPGFusion candidates with the target detector. A fused input is used only when
its top-five confidence exceeds that of the visible image by a factor of 1.5;
otherwise, the system preserves the visible input as a safety fallback.

Across LLVIP, M3FD, and an independent official-test subset of DroneVehicle,
CS-ECF improved 9 of 12 detector checkpoints. The dataset-balanced AP50 gain was
0.0133, with a training-run-by-group hierarchical 95% interval of [0.0026, 0.0261].
Removing the shield increased the worst observed AP50 loss from 0.0040 to
0.0584. The two DroneVehicle checkpoints improved by 0.0251 and 0.0293; they
belong to the same training run and are reported as correlated checks rather
than independent architectures. A thermal-mask ablation on DroneVehicle further
showed a small but consistent advantage over global detail injection
(0.0007 AP50 on average).

Against the stronger Always CDF baseline, the dataset-balanced difference was
0.0102 AP50, with a training-run-by-group interval of [-0.00057, 0.02980].
The interval included zero; the point estimate was concentrated in M3FD and
DroneVehicle and was not present on LLVIP. We therefore describe a
risk-control and conditional-generalization contribution, not dominance over
the best fixed fusion input.

We believe the work is relevant to the journal's readership because it addresses
how thermal information should be admitted into a deployed detection system,
uses real infrared–visible datasets, and reports both the benefit and cost of
the fusion candidates. The analysis also identifies an important boundary:
M3FD gains are concentrated in the people class, and the LLVIP intervals cross
zero. We therefore present CS-ECF as a risk-controlled evaluator-conditioned
system for pedestrian-centric and mixed IR–visible detection, not as evidence
of uniform improvement over all object classes.

The manuscript is original, has not been published previously, and is not under
consideration by another journal. An earlier, related manuscript by the same
authors on evaluator-dependent fidelity-index rankings was previously submitted
to this journal and was not accepted. It is not under consideration elsewhere.
The present manuscript is a distinct method study; it does not reproduce the
earlier fidelity-ranking claims, and all shared datasets and evaluator
infrastructure are identified in the manuscript. All authors have approved the
submission and declare no competing interests. This research did not receive any
specific grant from funding agencies in the public, commercial, or
not-for-profit sectors. Data and code availability are stated in the manuscript.

Thank you for your consideration.

Sincerely,

Zigao Wu  
North China Electric Power University  
Baoding, Hebei 071003, China  
zgwu@ncepu.edu.cn

