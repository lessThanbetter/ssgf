# Reviewer-Driven Revision Notes for v21

Date: 2026-10-29

## Main changes

1. The original task-balanced `+0.0092` estimate is labeled retrospective.
2. A scene-disjoint LLVIP confirmation tests the unchanged 1.5 policy.
3. The primary six-checkpoint mean is `-0.00065`; the eight-checkpoint
   sensitivity mean is `+0.00191`.
4. The manuscript now leads with the prospective result rather than the
   retrospective magnitude.
5. The positive evidence is narrowed to CS-ECF minus no shield:
   `+0.00976 [+0.00113,+0.02252]`, with all six primary checkpoints positive.
6. CS-ECF is explicitly not superior to Always CDF on the prospective LLVIP
   analysis.
7. DroneVehicle scoring now uses native rotated-box IoU and maximum-cardinality
   matching.
8. Two additional independent DroneVehicle runs are reported without dropping
   the negative run.
9. The image-quality control audit is reported as degenerate and does not
   support a confidence-specific mechanism.
10. Thermal p-values and significance stars are removed.
11. Stale per-dataset values are replaced by canonical equal-group values.
    Historical pooled values remain under explicit `pooled_` keys.
12. The preflight now checks prospective point estimates, intervals,
    independent-run values, protocol hashes, score-array counts, canonical
    copies, and nested retrospective evidence.

## Claim boundary

Supported:

- evaluator- and domain-dependent fusion utility;
- a near-zero prospective LLVIP mean relative to visible;
- a positive mean-tail advantage of the shield over no shield in the primary
  confirmation;
- heterogeneous DroneVehicle results across independent runs.

Not established:

- reproduction of the original `+0.0092` effect;
- superiority over Always CDF;
- a confidence-specific mechanism relative to a matched image-statistic gate;
- uniform improvement, universal portability, risk guarantees, or low-latency
  deployment.
