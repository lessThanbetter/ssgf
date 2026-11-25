# VEDAI external-domain confirmation protocol

Recorded: 2026-10-29T03:20:00+08:00

## Purpose

This protocol defines an external-domain test written before any CS-ECF score
was calculated on VEDAI. It does not replace LLVIP, M3FD, or DroneVehicle.
Its purpose is to test whether the already fixed CS-ECF policy transfers to an
aerial visible--infrared vehicle benchmark with different sensors, altitude,
target scale, and class composition.

## Dataset choice before outcome access

VEDAI was selected using metadata only:

- official 512-pixel visible and infrared image pairs;
- official oriented vehicle annotations;
- ten disjoint official test-fold partitions;
- five reportable vehicle classes: car, truck, tractor, camping car, and van;
- a distinct aerial sensor and target-scale regime;
- research-only terms available from the official provider.

The test fold is fold01, selected as the lexicographically first official fold.
Fold02 is used for detector validation. The union of fold03 through fold10 is
used for detector training. No CS-ECF outcome was calculated before this
protocol was written.

## Data integrity

- Template archive SHA-256:
  `AC44A27A5330F0EDF0158CC6F19DE4A7CC060A3FE9AA85A5BC9F9BCB2A6B5CB7`
- Annotations archive SHA-256:
  `AA0085FBAB16E61A325CD0DC18E4C4D192C4E0DCF5FC8A9A0166CFB06814725A`
- Official provider:
  `https://downloads.greyc.fr/vedai/`
- Terms:
  `https://downloads.greyc.fr/vedai/TermsandConditionsofUseVeDAI2014.pdf`

The prepared subset has 968 training images, 121 validation images, and 121
test images. Official fold membership is disjoint across the three subsets.
Polygon coordinates are clipped to the image boundary and converted directly
to native YOLO OBB labels.

## Detector

- Initialization: official `yolo11n-obb.pt`.
- Training input: visible images only.
- Two fixed runs with seeds 0 and 1.
- Thirty epochs, batch size 4, 640-pixel inputs, deterministic cuDNN.
- Final `last.pt` checkpoints are fixed before test scoring.
- Test results are not used for epoch, checkpoint, class, or threshold
  selection.

## Candidate generation

The four candidates are:

- visible;
- consensus-detail fusion using pixel mean and IFCNN details with
  `q=0.95`, `alpha=0.20`, and `sigma=5`;
- PIAFusion;
- SPGFusion.

Candidates are generated at the registered 512-pixel source geometry.
Fusion outputs are restored to source dimensions before scoring when needed.

## Policy

The policy remains unchanged:

- candidate score is the mean of the five highest confidences;
- a fused candidate is selected only when its score is at least 1.5 times the
  visible score;
- visible is otherwise retained.

No VEDAI result may change the candidate set, score depth, margin, class set,
split, detector checkpoint, or endpoint.

## Endpoints

Primary endpoint:

- equal-block mean AP50 difference between CS-ECF and visible for all five
  classes.

Secondary endpoints:

- CS-ECF minus Always CDF;
- CS-ECF minus no shield;
- car-only AP50;
- per-class AP50;
- fraction of images improved, unchanged, and degraded;
- switched-input fraction.

The 121 sorted test images form ten fixed adjacent-stem blocks of 12 or 13
images. Equal-block means are primary. Image-level bootstrap intervals are
secondary and descriptive. No significance claim is made because images may
share acquisition context.

## Reporting

All classes, both training runs, all four policies, and negative outcomes are
reported. The external-domain result cannot replace a negative primary result
from another dataset. No post-outcome fold or class substitution is allowed.
