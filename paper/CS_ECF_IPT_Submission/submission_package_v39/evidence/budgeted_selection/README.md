# Budgeted candidate-library analysis

This directory contains the post-review analysis requested to test whether
CS-ECF needs all four detector passes and all three fusion candidates.

## Selection protocol

The score depth (`K = 5`) and visible margin (`1.5`) remain unchanged. Every
candidate subset containing visible is evaluated on the original four
development evaluator cells:

- LLVIP: YOLO26n and RT-DETR-l;
- M3FD: YOLO26n and YOLOv8n.

A subset is feasible when every development evaluator gain is at least
`-0.0005 AP50`. Among feasible subsets whose mean gain is within `0.00001` of
the maximum, the analysis chooses the fewest detector forwards and then the
lowest measured component cost.

This procedure selects `visible + PIAFusion` at the frozen 1.5 margin. It uses
two detector forwards instead of four and omits the 516 ms SPGFusion component.
The estimated serial component cost is 102.7 ms versus 706.3 ms for the full
candidate library. These are component estimates and not end-to-end wall time.

## Main results

- Development: mean AP50 gain `+0.00206`; worst evaluator gain `+0.00039`;
  all four evaluator cells are positive.
- LLVIP, confidence-ordered holdout: mean gain versus visible `+0.00337`,
  conditional interval `[-0.00016, +0.01267]`; five of six checkpoints are
  positive.
- LLVIP versus no shield: `+0.01218`, conditional interval
  `[+0.00136, +0.02915]`; all six checkpoints are positive.
- LLVIP versus full CS-ECF: `+0.00325`, conditional interval
  `[-0.00026, +0.01051]`.
- VEDAI: `-0.00329` versus visible and `+0.05131` versus no shield.
- DroneVehicle independent runs: `-0.00136` versus visible and `+0.00263`
  versus no shield.
- External evaluators: `+0.01844` versus visible and `+0.00300` versus no
  shield.

The selected subset is therefore a checkpoint- and domain-specific budget
reduction. It does not transfer as a universal accuracy method. A deployment
must use target-domain validation and fall back to visible when the selection
constraint fails.

## Status

This is an exploratory post-review analysis. The 1.5 margin and development
evaluators were frozen previously, but the two-candidate subset was not part of
the original prospective protocol. The held-out comparisons are useful for
deciding how to revise the manuscript, but they do not turn the new subset into
a prospectively confirmatory result.

## Reproduce

From the repository root:

```powershell
python analysis/cs_ecf/budgeted_candidate_selection.py
python analysis/cs_ecf/plot_budgeted_candidate_selection.py
```

Outputs:

- `analysis.json`
- `validation_grid.csv`
- `retrospective_grid.csv`
- `FigureBudgetedSelection.pdf`
- `FigureBudgetedSelection.png`
- `FigureBudgetedSelection.svg`
