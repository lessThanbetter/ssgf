# CS-ECF analysis

These scripts reproduce the confidence-shielded evaluator-conditioned fusion
selection analysis and the IPT manuscript figures.

## Inputs

- Detector per-image score archives and fused candidate scores are under
  `results/selective_fusion/`.
- The final aggregate outputs are frozen under the same directory.
- Images and checkpoints are not included in the repository. Dataset locations
  are documented in the top-level README.

## Main analysis chain

1. `confidence_candidate_selection.py` — candidate and margin sensitivity.
2. `confidence_margin_validation.py` — validation-only margin choice.
3. `confidence_shield_summary.py` — LLVIP/M3FD policy summary.
4. `confirmatory_three_dataset_summary.py` — three-dataset summary.
5. `confirmatory_group_hierarchical.py` — detector-by-group hierarchy.
6. `confirmatory_policy_comparison.py` — fixed and selected baselines.
7. `confidence_shield_class_diagnostics.py` and
   `confidence_shield_class_bootstrap.py` — M3FD class effects.
8. `analyze_ipt_thermal_stratification.py` and
   `analyze_ipt_mask_ablation.py` — IPT thermal analyses.
9. `render_confidence_shield_report.py` — generated Markdown report.

The scripts currently retain the research workspace's absolute path constants.
Before running in a different checkout, set those roots to the clone location or
export a local data root. The frozen JSON/CSV/NPZ results are sufficient to
verify the reported tables without re-running detector inference.

