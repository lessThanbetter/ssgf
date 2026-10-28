# Major-revision evidence

The JSON files in this directory support the statistical and boundary analyses
in the manuscript and supplement.

- `major_revision_statistics.json`: checkpoint contrasts, sign test, Holm/BH
  values, hierarchical intervals, stability runs, policy controls, and oracle.
- `major_revision_statistics_raw.json`: pre-postprocessing statistical output.
- `image_quality_baselines.json`: VIF, SSIM, entropy, random-policy, and oracle
  comparisons.
- `image_quality_features.json`: per-image quality features used by those
  selectors.
- `external_evaluator_diagnostics.json`: decision fractions, per-scene outcomes,
  switched-image degradation, and calibration summaries.
- `m3fd_class_summary.json`: M3FD annotation counts and class-specific AP50
  intervals.
- `alignment_csecf_sensitivity.json`: LLVIP ECC-residual quartiles and
  checkpoint contrasts.
- `prefreeze_validation_summary.json`: complete margin search used before the
  confirmatory tests.
- `confidence_shield_validation_search.json`: source search grid for the
  pre-freeze policy selection.
- `llvip_test_sampling_summary.json`: 3,463-pair official test frame, non-empty
  count, and 300-pair subset accounting.
- `llvip_test_manifest_3463.json`: complete official-test stem list used to
  verify subset membership.

All result files were generated on 2026-10-28. The reproduction scripts and
their required inputs are documented in `../../reproduction/README.md`.

`../reference_audit.json` is the separate Crossref field-level audit for all 45
DOI-backed bibliography entries.
