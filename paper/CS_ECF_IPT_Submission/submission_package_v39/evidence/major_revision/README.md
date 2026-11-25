# Major-revision evidence

The JSON files in this directory support the statistical and boundary analyses
in the manuscript and supplement.

- `equal_group_statistics.json`: canonical equal-group point estimates,
  paired resampling intervals, stability runs, AP50--95 sensitivity, and the
  retrospective score-depth sensitivity grid.
- `major_revision_statistics.json`: merged checkpoint summaries and
  descriptive policy controls. Historical pooled values are retained under
  explicit `pooled_` keys.
- `major_revision_statistics_raw.json`: complete numerical copy retained for
  audit.
- `image_quality_baselines.json`: VIF, SSIM, entropy, random-policy, and
  label-using heuristic comparisons.
- `image_quality_features.json`: per-image quality features used by those
  selectors.
- `external_evaluator_diagnostics.json`: decision fractions, per-scene outcomes,
  switched-image degradation, and calibration summaries.
- `m3fd_class_summary.json`: M3FD annotation counts and class-specific AP50
  intervals.
- `alignment_csecf_sensitivity.json`: LLVIP ECC-residual quartiles and
  checkpoint contrasts.
- `prefreeze_validation_summary.json`: complete original development margin
  search. The search predates this package but lacks an external timestamp.
- `confidence_shield_validation_search.json`: source search grid for the
  original development policy selection.
- `llvip_test_sampling_summary.json`: 3,463-pair official test frame, non-empty
  count, and 300-pair subset accounting.
- `llvip_test_manifest_3463.json`: complete official-test stem list used to
  verify subset membership.
- `llvip_primary_split_manifest.json`: exact train, validation, and test stems
  with hashes and overlap checks.
- `m3fd_primary_group_manifest.json`: exact 200-image test stems and the two
  held-out frame groups.
- `dronevehicle_100frame_groups.json`: three selected 100-frame blocks and the
  nested source-prefix counts.

The files in this directory describe the retrospective analysis generated on
2026-10-28. Prospective evidence generated on 2026-10-29 is under
`../prospective/`. Reproduction scripts and required inputs are documented in
`../../reproduction/README.md`.

`../reference_audit.json` is the separate Crossref field-level audit for all 45
DOI-backed bibliography entries.
