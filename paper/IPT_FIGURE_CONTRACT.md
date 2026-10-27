# IPT Figure Contract

Date: 2026-10-27  
Target journal: Infrared Physics & Technology  
Backend: Python / matplotlib  
Primary export width: 183 mm (double column)  
Text floor: 5 pt rendered; source minimum 7 pt  
Formats: editable SVG + PDF, 600 dpi TIFF, PNG preview

## Figure 1 — Fusion utility is evaluator-dependent

Core conclusion: the same IR–visible candidate library produces positive and
negative AP50 changes across detector checkpoints, so one fusion image cannot be
treated as universally optimal.

Archetype: quantitative grid with one hero heatmap.

Panel map:

- a — hero heatmap of fixed-policy gains for 12 checkpoints.
- b — dataset-level means for Always CDF, Always PIAFusion, Always SPGFusion,
  Top-5 no-shield, and CS-ECF.

Evidence hierarchy:

- hero: checkpoint-by-policy signed gains;
- validation: dataset-level sign reversal and CS-ECF positive model count.

Statistics: gain is AP50 difference from visible; no error bars in the heatmap;
dataset-level means use the checkpoint means in the generated result files.

Source data:

- `confirmatory_policy_comparison.json`
- export `figure1_source.csv`

Reviewer risk: the heatmap must not imply statistical significance for
individual cells. Caption states that cells are point contrasts.

## Figure 2 — CS-ECF system

Core conclusion: CS-ECF admits thermal fusion only when the target detector
supports a large visible-relative confidence margin; otherwise it preserves the
visible fallback.

Archetype: schematic-led composite.

Panel map:

- a — hero workflow from visible/IR pair to CDF, PIAFusion, SPGFusion, candidate
  confidence, shield, and selected detector input.
- b — decision equation and selected-image counts for the three confirmatory
  datasets.

Evidence hierarchy:

- hero: complete inference pathway;
- support: measured switch counts, not stylized imagery.

Statistics: counts are image-level selections from the frozen policy; dataset
sample sizes are 300, 200, and 300 for LLVIP, M3FD, and DroneVehicle.

Source data:

- `confidence_shield_three_dataset_summary.json`
- export `figure2_source.csv`

Reviewer risk: the figure must show that the detector is reused, not an external
label oracle. The scheme must show visible fallback explicitly.

## Figure 3 — Risk–return trade-off

Core conclusion: the unshielded selector has a higher mean on DroneVehicle but a
much worse worst-case LLVIP loss; the shield trades some average gain for a
smaller tail loss.

Archetype: quantitative grid with a hero risk–return plane.

Panel map:

- a — dataset-balanced mean gain versus worst checkpoint gain for five policies.
- b — paired checkpoint gains for no-shield and CS-ECF, sorted by no-shield gain.

Evidence hierarchy:

- hero: risk–return plane;
- validation: paired differences preserve checkpoint identity.

Statistics: point estimates only; no significance stars. The legend defines the
worst-case statistic and dataset-balanced weighting.

Source data:

- `confirmatory_policy_comparison.json`
- export `figure3_source.csv`

Reviewer risk: axes must not truncate away the loss region; the zero lines remain
visible.

## Figure 4 — Thermal constraints and class boundary

Core conclusion: the thermal mask provides a small consistent fixed-input gain,
but thermal contrast alone is not a sufficient switching rule, and M3FD gains
concentrate in people.

Archetype: asymmetric mixed quantitative grid.

Panel map:

- a — M3FD per-class gains for YOLO26n and YOLOv8n.
- b — thermal-mask versus global-injection CDF gains on DroneVehicle.
- c — Spearman association between switch fraction and thermal contrast/visible
  luminance across datasets.

Evidence hierarchy:

- hero: class concentration and its boundary;
- validation: mask ablation and thermal association.

Statistics: class bars report point AP50 gain; mask bars report fixed-input gain;
panel c reports Spearman rho and p as descriptive association.

Source data:

- `confidence_shield_class_bootstrap.json`
- `ipt_mask_ablation.json`
- `ipt_thermal_stratification.json`
- export `figure4_source.csv`

Reviewer risk: never imply that high thermal contrast always causes a switch or
improves AP. The DroneVehicle negative association remains visible.

