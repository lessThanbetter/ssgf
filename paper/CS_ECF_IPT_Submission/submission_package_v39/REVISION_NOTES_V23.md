# Revision Notes for v23

Date: 2026-10-29

## Reviewer-directed five-point revision

| Request | Revision |
|---|---|
| Add a recent joint fusion-detection baseline | Trained YOLOv11-RGBT early fusion on the fixed M3FD split and added the system-level comparison |
| Compress the main text to one evidence chain | Reordered the manuscript around evaluator dependence, shield-controlled tail loss, and the visible/CDF boundary |
| Move detailed checkpoints, scenes, and controls | Retained compact main summaries; moved complete checkpoint, scene, calibration, class, thermal, and control evidence to seven supplementary figures and tables |
| Replace a weak summary display with real switch cases | Added one successful and one failed image-level switch under a fixed selection rule |
| Give operational guidance | Added explicit yes/no conditions for visible fallback, Always CDF, and CS-ECF |

## Editorial restructuring

1. Reduced the main manuscript from 28 to 17 pages by replacing the detailed
   retrospective results and discussion with compact evidence-led sections.
2. Moved checkpoint-level intervals, complete CDF comparisons, calibration,
   class and thermal analyses, and auxiliary controls to the Supplementary
   Material.
3. Replaced the separate retrospective summary figures in the main text with a
   single three-panel evidence figure covering evaluator dependence,
   confirmatory comparisons, and observed tail loss.
4. Added a qualitative case-study figure showing one successful and one failed
   CS-ECF switch. Cases are selected by a fixed rule: the largest positive and
   negative per-image AP50 changes among switched images with at least two
   ground-truth boxes.
5. Added an explicit operational recommendation that visible fallback remains
   the default unless a large switch-induced regression is the main risk and
   the target detector has been validated on held-out data.

## Recent joint fusion-detection baseline

The public YOLOv11-RGBT early-fusion framework was trained on the same
1,600/400/200 M3FD split under a fixed 40-epoch budget. The model receives a
four-channel visible--infrared input and optimizes fusion and detection in one
network. The configuration, initialization, checkpoint hash, held-out
predictions, class-level AP, and frame-group AP are recorded under
`evidence/yolo11rgbt_baseline/`. This is a recent end-to-end system reference,
not an architecture-matched substitute for the within-checkpoint CS-ECF
comparisons. Under the study AP protocol, the single run obtained AP50 0.0963
and AP50--95 0.0467 on the fixed M3FD test groups.

## Reporting boundary

The paper does not claim a general visible-relative gain, superiority over
Always CDF, a confidence-specific mechanism, a formal risk guarantee, or
universal evaluator portability.
