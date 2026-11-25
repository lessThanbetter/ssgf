# Revision notes for v24

Date: 2026-10-29

## Review-driven changes

1. Confidence-ordered matching sensitivity
   - Rescored all six final LLVIP checkpoints and both VEDAI runs with the
     standard descending-confidence assignment rule.
   - The AP50 conclusion is unchanged: null versus visible, negative versus
     fixed CDF, and positive versus no shield.
   - AP50--95 is now reported as a separate negative boundary result.

2. Switch-budget-matched random control
   - Added 1,000 LLVIP random allocations.
   - Each allocation preserves the exact number of switched images and the
     exact multiset of CDF, PIAFusion, and SPGFusion selections.
   - CS-ECF exceeds the matched-random median by `+0.01392`
     `[+0.00964,+0.02023]` under standard matching.

3. Main evidence chain
   - Figure 2 now shows fixed CDF, PIAFusion, and SPGFusion differences across
     all 12 retrospective checkpoints, directly exposing sign changes.
   - Held-out LLVIP and VEDAI comparisons use standard matching.
   - The threshold sensitivity grid moved to Supplementary Figure S8.

4. External-evaluator boundary
   - Figure 3 now includes the no-shield contrast.
   - The text states that no shield exceeds CS-ECF on both zero-shot
     evaluators, while the LLVIP-trained evaluator reverses the visible-relative
     sign.

5. Case studies
   - Replaced the prior endpoints with frame `070142` for a true recovered
     pedestrian and frame `070162` for a lost matched true positive.
   - Solid boxes now show only matched true-positive predictions, so the
     displayed evidence agrees with the AP calculation.

6. Operational guidance
   - Figure 5 now presents four explicit deployment gates.
   - AP50-only validity, matched-random validation, detector-specific
     validation, and four-pass latency are all required.

7. Evidence completeness
   - Added complete 22-group M3FD split manifest.
   - Added all 12 external-evaluator per-image score arrays and hashes.
   - Fixed the `equal_group_statistics.py` terminal `NameError`.
   - Corrected Figure S7 scope and Figure S5 source data.

## Claim boundary

The manuscript is positioned as a boundary study. It does not claim a new
fusion operator, a new selective-prediction theory, general superiority over
visible input or fixed CDF, localization robustness, universal portability, a
formal risk guarantee, or a tuned performance ranking from the single-seed
YOLOv11-RGBT pilot.
