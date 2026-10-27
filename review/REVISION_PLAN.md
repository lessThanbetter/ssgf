# INFPHY-D-26-02136 — what the rejection says and what to do

Decision: reject, three reviewers. Reviewer 1 wrote 20 major and 12 minor points, all
about the submitted version; Reviewer 2 wrote four; Reviewer 3 wrote three sentences
(structure, figures, novelty).

The reviews are largely correct. The most damaging point is not the novelty objection
but an internal contradiction: the headline is measured against the colour-scored
visible source and Table 3 was printed in the single-channel convention, where several
fused inputs do beat the source. A reader could not verify the headline from the
manuscript, which is fatal for a paper whose subject is measurement discipline.

## Fixed in the current build (this pass)

| ID | Point | Fix |
|---|---|---|
| M1 | Headline contradicted by the table | Table 3 now lists the visible source in both conventions (0.8289/0.8209/0.8624/0.7561 single channel; 0.8809/0.8723/0.9065/0.8541 colour), and the lead-in states that several fused inputs beat the source in the single-channel convention but none does in the colour convention. |
| M2a | 0.830 to 0.881 without an architecture | Now 0.829 to 0.881 under YOLO11n, with increments 0.051, 0.044 and 0.098 for YOLO26n, RT-DETR-l and the modality-agnostic detector. |
| M2b | 0.152 not traceable in Table 3 | The colour source row makes 0.8809 − 0.7296 = 0.1513 visible in the table. |
| M2c | 42/42 versus a single RT-DETR-l run | The table caption says the 42/42 count uses the three-run mean of Supplementary Sec. 17 while the column is one run. |
| M9 | "one control" never named | The abstract now says "the activity control"; Section 5 names it with 0.8836 against 0.8624 (+0.021) in the single-channel convention and −0.014 in the colour convention. |
| M20a | 14 variants versus 12 inputs | Section 4 states the arithmetic: ten rules of Table 3 plus four scrambled controls of Supplementary Table S16 give fourteen inputs and forty-two contrasts over three architectures. |
| M20b | keywords differ from the system | The manuscript now carries the same five keywords as the submission system. |
| Minor 11/12 | Zenodo DOI and AI declaration | Both are in the manuscript: Code availability (10.5281/zenodo.22984389) and the AI declaration, which is language-only. |
| R2-1 | Fig. 5 caption contradicts its own correlation | The caption now says SSIM against the visible source stays positive under all four evaluators with a threefold range, while the reported VIF changes sign and entropy stays between −0.37 and +0.06. |

## Open points, by cost

## Fixed in the P1 pass (this round)

- **M3**: the text now states that the authors' own four rules are among the fourteen that
  lose (top-hat −0.050, probe −0.055 under YOLO11n), explains that the probe is distilled
  from the prior and is not expected to exceed its teacher, and separates the correlation
  by family: −0.10/+0.10/+0.10/+0.50 within the authors' five rules, against
  −0.80/−0.80/−1.00/−0.80 within the four published networks.
- **M6**: the two index constructions (reported: widths 2^{s-1}, no decimation, 8-bit
  noise; reference: canonical widths, decimation, input-unit noise) are now defined in
  the main text.
- **M7**: the discussion now recommends deploying the visible source for a visible-trained
  detector, and fusion only for detectors trained on thermal or fused data, with a rule
  for the case where the evaluator is not fixed at design time.
- **M8**: the modality-agnostic detector is now specified: YOLO11n, both modalities, 1350
  training / 150 validation pairs, 300 test images held out, four independent clean-split
  runs.
- **M10**: the text names the activity-only control as the best family member under
  YOLO26n and RT-DETR-l and gives the probe deficit against it.
- **M13**: the FLIR table is now stated to be single-channel and not colour-rescorable.
- **M14**: replaced by "the sign is set by the rule and the evaluator's modality, while
  the magnitude varies with the architecture by about a factor of two", with the three
  SPGFusion numbers.
- **M15**: the family (42 contrasts over three architectures) and the Benjamini–Hochberg
  criterion (FDR 0.05) are now named.
- **M16**: the 0.014 lower end is traced to the activity-only control's three RT-DETR-l
  runs (−0.013, −0.018, −0.011).
- **M18**: the cost of all four rules and four baselines is now a sentence in the main
  text.
- **M17**: a segmentation paragraph now gives the network, dataset, class count and the
  strict all-nine-classes convention.
- **M5**: the per-sequence sign is now given for all four families (six negative, one
  positive, the weakest-source sequence), and the five strata are stated to be
  non-pre-registered and descriptive.
- **R2-4** and **minor 1–10**: matching order, 101-point AP, the library release, LLVIP
  and FLIR sizes, the pedestrian histogram, the definition of "resolved", the budget
  confound, the stratum thresholds, the Table 4 convention, the validation caveat and the
  library version are now in the manuscript.
- Numeric correction found in this pass: the headline range "0.023 to 0.152" was wrong;
  the 42 colour-scored contrasts run from 0.014 (activity control, three-run mean) to
  0.152. Abstract, highlights, introduction, contribution, results and conclusion were
  corrected, and the clean-split lower bound in Section 6 is 0.022.

### Text only (1–2 days)

| ID | Point | Action |
|---|---|---|
| M3 | The paper's own rules also lose, and the probe trails the prior on two benchmarks | Add a symmetry sentence: the four rules of this paper are among the fourteen that lose, so the negative is not a defence of the authors' family. Explain that the probe is distilled from the prior and is therefore not expected to exceed its teacher; report the correlation separately for the authors' family and the published family. |
| M6 | Load-bearing evidence lives in a supplement that was not sent | Upload the supplement as a separate file, cite it by section from the main text, and move the two index definitions and the per-seed values into the main manuscript. |
| M7 | No actionable recommendation | State plainly that for a detector trained on visible images one should deploy the visible source, and that fusion is justified only for detectors trained on thermal or fused data; discuss the case in which the evaluator is not fixed at design time. |
| M8 | Modality-agnostic detector underspecified | State its training pairs (both modalities, same 1500), whether any fusion rule builds its training inputs, and that the clean-split panel uses four runs of the same detector. |
| M10 | Activity-only beats both targetness rules under two of three detectors | Report the full within-family ranking per evaluator and say that the targetness weighting is not the best downstream member of its family. |
| M13 | FLIR replication is in the weaker convention | State in the main text that Table 4 is single-channel and cannot be colour-rescored because the FLIR checkpoints are unavailable, so it replicates the sign but not the colour-convention magnitude. |
| M14 | "Penalty set by the rule rather than the detector family" overreaches | Replace with: the sign is set by the rule and the evaluator's training modality, while the magnitude varies with the architecture by about a factor of two. |
| M15 | Multiplicity method unnamed | Name Benjamini–Hochberg, define the family (42 contrasts over three visible-trained architectures), give the threshold (q < 0.05) and print q values for the borderline rows. |
| M16 | 0.023 lower bound not checked against seed noise | Give the four-seed range of the rule that produces 0.023 and show that it excludes zero. |
| M18 | Cost table missing from the main text | Restore the four-rule and four-baseline parameter/latency table to the main text. |
| M20c–g | Naming the index, duplicate VIF tables, naming the two networks, Fig. 1 caption | Name the reported index at first use, state that Tables 1 and 2 share the same measurements, name SPGFusion and PIAFusion, and make Fig. 1's caption self-contained. |
| Minor 1–10 | Definitions, split sizes, FLIR subset size, pedestrian histogram, "resolved", budget confounding, stratum thresholds, Table 4 convention, validation caveat in Section 4, library version | Add these sentences in the places the reviewer lists. |
| R2-2 | Validation set overlaps the test set | Restructure Section 4 so that the clean-split (test images held out) result is the primary table and the original protocol is the optimistic comparison. |
| R2-4 | Detection procedure unclear | State that boxes are matched per image after confidence sorting, that TP flags are pooled over images and that AP is the detector library's own 101-point interpolation. |

### Analysis with existing data (3–5 days)

| ID | Point | Action |
|---|---|---|
| M4 | Correlation table hides which inputs are missing | Print fidelity versus contrast for all fourteen inputs, name the thirteen that have both values, state the retention criterion, and give the reference VIF for all eight retained variants. |
| M5 | Strata chosen post hoc | Clarify that strata are defined from the annotation and the visible source, not from detection quality; add a Benjamini–Hochberg correction over strata; report per-sequence sign counts for all four input families rather than SPGFusion only. |
| M11 | Normalisation argued, not shown | Report the mean and spread of the per-source weights at each temperature for the three rules, from the recorded sweeps. |
| M17 | Segmentation barely reported | State network, dataset (MSRS-test, 361 images), classes, and the strict all-nine-classes mIoU convention, and why the present-classes convention (about 0.30) flips the visible/infrared ranking. |
| M19 | Registration/annotation alternative | State the registration procedure and add the alternative explanation to the limitations; an alignment-error estimate requires image-level correspondence metadata. |

### New computation or experiment (1–2 weeks)

| ID | Point | Action |
|---|---|---|
| R2-3 | Reference-implementation VIF for every method | Rescore the reference construction on the complete set of TNO/MSRS/RoadScene fused outputs; the LLVIP subset exists for eight variants, but the three fidelity benchmarks need re-running. |
| M11 (full) | Weight distributions | Export the per-pixel weights for one scene per rule to show the distributions rather than only their deviation. |
| M19 (full) | Alignment subset | Needs pairwise alignment metadata or a re-registration, then a contrast on the best-aligned subset. |

## What this means for the next submission

The paper cannot be resubmitted as it stands, and an appeal would fail because the
contradiction between Table 3 and the text is verifiable. The realistic path is:

1. Finish the text-only list and the existing-data analyses, which also answers the two
   IPT reviewers' most substantive points.
2. Restructure the paper around the clean-split result, with both source conventions
   printed everywhere.
3. Submit to a venue that publishes measurement and benchmark work rather than a new
   fusion network: IEEE Transactions on Instrumentation and Measurement, or Measurement.
   Optics & Laser Technology is a fallback if an infrared-optics framing is preferred.
   Pattern Recognition, TMM, TIP and IPT itself are not viable for this positioning.

If the authors instead want to keep IPT as a target, the only route is a substantial
rewrite with a new method contribution, which is a different paper.

## Post-rejection pass completed

- **R2-2:** Section 4 now leads with a fixed 1,350/150/300, zero-overlap split. Table 4
  reports all 14 inputs under clean YOLO11n, YOLO26n, RT-DETR-l and the clean
  modality-agnostic control. The pooled audit remains as a labelled sensitivity
  analysis.
- **R2-2 numbers:** all 42 clean visible-only evaluator-input contrasts are negative,
  over -0.046 to -0.246 mAP50. YOLO26n and RT-DETR-l each resolve 14/14 inputs after
  BH correction. The clean modality-agnostic split gives four stable gains of
  0.022 to 0.029.
- **M4:** Table 8 prints the reported VIF, visible-source VIF, reference VIF where
  retained, pooled contrast and clean-split contrast for all fourteen inputs. It names
  the six variants without a reference score and explains the 30-image retention.
- **M11:** the main text now reports the per-image mean absolute weight deviation and
  cross-image standard deviation at all four temperatures on TNO and MSRS, in addition
  to matched-deviation controls.
- **R3 structure and figures:** Fig. 2 now separates the primary and optimistic
  protocols; Fig. 11 states both the clean-split and pooled reversal counts. The
  manuscript compiles to 21 pages, a 250-word abstract, 11 figures and 11 tables.

## Final closure pass

- **R2-3 (full reference VIF) is complete.** All 14 LLVIP inputs were rescored on all
  300 images, and all eight methods were rescored on the complete TNO, MSRS-test and
  RoadScene sets. SPGFusion is first under the reported index and last under reference
  VIF on all three benchmarks; the frozen tables are in `results/reference_vif/`.
- **M19 (best-aligned subset) is complete as a sensitivity analysis.** The 300 LLVIP
  source pairs were recovered from the public release, and a translational ECC residual
  was estimated for every pair. The best-aligned clean-split quartile reverses the sign
  for mean fusion, top-hat and the probe, while the worst-aligned quartile contains the
  largest deficits. The source-to-results chain is in `results/alignment/`.
- **The finding changes the claim.** The pooled deficit remains, but it is not invariant
  to source alignment or scene composition. The manuscript now states this qualification
  in the abstract, results, discussion, limitations and conclusion.

## Latest verified closure, 26 October 2026

The historical action lists above are retained as an audit trail. The current build has
now also closed the following items.

- **M1, M12, M15:** the main text prints every fused input under both source conventions
  and gives the exact BH q value in each cell. The single-channel and colour-scored
  comparisons are corrected as separate 42-test families.
- **M5:** each visible-trained architecture has a separate BH correction over 14 inputs
  by six overlapping strata. The corrected counts replace the earlier uncorrected
  “all fourteen in the crowded stratum” statement.
- **M6, M16:** the main seed table now lists seeds 0/1/2/3 and the four-seed mean and
  range for all six contrasts.
- **M18:** the main text includes one protocol-matched table of parameters, latency and
  extra peak memory for the four author rules and four reproduced baselines.
- **Minor 1–10 and Reviewer 2.1/2.3/2.4:** metric definitions, official split sizes,
  pedestrian histogram, numeric strata thresholds, matching order, library version and
  complete reference VIF are present.

One item remains outside text-only closure.

Reviewer 3’s novelty objection remains. The current contribution is a measurement and
benchmark argument, not a new fusion network. A method-level claim requires a different
study. The current manuscript is therefore packaged for *Measurement*, where
evaluation validity and protocol evidence fit the journal scope.

## Measurement packaging

- The title now foregrounds evaluator-dependent measurement validity rather than a new
  fusion operator.
- The abstract treats fidelity metrics as proxy measurement instruments and states the
  test-independent holdout, 42/42 negative result, multiplicity correction and evaluator
  reversal.
- The submission package contains the main PDF, supplementary PDF, five editable
  highlights, a one-page cover letter and a pre-submission checklist.
- Figure 11 was rebuilt as a vector schematic anchored by real visible, infrared and
  fused examples, with the numerical claims verified against the released aggregates.
- The methodological caveat is explicit: the paper is not a new fusion-network paper,
  and its result is bounded to the measured pedestrian-detection population.
- The authors confirmed on 27 October 2026 that the data DOI is public and the author
  information and declarations are correct.
