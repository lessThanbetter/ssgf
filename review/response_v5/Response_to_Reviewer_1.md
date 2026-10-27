# Response to Reviewer Comments

**Previous manuscript:** Evaluator-Dependent Validity of Fidelity Metrics for
Visible-Infrared Image Fusion in Pedestrian Detection

**Previous submission:** INFPHY-D-26-02136

**Prior editorial decision:** Reject

We thank the reviewer for the detailed and technically demanding report. The
comments identified material inconsistencies in the earlier presentation, including
the mismatch between the claimed source convention and the printed table. We have
treated these as substantive corrections rather than editorial objections. The
current manuscript now separates the independent-test result from the pooled
sensitivity audit, prints both source conventions where they are available, and
states the limits of the stratification and alignment analyses. The table numbers
below refer to `PAPER_MEASUREMENT_v5.pdf`. Because the evaluator inventory is now
Table 1, the tables cited in the original report have shifted.

## Comment 1

> The headline claim is contradicted by the table said to support it. Section 4 states
> that “Table 3 holds the fused images fixed and varies only the detector … and no
> fused input reaches the unfused visible source under any of the three visible-trained
> architectures.” Table 3 is printed “in the single-channel convention of the reported
> protocol,” and in that convention the claim is false: under YOLO11n, top-hat fused
> (0.8306) and ensemble fused (0.8291) exceed the visible source (0.8289); under
> YOLO26n, activity-only (0.8310) and DenseFuse (0.8257) exceed the visible source
> (0.8209); and under RT-DETR-l the visible source is 0.8624 while mean (0.8646),
> activity-only (0.8836), probe (0.8651), ensemble (0.8708), DenseFuse (0.8715) and CSF
> (0.8632) all exceed it. The claim is only true under the colour-scored convention
> (visible source raised to 0.881), which the authors state is “the convention used
> above and in Supplementary Sec. 13” but which is not shown in Table 3. The reader
> therefore cannot verify the paper’s central result from the paper. This must be fixed
> by (a) printing Table 3 with the colour-scored visible source as an explicit column
> (or a second sub-table), so that the headline contrast is visible in the manuscript,
> and (b) correcting the sentence that describes Table 3, which as written describes a
> table that is not in the manuscript. I would also ask the authors to state plainly,
> in the main text, that in the single-channel convention used for all rows, the
> visible source is penalised and several fused inputs do beat it, since this is the
> convention the paper defends as treating every row alike, and it is the convention
> in which the negative result is weakest.

**Response.** We agree with this criticism and have removed the inconsistent
table-to-claim link. The primary independent-test comparison is now Table 5, where
the visible source is colour-scored under the detector's training convention. All 42
visible-trained evaluator-input contrasts are negative on the fixed
1,350/150/300 split. Table 5 reports the four-run YOLO11n and modality-agnostic
results with their full run ranges, and the single-run YOLO26n and RT-DETR-l results
with paired image intervals.

The pooled sensitivity audit is now separated in Tables 6 and 7. Table 6 reports
both source conventions and states that several single-channel comparisons are
unresolved or favourable to fusion. Table 7 prints per-input single-channel and
colour-scored values with Benjamini-Hochberg q values for the two single-run YOLO
evaluators. The text now says explicitly that the colour-scored pooled deficits span
0.014 to 0.152, while the single-channel convention does not support the same
universal negative claim.

**Manuscript change.** Section 4, Tables 5 to 7, pp. 5 to 7.

## Comment 2

> The numbers in the text do not reconcile with the numbers in the tables. Several
> specific instances: Section 4 states the visible source is “raised from 0.830 to
> 0.881 mAP50” by colour scoring, but Table 3 lists the visible source at 0.8289
> under YOLO11n, 0.8209 under YOLO26n and 0.8624 under RT-DETR-l. It is never stated
> which architecture the 0.830 and 0.881 belong to, nor whether the same increment
> applies to all three. The deficits are quoted as “0.023 to 0.152 mAP50” throughout
> the abstract, introduction and conclusion, but Table 3’s largest deficit is
> SPGFusion against the visible source: 0.8289 − 0.7296 = 0.0993 under YOLO11n and
> 0.8209 − 0.7290 = 0.0919 under YOLO26n. A deficit of 0.152 does not appear in Table
> 3 at all. The text later says “SPGFusion by 0.091 against 0.152 under YOLO11n”
> (Section 5), which implies the 0.152 is the RT-DETR-l figure, but Table 3 gives
> 0.8624 − 0.8105 = 0.0519 there, not 0.152. Please state, for every headline number,
> exactly which table cell and which convention produced it. Section 5 states that
> under RT-DETR-l “the control row that is mid-pack elsewhere becomes the best fused
> input of any kind under it once the reported protocol’s single-channel convention is
> used for every row”, but Table 3 shows activity-only at 0.8836 as the best under
> RT-DETR-l, and it is not clear which row is meant by “the control row.” The abstract
> says “all forty-two contrasts are resolved once the third architecture is repeated”
> while Section 4 says the RT-DETR-l column in Table 3 “is one training run; Fig. 2
> and Supplementary Sec. 17 use the mean of three.” The reader cannot tell whether the
> headline 42/42 result comes from the single run shown or the three-run mean.

**Response.** We agree that the earlier text mixed protocols and roundings. The
current manuscript uses two clearly separated result sets.

The 42/42 statement belongs to the primary independent split in Table 5. It counts
14 fused inputs under each of three clean visible-trained evaluators. YOLO11n was
repeated four times; YOLO26n and RT-DETR-l each have one clean run. Their deficits
span 0.046 to 0.246.

The 0.014 to 0.152 range belongs to the pooled, validation-overlapping sensitivity
audit, where the visible source is colour-scored and RT-DETR-l is represented by the
mean of three runs. We no longer use one range while citing a table from the other
protocol. The activity-only single-channel value under one RT-DETR-l run is +0.023,
but its BH q value is 0.218. The three-run mean is -0.014, with interval
[-0.027, -0.001], which is the value used in the pooled count.

**Manuscript change.** Abstract; Section 4; Table 5; Table 6; Table 7; pages 1 and
5 to 7.

## Comment 3

> The fidelity comparison is structured such that the authors’ own rules rank near
> the bottom, which weakens the argument they are used to make. Table 1 shows the
> authors’ rules at VIF 0.3298 (probe) and 0.3561 (top-hat) on TNO, against SPGFusion
> at 0.8135 and PIAFusion at 0.6051, that is, the authors’ rules rank fifth and sixth
> of seven, and the learned probe does not exceed the published baselines “on any
> benchmark” (Section 3). The paper’s central rhetorical move is that “the two methods
> the fidelity index ranks highest are the two that lose most” (SPGFusion and IFCNN).
> But since the authors’ own rules perform poorly on the very index being criticised,
> the argument has an appearance of being self-serving: the index is denounced because
> it favours methods other than the authors’. The paper would be considerably stronger
> if it (a) stated explicitly that the authors’ rules also fail to beat the visible
> source (top-hat −0.0504, probe −0.0552 against the colour-scored source, by the
> authors’ own numbers), so that the negative result is symmetric and not an artifact
> of the authors’ rules being favoured; and (b) reported the fidelity-versus-detection
> correlation separately for the authors’ family and for the published family, so the
> reader can see that the mismatch is not a property of one group. Note also that the
> manuscript never explains why the top-hat prior improves VIF over mean fusion by
> 33–36% yet the learned probe, trained by distillation against that same prior, is
> beaten by it on two of three benchmarks. That is an unexpected internal result that
> deserves comment.

**Response.** We agree that the result must not read as a defence of the proposed
family. The manuscript now states that the four author rules are among the inputs
that lose under visible-trained evaluators. In the pooled colour audit, top-hat and
the probe lose by 0.050 and 0.055 under YOLO11n, and the clean-split table reports
their per-run behaviour directly.

We also report fidelity-detection rank correlations by family. Within the author
rules the correlations are -0.10, +0.10, +0.10 and +0.50 across the evaluators.
Within the four published networks they are -0.80 under each evaluator. The
aggregate mismatch therefore does not arise solely because one family occupies a
different part of the fidelity ranking.

The probe is distilled from the top-hat prior and is not designed to exceed its
teacher on every reconstruction benchmark. We now state that the probe is level with
the prior on two benchmarks and behind it in detection by 0.0048, so the targetness
map buys fidelity stability rather than a universal downstream gain.

**Manuscript change.** Sections 4 to 6, especially pp. 6, 8, 12 and 18; Table 11;
Fig. 5.

## Comment 4

> The fidelity score reported for the authors’ own rules never appears in the
> detection comparison, so the index-to-detection link is not shown for their family.
> Table 7 correlates fidelity with contrast “over the thirteen inputs for which both
> are recorded,” but the reported-index column and the reference-VIF column are not
> populated for all rows and the reference VIF is “rescored on the eight variants
> whose fused outputs were retained.” The paper does not state which inputs are
> missing from the correlation, nor whether the authors’ four rules (mean, top-hat,
> probe, ensemble) are among the thirteen. Since those four are the only rules whose
> fidelity the paper controls, the correlation that matters most for the paper’s
> thesis is the one over its own family, and it is not separable from the table as
> printed. Please give the full fidelity-versus-contrast data for all fourteen inputs
> in one table, with the missing entries explained.

**Response.** The complete data are now in Table 11. It lists all 14 fused inputs
with reported-index VIF, reference pixel VIF, the pooled YOLO11n contrast, the clean
YOLO11n mean, and the clean YOLO26n and RT-DETR-l contrasts. The author family is
explicitly identified. The separate family correlations in the text use the same
values.

The earlier omission arose from an outdated fidelity export that did not contain CSF.
Figure 5 now identifies CSF as the one input excluded from that legacy 13-input
scatter because its score was not retained in that export. The complete rescoring
does include CSF and all other inputs in Table 11.

**Manuscript change.** Table 11, p. 12; Fig. 5, p. 13; Sec. 6.

## Comment 5

> The stratification claim is not fully supported by the evidence presented in the
> manuscript. Claim (iii) states that the deficit is “a stratification, not a regime
> effect,” resolved for all fourteen rules in the crowded, small-target and
> high-contrast strata and for only two in the bottom tercile. Figure 3a is said to
> count, “for each regime defined from the annotation and the unfused visible source
> alone, how many of the fourteen inputs are resolved below the source.” Three
> problems: (a) the strata are defined from the unfused visible source’s own detection
> quality, which is the same quantity being compared, so the stratification is
> defined partly in terms of the outcome and is not independent of it; (b) the
> manuscript says the sign of the association between contrast and deficit “changes
> from sequence to sequence” without quantifying how often, and sequence-level results
> are given only for the SPGFusion contrast (Table 11, restricted to sequences with at
> least twenty test images); and (c) selecting the strata post hoc from the same 300
> images used for the headline claim, with no correction for the stratum search, means
> the “resolved for all fourteen” statement carries an unquantified selection effect.
> Please report the strata counts with multiplicity correction over strata, state
> whether a pre-registered stratum definition would give the same partition, and give
> the per-sequence sign-count over all sequences rather than for one contrast only.

**Response.** We agree that the earlier stratification language exceeded the
evidence. The strata are now described as post hoc summaries of the pooled
sensitivity audit, not as universal regimes or causal categories. The manuscript
states that the strata use annotations and visible-source properties that co-vary
with the outcome, and that this dependence limits interpretation.

Benjamini-Hochberg counts are reported separately for each architecture over the
14-input by six-stratum family in Supplementary Table S16. They do not produce a
single evaluator-invariant partition. Under the clean split, the counts also differ
by evaluator and are reported in Supplementary Table S29. The text now says that
these analyses are exploratory and that a pre-registered study would be required to
test a regime hypothesis.

Per-sequence results are reported for all four input families in Supplementary
Tables S18 and S19. Six of seven LLVIP sequences are negative, and the single
positive sequence is the one with the weakest visible source.

**Manuscript change.** Section 4; Discussion, p. 19; Fig. 3; Supplementary Tables
S16, S18, S19 and S29.

## Comment 6

> The paper’s own methodological argument is undermined by keeping its most important
> evidence in an unavailable supplement. The manuscript repeatedly defers
> load-bearing evidence: the colour-corrected protocol (Sec. 12, which is the
> convention that makes the headline claim true), the clean-split protocol (Sec. 16),
> the per-seed tables (Table S7), the component ablation (Tables S1–S3), and the
> per-evaluator adaptation results (Tables S24, S25), the per-image scores and fused
> images, and the “remaining metrics.” The supplementary file was not provided with
> the review copy, and several of these items are essential to evaluating the central
> claims (see M1, M2). For publication, at least the following must be in the main
> manuscript: Table 3 in the colour-corrected convention (or both conventions side by
> side); the per-seed values behind Table 6; and the exact definition of the two index
> constructions whose disagreement is one of the paper’s two headline results.

**Response.** The current submission includes the supplementary PDF and has moved the
load-bearing material into the main manuscript. Table 5 carries the primary clean
split. Tables 6 and 7 print both source conventions and the relevant q values.
Table 10 gives seeds 0/1/2/3, the four-seed means and full ranges for the pooled
architecture contrasts. Section 3 defines the two VIF constructions and states where
they differ. Per-image scores, frozen aggregates, manifests and analysis code are
part of the released package.

Component ablation and additional model diagnostics remain in the supplement because
they test secondary questions. The main-text sections that rely on them now identify
the relevant supplementary table and summarize the result needed to evaluate the
claim.

**Manuscript change.** Section 3; Sections 4 to 6; Tables 5, 6, 7, 10 and 11;
Supplementary Material.

## Comment 7

> The paper does not explain what a practitioner should do, which is the stated
> purpose. The conclusion is that “a rule should therefore be reported with its
> evaluator and index.” This is a statement about reporting, not about choice, and the
> paper explicitly declines to recommend a rule. Given the empirical result that
> under a visible-trained detector nothing beats not fusing at all, the actionable
> conclusion would seem to be that for visible-trained detectors one should deploy
> the visible source directly, and that fusion is only justified for detectors
> trained on thermal or fused modalities. The manuscript should state this explicitly
> and discuss the practical consequence, rather than leaving the reader to infer it.
> Relatedly, the paper should say what it recommends for the common deployment case
> where the detector’s training modality is unknown or is itself part of the design
> decision.

**Response.** We agree and have added an explicit recommendation. For a detector
trained on visible images, the default input should be the visible source. A
deployment-specific validation remains necessary for unusually well-aligned sources
or a different target distribution. Fusion is justified when the detector is trained
on thermal or fused data.

When the evaluator is not fixed at design time, the detector modality should be
chosen first from the modality the deployment will actually observe. The fusion rule
should then be selected for that evaluator and deployment stratum, not from a pooled
fidelity score alone.

**Manuscript change.** Discussion, p. 19; abstract and Conclusion.

## Comment 8

> The modality-agnostic result needs a clearer account of what the detector was
> trained on and why it is the right control. Section 4 and Table 3 report an
> “agnostic” detector under which ten of fourteen fused inputs exceed the visible
> source (and the IR source reaches 0.9415, far above every fused input). The
> manuscript states this detector was “retrained with the 300 test images held out”
> but does not state what training data it was trained on, presumably both modalities,
> nor how the fusion of the two sources into the training set was constructed, nor
> whether it was given the same fourteen fused variants at training time or a
> different fusion rule. Since the reversal of the sign depends entirely on this
> detector, its construction is a load-bearing detail. Please specify the training
> modality composition, the fusion rule used to build its training inputs (if any),
> and whether the “clean split” experiment of Fig. 4a uses four independently trained
> runs of the same detector or of a different one.

**Response.** The evaluator is now specified in the main Methods and in Table 1. It
is a YOLO11n trained on the visible and infrared images from the same 1,350 training
pairs, concatenated as input channels. No post-training fusion rule is applied. The
150 held-out validation pairs choose the checkpoint, and the 300 test pairs are
excluded from training, validation and checkpoint selection. Four independently
trained seeds are used. Figure 4a and Table 5 use those same four
modality-agnostic runs.

This is a control for evaluator training modality, not a universal
modality-agnostic deployment model. The text limits the interpretation accordingly.

**Manuscript change.** Section 2; Table 1, p. 4; Section 4; Fig. 4a.

## Comment 9

> The “deficit is not uniform / one control is resolved above the source” result is
> reported but never interpreted. The abstract, Section 4 and the conclusion all note
> that under RT-DETR-l “one control is resolved above the source instead.” Which
> control, by how much, and why? A single fused input that beats the visible source
> under a visible-trained detector, when the paper’s headline is that none does, is a
> counterexample that the paper must confront rather than mention in passing. If the
> counterexample is the activity-only control (0.8836 vs 0.8624 in Table 3 under
> RT-DETR-l, a +0.021 gain), that is a substantial and directly relevant exception,
> and it happens to be the classical L1-norm activity weighting, i.e. the rule the
> paper’s targetness machinery is supposed to improve upon. Please analyse this case
> in the main text, and state whether the headline claim should be qualified to “no
> fused input beats the visible source except under RT-DETR-l where the activity-only
> control does.”

**Response.** The control is now named and interpreted. In the pooled single-channel
single-run audit, activity-only has a positive point contrast of about 0.023 under
RT-DETR-l, but its BH-adjusted q value is 0.218, so it is not a resolved gain. In
the three-run RT-DETR-l analysis the mean is -0.014 with interval
[-0.027, -0.001], below the source. The clean RT-DETR-l primary run also places all
14 fused inputs below the source.

We therefore do not hide the single-run positive sign. The text states that it is
unresolved and that the repeated run reverses its sign. The headline claim is tied
to the clean split and to the resolved pooled comparison, not to every individual
single-run point estimate.

**Manuscript change.** Abstract; Sections 4 and 5; Tables 5 to 7; p. 7.

## Comment 10

> The activity-only control outperforms the paper’s targetness rules under two of
> three visible-trained detectors, which undercuts the framing. Table 3: activity-only
> reaches 0.8277 / 0.8310 / 0.8836 under YOLO11n / YOLO26n / RT-DETR-l, versus top-hat
> at 0.8306 / 0.8170 / 0.8632 and probe at 0.8258 / 0.8128 / 0.8651. Under YOLO26n
> and RT-DETR-l the plain activity rule, which the paper defines as the control
> separating targetness from activity, beats both targetness rules. Table 6 even
> reports “probe − activity-only” as −0.017 under YOLO11n and −0.011 under YOLO26n,
> i.e. the learned probe is worse than the parameter-free activity weighting. The
> paper reports these numbers faithfully but does not draw the implication: the
> targetness weighting that occupies Fig. 1 and Section 2 is, downstream, not the
> best member of its own family. This should be stated and discussed, and the claim
> that the top-hat prior “beats mean fusion” should be replaced by a statement about
> the full ranking within the family under each evaluator.

**Response.** We agree. The manuscript now states directly that the parameter-free
activity rule is the best member of the author family in some clean single-run
evaluators and that the targetness weighting does not provide a downstream gain. It
also states that the probe loses to activity-only by 0.049 to 0.061 under the two
clean single-run evaluators in the relevant comparison.

The fidelity claim about the top-hat prior is retained only for the fidelity score.
The downstream claim is now about the full ranking under each evaluator. We do not
present targetness weighting as the best downstream rule.

**Manuscript change.** Sections 4 to 6, especially p. 8 and pp. 18 to 19; Tables 5
and 11.

## Comment 11

> The (w_v) and (m) normalisation argument is asserted rather than demonstrated.
> Section 2 argues that the product (m), not (w_v), sets how soft the rule is, “which
> is what makes fidelity comparisons between rules with different weight spreads
> meaningful.” Table 9 is said to report a temperature sweep with detection at four
> temperatures under two evaluators. The claimed invariance of the comparison under
> the (m) normalisation should be shown directly, e.g. by reporting the resulting
> weight distributions (mean and spread of (w_v)) at each (τ) and demonstrating that
> the normalised weights are comparable across rules with different activity spreads.
> As written, a reader cannot check whether the normalisation actually achieves what
> is claimed, and this matters because the paper’s targetness rules differ from mean
> fusion precisely in weight spread.

**Response.** The manuscript now reports the per-image mean absolute deviation of
the visible-source weight from 0.5 and its cross-image standard deviation at
temperatures 3, 1, 0.3 and 0.1. Values are given for activity-only, the top-hat prior
and the probe on TNO and MSRS. The comparison is also repeated at matched weight
deviation in Supplementary Table S2. Figure 8 shows the detection consequence across
the same temperature range.

These statistics expose the difference in weight sharpness rather than asserting
that the normalized comparison is invariant by construction.

**Manuscript change.** Section 6; Fig. 8; Supplementary Table S2; p. 15.

## Comment 12

> The single-channel convention, which the paper defends, is acknowledged to
> penalise the visible source, but the paper then uses the colour convention for its
> headline and the single-channel convention for its tables. The authors state in the
> limitations that “every input is scored in a single channel, which we chose so that
> all rows are treated alike but which penalises the visible source, the only input
> that could have been scored in the colour convention the detectors were trained on.”
> This is a defensible choice for the table, but it makes the paper’s headline claim
> (which requires the colour-scored source) and the paper’s table (which shows the
> single-channel source) describe two different experiments. The honest resolution is
> to present both conventions everywhere, and to state the sign of the result under
> each. The paper is admirably explicit about this tension in the discussion, but the
> abstract, introduction and conclusion quote only the colour convention while citing
> a table that shows the other.

**Response.** We agree that the earlier presentation used two incompatible
presentations. The primary table now reports the colour-scored visible source and the
fused inputs in the detector's native colour convention. The pooled audit reports
both conventions, with separate 42-test BH families. Table 6 summarizes the sign and
resolution under each source convention. Table 7 gives the per-input single-channel
and colour values with q values.

Abstract, Results and Conclusion now identify which convention and protocol produce
each reported range. The single-channel result is described as a sensitivity
analysis, not as support for a universal negative claim.

**Manuscript change.** Abstract; Sections 4 and 5; Tables 5 to 7; pp. 1 and 5 to 7.

## Comment 13

> The second benchmark is reported as measured without colour rescoring, so the
> replication does not carry the same convention as the primary result. The
> limitations state that the second benchmark (aligned FLIR-ADAS) “has no
> colour-corrected rescoring, because its detectors were trained with a different
> budget and their weights are no longer available, so it is reported as measured.”
> This means the replication of the headline claim on the second benchmark is in the
> single-channel convention, i.e. the convention in which the primary result is
> weakest. The replication therefore does not replicate the headline result as
> stated. Please clarify in the main text which convention Table 4 uses, and state
> whether the “no fused input beats the visible source” statement holds on the second
> benchmark under the same convention as the first.

**Response.** We agree with the limitation and now state it at the point of the
result. Table 8 is explicitly labelled as single-channel. The FLIR checkpoints are
unavailable, so colour-rescoring cannot be performed. The benchmark replicates the
negative sign under the measured convention, but it does not replicate the
colour-convention magnitude. Section 4 and the Discussion state this distinction.

We therefore do not claim that the two benchmarks are identical measurements. The
FLIR result is a directional replication with a stated convention boundary.

**Manuscript change.** Section 4; Table 8, p. 10; Discussion, p. 19.

## Comment 14

> The claim that “the penalty is set by the fused rule rather than by the task
> difficulty or the detector family” is not fully supported. Section 8 asserts this,
> but the evidence shows the penalty varying materially with the detector: SPGFusion
> loses 0.0993 under YOLO11n and 0.0919 under YOLO26n (Table 3) but only 0.0519 under
> RT-DETR-l, and the paper itself notes “the deficits under RT-DETR-l are the mildest
> of the three architectures.” Table 6 shows the probe-against-mean contrast with
> disjoint four-seed ranges across the two YOLO families. The correct statement is
> that the sign is set by the rule plus the evaluator’s training modality, while the
> magnitude varies with the architecture by a factor of about two. Please revise this
> sentence to match the data.

**Response.** We agree with the correction. The sentence has been replaced by a
statement that the sign is determined by the fusion rule and evaluator training
modality, while the magnitude varies with architecture. The text uses the clean
deficit ranges and the pooled SPGFusion contrasts to support this narrower claim.

We no longer attribute the penalty only to the fused rule or assume
architecture invariance.

**Manuscript change.** Discussion, pp. 19 to 20; Conclusion, p. 20.

## Comment 15

> The statistical framework is described as paired resampling but the family-wise
> error control is not specified. The manuscript states that “every interval in this
> paper resamples the same 300 images 1000 times and applies one draw to both arms of
> a contrast” and that “the multiplicity of the 42 headline contrasts is corrected.”
> The correction method is not named, and it is not stated whether the correction is
> applied within each of the four evaluator blocks, within each figure, or across the
> whole paper. Nor is it stated what the corrected criterion is. Since the paper’s
> argument is explicitly a measurement-discipline argument, the correction procedure
> must be stated precisely (method, family definition, and resulting adjusted
> threshold), and the corrected intervals should be printed in the tables rather than
> only in a supplement.

**Response.** The procedure is now named and defined. Paired bootstrap intervals use
the same 300-image resampling draw for the two sides of each contrast. Multiplicity
is controlled with Benjamini-Hochberg at a false-discovery rate of 0.05. The pooled
visible-trained family contains 14 inputs under each of three evaluators, giving 42
contrasts under each source convention. The two conventions are separate families.
Single-run clean evaluators receive separate 14-test BH families, and the clean
six-stratum analysis uses 14 by 6 comparisons per evaluator.

Table 5 reports the clean single-run q values through its caption and the
supplementary per-input table. Table 7 prints the pooled single-channel and
colour-scored q values for the two YOLO evaluators. RT-DETR-l per-input q values are
in Supplementary Table S23.

**Manuscript change.** Section 2; Sections 4 and 6; Tables 5 to 7; Supplementary
Sec. 16 and Table S23.

## Comment 16

> The 0.023 lower bound of the deficit range is never checked against the seed
> spread in the same table. Table 6 reports the four-seed ranges for the two YOLO
> families, e.g. probe − VIS at [−0.043, −0.003] under YOLO11n and [−0.028, −0.008]
> under YOLO26n, and top-hat − mean at [−0.002, +0.025]. The smallest headline
> deficit is stated as 0.023, but Table 6 shows that the seed range for at least one
> related contrast spans zero (probe − mean +0.005 [−0.002, +0.014]; top-hat − mean
> +0.006 [−0.002, +0.025]). Please state, for the specific rule that produces the
> 0.023 lower bound, its four-seed range, so the reader can verify that the smallest
> headline deficit exceeds the seed noise. The paper is careful about this for some
> contrasts and not for others.

**Response.** We agree that seed noise must be visible for the relevant contrast. The
current manuscript separates the pooled and clean lower bounds. In the pooled
colour audit, the 0.014 minimum is the activity-only RT-DETR-l three-run mean, whose
three run values are -0.013, -0.018 and -0.011. In the clean split, the visible-
trained deficits begin at 0.046, while the modality-agnostic reversal begins at
+0.022.

Table 10 reports seeds 0/1/2/3, the four-seed mean and the full range for the six
architecture contrasts. Table 5 reports the full range across independent clean
repeats. The text no longer attaches one lower bound to a different protocol.

**Manuscript change.** Tables 5 and 10; Sections 4, 6 and 7.

## Comment 17

> The segmentation experiment is barely reported. The manuscript states that the
> evaluation uses “two segmentation networks” and Table 14 includes an “mIoU” column
> under “the modality-agnostic segmenter” (mean 0.1225, top-hat 0.1270, probe 0.1270,
> ensemble 0.1269). An mIoU of 0.12 is very low for a segmentation metric, and the
> manuscript does not state what is being segmented, on which dataset, with which
> network, or how this result relates to the paper’s argument. Either report the
> segmentation analysis properly (network, dataset, classes, metric definition) or
> remove it; as it stands it invites the reader to doubt the authors’ measurement
> care, which is the opposite of the paper’s intent.

**Response.** We retained the analysis and documented the protocol. The network is a
lightweight U-Net with one input channel. It is trained on MSRS training pairs and
evaluated on the 361 MSRS-test pairs with eight foreground classes plus background.
The main table uses the strict nine-class convention, which assigns zero to a class
absent from both prediction and label. This accounts for the low absolute mIoU.

The usual present-classes convention gives an mIoU near 0.30 and reverses the
visible and infrared ranking. The text reports both conventions and uses
segmentation as a third downstream ranking, not as the primary claim.

**Manuscript change.** Section 6; Fig. 3c; Supplementary Sec. 17; pp. 9 and 17.

## Comment 18

> The manuscript never states the computational cost of the proposed rules in the
> main text. Section 2 gives the probe at “0.07 M parameters and 9.4 ms per 512 × 384
> pair,” and the SI is said to contain the “inference cost.” Since one of the paper’s
> implicit arguments is that the parameter-light targetness rules are a reasonable
> design point, the cost of all four of the authors’ rules and of the four reproduced
> baselines should be tabulated in the main text (parameters, latency, and memory),
> under the same measurement protocol. This matters for the practitioner the paper
> addresses.

**Response.** Table 4 now reports parameters, latency and extra peak GPU memory for
the four author rules and four reproduced baselines under one protocol: 512 by 384,
FP32, batch size one and an RTX 2080 Ti. The caption states the warm-up and repeat
convention and notes that SPGFusion also requires DINO and CLIP feature extraction.

The earlier 9.4 ms value is no longer used as the main comparison.

**Manuscript change.** Table 4, p. 6; Sections 5 to 6.

## Comment 19

> The paper does not examine whether the negative result is an artifact of the
> registration or the annotation. Both benchmarks are registered VIS–IR pairs with
> COCO-style person annotations. The paper’s negative result could in principle
> partly reflect residual misregistration between the modalities at the annotation
> level (the annotations are drawn on one modality and used for both), or annotation
> conventions that favour visible appearance. The manuscript mentions neither
> possibility. Given that the recommended conclusion is “do not fuse for a
> visible-trained detector,” this alternative explanation should be ruled out, at
> least by reporting the registration procedure and any alignment-error estimate, and
> by reporting the contrast for the subset of images with the best alignment.

**Response.** We agree that registration and annotation are alternative explanations.
We now report the dataset's spatial alignment procedure and an independent
translational ECC residual for all 300 LLVIP pairs. The median residual is 2.53 px
and the best-quartile cutoff is 1.61 px.

The best-aligned clean-split quartile reverses the sign for mean fusion, top-hat and
the probe in all four clean runs, while the worst-aligned quartile contains the
largest deficits. This analysis also shows that alignment co-varies with contrast
and target size, so it does not isolate a causal registration effect. We report it
as a sensitivity analysis and include shared annotations as a limitation.

**Manuscript change.** Section 2; Section 6; Discussion, pp. 19 to 20;
Supplementary Sec. 22.

## Comment 20

> Several internal inconsistencies, undefined terms and reporting defects.
>
> • The abstract says “fourteen fused variants, fourteen detectors and two
> benchmarks,” but Table 3 lists twelve inputs (two sources plus ten fused rows
> including the four permutations) and the text elsewhere refers to “the same twelve
> inputs.” Please reconcile the counts.
>
> • The keyword list under the title (“image fusion, infrared, visible, saliency,
> targetness”) differs from the submission-system keywords (“infrared and visible
> image fusion, fidelity metrics, evaluation protocol, object detection, multimodal
> pedestrian detection”); unify.
>
> • Section 5 refers to “the eight variants whose fused outputs were retained”
> without stating the retention criterion or why outputs were discarded from a
> study whose stated deliverable is complete artifacts.
>
> • Fig. 2a is described as “42 contrasts” (3 visible-trained architectures × 14
> inputs) but the abstract lists “fourteen fused variants”; 14 × 3 = 42 only if the
> two sources are excluded, so the count of “fourteen” and the count of “42” are
> inconsistent. State the arithmetic.
>
> • The manuscript refers to “our own rules” and “the reported index” without naming
> the index until Section 3; the index should be named and defined at first use.
>
> • Table 1, Table 2 and Table 14 report VIF for the same rules on the same datasets,
> but Table 1’s top-hat TNO value (0.3561) and Table 2’s identical value appear to be
> the same measurement reported twice with different comparison sets; make clear
> whether these are the same numbers.
>
> • The manuscript says “the two end-to-end networks dominate pixel fidelity” but
> does not state that IFCNN (0.4440 on TNO) is beaten by SPGFusion (0.8135) and
> PIAFusion (0.6051) but beats DenseFuse (0.3697) and both of the authors’ rules, the
> sentence should name the networks.
>
> • Figure captions and the algorithm listing are deferred to the supplement; at
> least Fig. 1’s caption in the manuscript should fully specify the pipeline without
> requiring the supplement.

**Response.** We addressed these reporting defects as follows.

1. The new evaluator inventory in Table 1 states the arithmetic. The study has 14
   fused inputs and two source images. Detection uses 26 training checkpoints plus a
   separate U-Net segmentation checkpoint. The 42 headline contrasts are 14 fused
   inputs under each of three clean visible-trained evaluators.
2. The manuscript keywords now match the submission-system keywords.
3. The complete 14-input fidelity and detection comparison is Table 11. The
   reference VIF was fully rescored, and the legacy 13-input scatter is explained as
   a historical export rather than a retention rule for the main table.
4. Figure 2 and its caption now define the 3 by 14 arithmetic.
5. The reported VIF construction is named and defined in Section 3, and the
   reference construction is defined alongside it.
6. The caption of Table 3 states that its VIF measurements are the same values shown
   for those rows in Table 2.
7. The fidelity discussion now names SPGFusion, PIAFusion, IFCNN and DenseFuse and
   states the pairwise order rather than referring only to “end-to-end networks.”
8. The Figure 1 caption now defines the encoder, targetness maps, activity maps,
   normalized weights, temperature softmax and decoder without requiring the
   supplement.

**Manuscript change.** Front matter; Table 1; Table 2 caption; Table 3; Fig. 1
caption; Fig. 2 caption; Table 11; sections 3 to 6.

## Minor Comment 1

> Define VIF, QAB/F, SSIM, mAP50, mAP50-95, mIoU and ANFIS-style fidelity indices at
> first use.

**Response.** The required definitions are now supplied at first use. The
manuscript defines VIF, SSIM, EN, PSNR, mAP50, mAP50-95 and mIoU, and identifies
QAB/F and ANFIS-style indices as indices from the cited literature rather than
reimplementing them under a different definition.

**Manuscript change.** Abstract; Section 2; Section 3; Table 2 caption.

## Minor Comment 2

> State the LLVIP training/validation/test split explicitly in the main text; the
> reference to “1500 visible LLVIP training pairs” and “the same 300 test images”
> should be accompanied by the official split sizes.

**Response.** The main text now gives the official LLVIP partition and the selected
experiment split. The official release has 15,488 pairs in partitions of 12,014
training and 3,474 test pairs. The fixed experiment split uses 1,350 training, 150
validation and 300 test pairs, with zero stem overlap.

**Manuscript change.** Section 2; Table 1; Section 4.

## Minor Comment 3

> State the number of images in the FLIR-ADAS aligned release used here relative to
> the full release.

**Response.** The manuscript states that the aligned FLIR release contributes 4,129
training and 1,013 validation pairs. The same selection procedure gives 1,500
training and 300 test pairs for the reported experiment.

**Manuscript change.** Section 2; Table 8 caption.

## Minor Comment 4

> Give the number of pedestrians per image distribution, since the stratification
> analysis depends on it.

**Response.** The distribution is now reported. The 300 LLVIP test images contain
744 pedestrians. Counts are 98 images with one pedestrian, 90 with two, 54 with
three, 25 with four, 15 with five, 8 with six, 2 with seven, 5 with eight and 3 with
nine.

**Manuscript change.** Section 2; evaluation-protocol paragraphs.

## Minor Comment 5

> The phrase “resolved” (as in “all forty-two contrasts are resolved”) is used
> throughout as a technical term for “the interval excludes zero”; define it once.

**Response.** The manuscript now defines a contrast as resolved when its 95 percent
paired bootstrap interval excludes zero, or when the specified BH-adjusted q value
is below 0.05 for analyses that use multiplicity correction.

**Manuscript change.** Section 2, evaluation protocol.

## Minor Comment 6

> Section 4 states that the RT-DETR-l column changes “architecture and optimisation
> budget together”; consider adding a budget-matched architecture comparison to
> separate the two factors, or state explicitly that they are confounded.

**Response.** We state the confound explicitly. RT-DETR-l uses batch size 4 rather
than 16 to fit the available GPU and differs from the YOLO models in architecture,
optimizer and matching procedure. We do not interpret the RT-DETR-l difference as a
pure architecture effect.

**Manuscript change.** Section 2; Discussion limitation, pp. 19 to 20.

## Minor Comment 7

> The stratification is reported as “top contrast tercile, ≥3 pedestrians, small
> targets”; give the numeric thresholds for contrast and target size.

**Response.** The thresholds are now numeric. The visible RMS contrast boundaries
are 0.0830 and 0.1953, the small-target cutoff is 26.9022 pixels, and the crowded
condition is three or more pedestrians per image.

**Manuscript change.** Section 4; Fig. 3 description.

## Minor Comment 8

> Table 4 should state in its caption which convention the visible source is scored
> in.

**Response.** The equivalent FLIR table is now Table 8, and its caption states that
the values use the single-channel convention. The main detection tables also state
the source convention and distinguish colour-scored from single-channel families.

**Manuscript change.** Table 5 caption; Table 8 caption.

## Minor Comment 9

> The discussion’s statement that the visible source’s validation split “is taken
> from the same 300 images that are scored” is a significant caveat; move it into
> Section 4 so that the reader meets it with the result.

**Response.** The caveat is now stated where the pooled audit is introduced. The
text identifies the validation overlap before presenting the optimistic results,
and the clean split is explicitly designated as the primary comparison.

**Manuscript change.** Section 4, p. 5 and Table 6 caption.

## Minor Comment 10

> Reference to “the detector library’s own metric implementation” should name the
> library and version.

**Response.** The manuscript names Ultralytics 8.4.189 and gives the confidence
threshold, IoU matching rule and 101-point AP interpolation convention.

**Manuscript change.** Section 2, evaluation protocol.

## Minor Comment 11

> The Zenodo DOI should be a formal data/code availability statement in the
> manuscript, not only in the cover letter.

**Response.** The Data availability statement now gives the public Zenodo DOI,
`10.5281/zenodo.22984389`, and identifies the released data and analysis artifacts.

**Manuscript change.** Data availability section, p. 21.

## Minor Comment 12

> The AI-assisted language-editing declaration in the cover letter should also
> appear in the manuscript (the letter says “the declaration in the manuscript states
> this”, verify that it does).

**Response.** The manuscript now includes the required declaration and limits it to
language editing. The authors remain responsible for the scientific content.

**Manuscript change.** Declaration of generative AI and AI-assisted technologies,
p. 21.
