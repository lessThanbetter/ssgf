# Response to Reviewer Comments

**Previous manuscript:** Evaluator-Dependent Validity of Fidelity Metrics for
Visible-Infrared Image Fusion in Pedestrian Detection

**Previous submission:** INFPHY-D-26-02136

**Prior editorial decision:** Reject

We thank the reviewer for the direct assessment of structure, figure quality and the
scope of the contribution. We have treated the novelty concern as a positioning
problem rather than trying to disguise the existing method work as a new fusion
network. The revised manuscript presents the work as a measurement-validity and
evaluation-protocol study. Table and figure references below are to
`PAPER_MEASUREMENT_v5.pdf`.

## Comment 1

> The overall structure of the paper is rather disorganized, making it difficult to
> clarify the logic.

**Response.** We have reorganised the manuscript around four questions and moved the
claim logic to the first page of each result section.

The Introduction now states the measurement problem, the fixed-input comparison and
the two result dimensions: evaluator training modality and fidelity-index
construction. Section 3 defines the fidelity measurements. Section 4 begins with
the fully independent clean split and keeps the pooled, validation-overlapping
analysis as a sensitivity check. Section 5 then examines evaluator and index
reversal. Section 6 contains the supporting and robustness analyses. Section 7
presents adaptation, and Sections 8 and 9 separate the stable result, the remaining
limits and the scope of the conclusion.

Figure 2 now separates the clean primary comparison from the pooled audit in a
single visual sequence, and Figure 11 summarizes the final claim structure. Table 1
provides an evaluator inventory so that the evidence hierarchy is explicit before
the experiments are discussed.

**Manuscript change.** Whole manuscript; especially Table 1, Fig. 2, Figs. 4 and 5,
and Fig. 11.

## Comment 2

> The quality of the charts is poor.

**Response.** We rebuilt the quantitative figures as vector graphics with a common
typographic and colour system. Figure 1 retains its original two-stage framework
layout, but its labels, real-data target maps, weight maps and connections were
redrawn. Figures 2 to 10 now use Arial-based vector data plots with consistent
panel labels, axes, legends and direct comparisons. Figure 11 combines real image
examples, a detection-contrast matrix and the reported/reference VIF rank reversal.

All 11 main figures were checked in the compiled A4 PDF for caption placement,
readability and panel continuity. The complete figure set is supplied as editable
SVG files in the submission package.

**Manuscript change.** Figures 1 to 11; `Figures_SVG.zip`.

## Comment 3

> There is a lack of core innovation.

**Response.** We agree that the manuscript does not introduce a new fusion
architecture. The targetness-weighted family is used as a controlled set of fused
inputs, not as the paper's claimed novelty.

The contribution is now stated as a measurement-validity finding. With all 300 test
images held out, the same fused inputs change sign when the evaluator training
modality changes, and the same fidelity score family can place the same method first
or last when its construction is changed. Therefore, a fusion ranking must be
interpreted with its evaluator, index construction and test protocol rather than as
a property of the fused image alone.

The title and abstract now foreground evaluator-dependent measurement validity. The
limitation is explicit: the findings concern the measured pedestrian-detection
population, and the paper does not claim a universal law about all fusion or
detection systems.

**Manuscript change.** Title; abstract; Introduction; Discussion; Conclusion.
