# Measurement submission package

## Positioning

- Article type: Research Paper.
- Contribution: measurement validity of fidelity scores for downstream
  visible-infrared fusion evaluation, supported by a test-independent protocol.
- Primary claim: under visible-trained evaluators, every fused input loses to the
  unfused visible source at mAP50; under a modality-agnostic evaluator the sign reverses.
- Claim boundary: the paper does not introduce a new fusion architecture and does not
  claim that fusion is universally harmful.

## Files

| File | Purpose | Status |
|---|---|---|
| `PAPER_IPT_v3.tex` | Internal source filename retained for build stability | Retargeted to `Measurement` |
| `PAPER_MEASUREMENT_v5.pdf` | Standalone main-manuscript preview | Compiled, 23 pages |
| `Supplementary_Material_MEASUREMENT.pdf` | Standalone supplementary preview | Compiled, 37 pages |
| `measurement_submission/Manuscript.pdf` | Portal upload: main manuscript | Compiled, 23 pages |
| `measurement_submission/Supplementary_Material.pdf` | Portal upload: supplementary material | Compiled, 37 pages |
| `measurement_submission/Highlights.txt` | Portal upload: plain-text highlights | Ready |
| `measurement_submission/Highlights.docx` | Portal upload: editable highlights | Generated and visually checked |
| `measurement_submission/Cover_Letter.pdf` | Portal upload: editor cover letter | Compiled, 1 page |
| `measurement_submission/Figures_SVG.zip` | Portal upload: editable vector figures | Ready |

## Highlights

Each item is at most 85 characters including spaces.

1. Fidelity scores are used to choose fusion rules without downstream validation.
2. Test-independent splits retrain detectors with all 300 test images held out.
3. All 42 clean visible-detector comparisons favor the unfused visible image.
4. Deficits span 0.046 to 0.246 mAP50; both one-run evaluators survive BH correction.
5. Evaluator modality reverses gains, so rankings require measurement context.

## Evidence checks

- Main manuscript: 23 pages, 11 figures, 14 tables, 47 references.
- Supplementary material: 37 pages, 29 tables.
- Compilation: no errors, undefined citations, undefined references or overfull boxes.
- Figure set: Figure 1 was redrawn in the original two-stage visual language;
  Figures 2-10 use the revised vector system; Figure 11 is retained.
- Fixed split: 1,350 training, 150 validation and 300 test images.
- Pairwise split overlap: zero.
- Test annotations: 744 pedestrians.
- Clean evaluator coverage: YOLO11n, YOLO26n, RT-DETR-l and modality-agnostic YOLO11n.
- Fused-input coverage: 14 inputs plus both sources under each evaluator.
- Multiplicity: separate 14-test BH families for each single-run clean evaluator.
- Run: `python scripts/verify_clean_r2.py --root .`

## Author actions before portal submission

- Select the final article type and any journal-specific reviewer suggestions in the
  submission portal.
- Upload the main PDF, supplementary PDF, highlights DOCX and cover-letter PDF.

The authors confirmed on 27 October 2026 that the archived data DOI is public and that
the author list, affiliations, funding, competing interests and AI-use declaration are
correct.

