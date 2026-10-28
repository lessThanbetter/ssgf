# CS-ECF IPT Working Draft

Date: 2026-10-28  
Target: *Infrared Physics & Technology*  
Status: 11-figure redesign and language polish complete; v18 package and Zenodo `v2.4.1`
archival release prepared for the live submission-portal check

## Main files

- `main.tex` — compiled Elsevier preprint draft.
- `main.pdf` — current rendered working draft.
- `supplementary.tex` and `supplementary.pdf` — complete split manifests,
  per-checkpoint results, direct-baseline comparison, validation record, and
  thermal controls.
- `cover_letter.tex` and `cover_letter.pdf` — compiled editor-facing draft.
- `references.bib` — 46 verified entries covering classical, sparse,
  generative, encoder-decoder, transformer, diffusion, task-aware,
  multispectral, detector, uncertainty, and risk-control literature.
- `REFERENCE_VERIFICATION.md` — DOI-backed verification ledger.
- `figures/` — 11 publication figures in PDF form.
- `source_data/` — 11 figure-level CSV source data files.
- `submission_materials/` — cover-letter, Highlights, keyword, caption,
  checklist, and figure-QA drafts.
- `submission_package_v18/` — final assembled upload package with PDFs, LaTeX,
  figures, source data, external-evaluator evidence, reproduction scripts, and
  SHA-256 manifest; includes the multiplicity, sensitivity, baseline, class,
  calibration, registration, PDF-text, and field-level reference audits.
- `submission_materials/FINAL_TECHNICAL_QA.md` — completed build, consistency,
  reference, PDF, and archive checks.
- `submission_materials/RELATED_WORK_EVIDENCE.md` — section-to-reference
  mapping and audit for the added literature.
- Zenodo version DOI for `v2.4.1`: `10.5281/zenodo.23728385`; concept DOI:
  `10.5281/zenodo.22984388`.

## Current verified result anchors

- 12 confirmatory detector checkpoints, 9 positive.
- Exact two-sided sign test for 9/12 positive checkpoints: `p=0.073`; no
  individual checkpoint survives Holm correction across the 12-member family.
- Dataset-balanced AP50 gain: +0.0133.
- Training-run-by-group hierarchical 95% CI: [+0.0026, +0.0261].
- Direct comparison with Always CDF: +0.0102 AP50;
  run-cluster-by-group 95% CI [-0.00057, +0.02980], including zero.
- LLVIP direct CDF contrast: -0.0057; M3FD +0.0147; DroneVehicle +0.0217.
- Zero-shot YOLOv9t: CS-ECF +0.0250 over visible and +0.0060 over CDF.
- Zero-shot YOLOv10s: CS-ECF +0.0213 over visible and -0.0059 over CDF.
- LLVIP-trained YOLOv9t boundary: CS-ECF -0.0108 over visible and -0.0040 over
  CDF; both scene-group intervals crossed zero.
- The fine-tuned YOLOv9t shield improved the unshielded policy by +0.0153, but
  did not prevent a loss relative to visible.
- No-shield worst loss: -0.0584; CS-ECF worst loss: -0.0040.
- Uniform-random balanced gain: -0.0341; per-image oracle upper bound: +0.0833.
- Image-quality selectors do not recover the confidence-based gain:
  maximum-VIF +0.0008, maximum-SSIM -0.0946, maximum-entropy -0.0599.
- Validation selection used 150 LLVIP and 400 M3FD images; K=5/margin=1.5
  produced balanced validation gain +0.0008, worst-evaluator gain -0.0001, and
  a 4.7% switch fraction.
- Fine-tuned YOLOv9t ECE was 0.048 versus 0.009-0.010 for zero-shot evaluators;
  41.4% of its switched images degraded.
- M3FD people-class counts and intervals are reported; the people class
  accounts for 87.7% and 99.6% of the positive class-mean contribution in the
  two checkpoints.
- LLVIP ECC median translation residual was 2.53 pixels; the gain was not
  confined to the best-aligned quartile.
- The 300-image LLVIP test set is an explicit random-seed-0 subset of the
  larger official test partition; the 1,350/150 training and validation sets
  are the first 1,350 and final 150 of a lexicographically ordered 1,500-pair
  subset of official non-empty training labels.
- The LLVIP release used here has 3,463 non-empty official test pairs, so 3,163
  eligible pairs were not selected.
- The complete pre-freeze margin grid is reported; the selected margin 1.5 was
  the first feasible setting and the largest balanced gain among feasible
  settings.
- All 45 DOI records passed a Crossref field audit of title, authors, year,
  volume, and pages with zero unresolved differences.
- DroneVehicle best/last gains: +0.0251 / +0.0293.
- Thermal mask advantage over global injection: +0.0007 AP50.
- Serial latency estimate: 706.3 ms.

## Before submission

1. Author confirmation is complete: retain the original author, affiliation,
   funding, competing-interest, CRediT, repository, AI-use, and originality
   wording unchanged.
2. Verify the current IPT Guide for Authors in the submission portal, including
   article type, abstract
   limit, Highlights requirements, figure count, and artwork rules.
3. Upload from `submission_package_v18/`. The CS-ECF Zenodo `v2.4.1` version
   DOI is `10.5281/zenodo.23728385`; the preceding study's DOI
   `10.5281/zenodo.22984389` must not be reused.
4. Re-run the LaTeX build, figure QA, PDF page audit, and numerical consistency
   checks again only if the portal requires a content change.

