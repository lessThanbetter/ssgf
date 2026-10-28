# CS-ECF IPT Working Draft

Date: 2026-10-28  
Target: *Infrared Physics & Technology*  
Status: author-confirmed working draft with original administrative metadata;
ready for the live submission-portal check after completed technical QA and
Zenodo archival release

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
- `figures/` — four publication figures in PDF form.
- `source_data/` — figure-level CSV source data.
- `submission_materials/` — cover-letter, Highlights, keyword, caption,
  checklist, and figure-QA drafts.
- `submission_package_v13/` — final assembled upload package with PDFs, LaTeX,
  figures, source data, external-evaluator evidence, reproduction scripts, and
  SHA-256 manifest.
- `submission_materials/FINAL_TECHNICAL_QA.md` — completed build, consistency,
  reference, PDF, and archive checks.
- `submission_materials/RELATED_WORK_EVIDENCE.md` — section-to-reference
  mapping and audit for the added literature.
- Zenodo version DOI: `10.5281/zenodo.23718751`.

## Current verified result anchors

- 12 confirmatory detector checkpoints, 9 positive.
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
3. Upload from `submission_package_v13/`. The CS-ECF Zenodo version DOI is
   `10.5281/zenodo.23718751`; the preceding study's DOI must not be reused.
4. Re-run the LaTeX build, figure QA, PDF page audit, and numerical consistency
   checks again only if the portal requires a content change.

