# CS-ECF IPT Working Draft

Date: 2026-10-28  
Target: *Infrared Physics & Technology*  
Status: compiled working draft with inherited administrative metadata; ready
with author checks, not yet journal-submission ready

## Main files

- `main.tex` — compiled Elsevier preprint draft.
- `main.pdf` — current rendered working draft.
- `cover_letter.tex` and `cover_letter.pdf` — compiled editor-facing draft.
- `references.bib` — 11 verified entries: recent IPT positioning papers plus
  method and dataset sources. Broader domain synthesis remains required.
- `REFERENCE_VERIFICATION.md` — DOI-backed verification ledger.
- `figures/` — four publication figures in PDF form.
- `source_data/` — figure-level CSV source data.
- `submission_materials/` — cover-letter, Highlights, keyword, caption,
  checklist, and figure-QA drafts.

## Current verified result anchors

- 12 confirmatory detector checkpoints, 9 positive.
- Dataset-balanced AP50 gain: +0.0133.
- Model-by-group hierarchical 95% CI: [+0.0028, +0.0213].
- No-shield worst loss: -0.0584; CS-ECF worst loss: -0.0040.
- DroneVehicle best/last gains: +0.0251 / +0.0293.
- Thermal mask advantage over global injection: +0.0007 AP50.
- Serial latency estimate: 706.3 ms.

## Before submission

1. Expand and integrate the bibliography beyond the current verified seed set.
2. Confirm the inherited author, affiliation, funding, competing-interest,
   CRediT, repository, and expanded AI-use statements.
3. Verify the current IPT Guide for Authors, including article type, abstract
   limit, Highlights requirements, figure count, and artwork rules.
4. Update the Zenodo record from the submission tag.
5. Re-run the LaTeX build, figure QA, PDF page audit, and numerical consistency
   checks after the placeholders are resolved.

