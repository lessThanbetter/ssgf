# CS-ECF IPT Working Draft

Date: 2026-10-28  
Target: *Infrared Physics & Technology*  
Status: compiled working draft with inherited administrative metadata; ready
with author checks, not yet journal-submission ready

## Main files

- `main.tex` — compiled Elsevier preprint draft.
- `main.pdf` — current rendered working draft.
- `supplementary.tex` and `supplementary.pdf` — complete split manifests,
  per-checkpoint results, direct-baseline comparison, validation record, and
  thermal controls.
- `cover_letter.tex` and `cover_letter.pdf` — compiled editor-facing draft.
- `references.bib` — 25 verified entries covering recent IPT positioning,
  representative classical and deep fusion families, detectors, uncertainty,
  and risk control. Broader domain synthesis remains desirable.
- `REFERENCE_VERIFICATION.md` — DOI-backed verification ledger.
- `figures/` — four publication figures in PDF form.
- `source_data/` — figure-level CSV source data.
- `submission_materials/` — cover-letter, Highlights, keyword, caption,
  checklist, and figure-QA drafts.

## Current verified result anchors

- 12 confirmatory detector checkpoints, 9 positive.
- Dataset-balanced AP50 gain: +0.0133.
- Training-run-by-group hierarchical 95% CI: [+0.0026, +0.0261].
- Direct comparison with Always CDF: +0.0102 AP50;
  run-cluster-by-group 95% CI [-0.00057, +0.02980], including zero.
- LLVIP direct CDF contrast: -0.0057; M3FD +0.0147; DroneVehicle +0.0217.
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
4. Create a new Zenodo version for the CS-ECF submission. The current DOI still
   resolves to the preceding fidelity-metric study and must not be cited as the
   archive of this manuscript.
5. Re-run the LaTeX build, figure QA, PDF page audit, and numerical consistency
   checks after the final author metadata confirmation.

