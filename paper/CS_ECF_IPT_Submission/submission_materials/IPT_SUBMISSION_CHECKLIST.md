# Infrared Physics & Technology Submission Checklist

Date: 2026-10-28

Target: Infrared Physics & Technology (Elsevier)

## Submission readiness

**Ready with author checks.** The manuscript, supplement, cover letter,
Highlights, figures, source data, declarations, and reproducibility evidence
are assembled. The final upload requires confirmation of the inherited
administrative facts and a check of the live IPT portal prompts.

The code revision is frozen at GitHub tag
`cs-ecf-ipt-submission-2026-10-28` (commit `2d53dd1`).

Reasons:

- the administrative declarations were inherited from the existing manuscript
  family and require final author confirmation;
- the ScienceDirect author-guide page returned a CAPTCHA during automated
  retrieval, so exact portal labels and any display-item prompts must be checked
  at upload;
- the 27-entry bibliography consists of verified journal, conference, dataset,
  and software records; further synthesis would improve breadth but is not a
  submission blocker.

## Deliverable matrix

| Item | Status | Source or missing input | Draft/output |
|---|---|---|---|
| Main manuscript | required | compiled working draft | `CS_ECF_IPT_Submission/main.tex` and `main.pdf` |
| Title page | required | author metadata imported from existing drafts | compiled title page in `main.pdf` |
| Cover letter | required/useful | result facts supplied | `IPT_COVER_LETTER_DRAFT.md`; compiled `cover_letter.pdf` |
| Highlights | verify IPT requirement | five verified-result bullets | `IPT_HIGHLIGHTS.txt` |
| Keywords | required | supplied | `IPT_KEYWORDS.txt` |
| Main figures | required | four figures exported and QA'd | `figures_cs_ecf_ipt/` |
| Tables | required | included in compiled draft | `CS_ECF_IPT_Submission/main.tex` |
| Supplementary information | required if referenced | compiled | `supplementary.tex` and `supplementary.pdf` |
| External evaluator checks | required supporting evidence | two zero-shot COCO models and one LLVIP-trained YOLOv9t | `INDEPENDENT_EVALUATOR_CHECK.md`; `results/independent_evaluators/` |
| Data availability | required | inherited and adapted for the current datasets | compiled in `main.tex` |
| Code availability | required | inherited GitHub/Zenodo/MIT statement | compiled in `main.tex` |
| CRediT contributions | required | inherited from existing drafts | compiled in `main.tex` |
| Competing interests | required | inherited no-conflict statement | compiled in `main.tex` |
| Funding | required | inherited no-specific-grant statement | compiled in `main.tex` |
| Related/preprint disclosure | required | inherited originality/status statement | compiled cover letter; no separate related list found |
| Suggested reviewers | portal-dependent | candidate identities not supplied | omit unless requested |
| Graphical abstract | portal-dependent | none supplied | submit only if requested |

## Completed evidence

- LLVIP 8-checkpoint confirmatory test;
- M3FD 2-checkpoint grouped six-class test;
- DroneVehicle 2-checkpoint independent official-test subset;
- DroneVehicle source archive size/hash and fixed-group manifest retained;
- dataset-balanced AP50 gain `+0.0133`;
- training-run-by-group hierarchical CI `[+0.0026,+0.0261]`;
- no-shield worst loss `-0.0584`, shield worst loss `-0.0040`;
- class-aware M3FD analysis showing a people-specific effect;
- thermal-top-hat stratification and thermal-mask/global ablation;
- standardized serial latency estimate `706.3 ms`;
- paired CS-ECF versus Always CDF comparison with direct intervals;
- independent zero-shot YOLOv9t/YOLOv10s confirmation;
- LLVIP-fine-tuned YOLOv9t negative boundary with frozen final checkpoint;
- verified 2024–2026 IPT fusion/detection exemplars in `IPT_TARGETING.md`.

## Cross-file consistency checks

The following canonical values must remain identical everywhere:

| Metric | Canonical value |
|---|---:|
| Confirmatory checkpoints | 12 |
| Positive checkpoints | 9 |
| Dataset-balanced AP50 gain | +0.0133 |
| Training-run-by-group hierarchical CI | [+0.0026,+0.0261] |
| No-shield worst AP50 loss | -0.0584 |
| CS-ECF worst AP50 loss | -0.0040 |
| DroneVehicle best gain | +0.0251 |
| DroneVehicle last gain | +0.0293 |
| DroneVehicle group CI | [+0.0002,+0.0406] |
| Thermal-mask advantage over global | +0.0007 |
| Serial latency estimate | 706.3 ms |
| CS-ECF minus Always CDF, balanced | +0.0102 |
| Direct-comparison hierarchical CI | [-0.00057,+0.02980] |
| Zero-shot gain over visible | +0.0213 to +0.0250 AP50 |
| Zero-shot gain over CDF | -0.0059 to +0.0060 AP50 |
| LLVIP-trained YOLOv9t minus visible | -0.0108 AP50 |
| LLVIP-trained YOLOv9t scene-group CI | [-0.0342,+0.0050] |
| LLVIP-trained YOLOv9t shield gain over no-shield | +0.0153 AP50 |

## Author checks

- Confirm Research Paper against the current IPT author guide.
- Confirm continued use of the extracted author order, affiliations, emails,
  ORCIDs, CRediT roles, no-specific-grant funding, and no-competing-interest
  statements for this new manuscript.
- Confirm the named AI tool and final AI declaration wording.
- Confirm that the cover-letter assertion “not previously published / not under
  consideration” is complete; the reviewed drafts contain no separate preprint
  identifier or related-manuscript list.
- Confirm that the frozen GitHub submission tag remains publicly accessible;
  do not reuse the preceding study's Zenodo DOI.
- Add reviewer suggestions only if the IPT portal requests them.

## Next actions

1. Complete the four author checks above.
2. Open the live IPT submission portal and confirm article type, portal labels,
   abstract/display limits, and optional items.
3. Confirm that the frozen GitHub submission tag remains accessible.
4. Recompile both PDFs and re-run numerical, reference, placeholder, and visual
   QA.
5. Regenerate the submission ZIP and SHA-256 manifest.
