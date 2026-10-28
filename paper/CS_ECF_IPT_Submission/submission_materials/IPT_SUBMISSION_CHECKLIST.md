# Infrared Physics & Technology Submission Checklist

Date: 2026-10-28

Target: Infrared Physics & Technology (Elsevier)

## Submission readiness

**Author-confirmed; live portal check pending.** The
manuscript, supplement, cover letter, Highlights, figures, source data,
declarations, and reproducibility evidence are assembled and have passed the
final local technical QA. The author has confirmed that the inherited
administrative facts and declaration text remain unchanged from the original.
The final upload requires a check of the live IPT portal prompts.

The code revision is frozen at GitHub tag
`cs-ecf-ipt-submission-2026-10-28` (commit `2d53dd1`).

Reasons:

- the administrative declarations were confirmed by the author on 2026-10-28 as
  unchanged from the original;
- the ScienceDirect author-guide page returned a CAPTCHA during automated
  retrieval, so exact portal labels and any display-item prompts must be checked
  at upload;
- the GitHub tag is remotely accessible and resolves to the documented commit;
- the 27-entry bibliography consists of verified journal, conference, dataset,
  and software records; further synthesis would improve breadth but is not a
  submission blocker.

## Technical gate

| Gate | Status | Evidence |
|---|---|---|
| LaTeX build | complete | `main.pdf`, `supplementary.pdf`, `cover_letter.pdf` |
| Numerical/reference audit | complete | `FINAL_TECHNICAL_QA.md` |
| PDF visual and font audit | complete | `FINAL_TECHNICAL_QA.md` |
| GitHub submission tag | complete | remote tag resolves to `2d53dd1` |
| Zenodo archival release | complete | `10.5281/zenodo.23718751`, version `v2.0.0` |
| Author administrative facts | complete | author confirmed unchanged on 2026-10-28 |
| Live IPT portal prompts | pending | CAPTCHA/login blocks unattended check |

## Deliverable matrix

| Item | Status | Source or missing input | Draft/output |
|---|---|---|---|
| Main manuscript | required | compiled working draft | `CS_ECF_IPT_Submission/main.tex` and `main.pdf` |
| Title page | required | author metadata imported from existing drafts | compiled title page in `main.pdf` |
| Cover letter | required/useful | result facts supplied | `IPT_COVER_LETTER_DRAFT.md`; compiled `cover_letter.pdf` |
| Highlights | verify IPT requirement | five verified-result bullets | `IPT_HIGHLIGHTS.txt` |
| Keywords | required | supplied | `IPT_KEYWORDS.txt` |
| Main figures | required | four figures exported and QA'd | `figures/` |
| Tables | required | included in compiled draft | `CS_ECF_IPT_Submission/main.tex` |
| Supplementary information | required if referenced | compiled | `supplementary.tex` and `supplementary.pdf` |
| External evaluator checks | required supporting evidence | two zero-shot COCO models and one LLVIP-trained YOLOv9t | `INDEPENDENT_EVALUATOR_CHECK.md`; `results/independent_evaluators/` |
| Data availability | required | inherited and adapted for the current datasets | compiled in `main.tex` |
| Code availability | required | GitHub plus CS-ECF Zenodo/MIT statement | compiled in `main.tex` |
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
- three PDFs rebuilt and visually inspected after the final source update;
- 26 of 26 DOI records resolved through Crossref; the software entry has no DOI;
- 35 archived pre-QA files matched the v9 SHA-256 manifest before the
  administrative packages were assembled.
- Zenodo version `v2.0.0` published as `10.5281/zenodo.23718751` under the
  shared software concept DOI `10.5281/zenodo.22984388`.

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

## Author confirmation record

On 2026-10-28, the author confirmed retention of the original administrative
and declaration content without changes:

- author order, affiliations, emails, ORCIDs, CRediT roles, no-specific-grant
  funding, and no-competing-interest statements;
- the named OpenAI Codex AI tool and final AI declaration wording;
- the cover-letter “not previously published / not under consideration”
  assertion and its related-manuscript disclosure;
- Research Paper as the intended article type.

- Add reviewer suggestions only if the IPT portal requests them.

The GitHub tag and CS-ECF Zenodo version DOI are already verified. Do not reuse
the preceding study's version DOI `10.5281/zenodo.22984389`.

## Next actions

1. Open the live IPT submission portal with an interactive browser and confirm
   article type, portal labels,
   abstract/display limits, and optional items.
2. Upload the files from `submission_package_v12/`.
3. If the portal requires any scientific or administrative change,
   recompile the PDFs and regenerate the manifest before upload.
