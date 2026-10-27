# Infrared Physics & Technology Submission Checklist

Date: 2026-10-28  
Target: Infrared Physics & Technology (Elsevier)

## Submission readiness

**Ready with author checks for the identified manuscript; journal submission is
still blocked by the incomplete domain bibliography, an unversioned Zenodo
archive, and unverified current IPT instructions.**

Reasons:

- the bibliography is a verified 27-entry set rather than a full domain review;
- exact current IPT author-instruction limits and declarations still need
  verification;
- the administrative declarations were inherited from the existing manuscript
  family and require final author confirmation;
- the supplementary file is assembled and compiled, but must be rechecked after
  the final main-text changes.

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
| Independent evaluator check | internal evidence | two zero-shot COCO models | `INDEPENDENT_EVALUATOR_CHECK.md`; `results/independent_evaluators/` |
| Data availability | required | inherited and adapted for the current datasets | compiled in `main.tex` |
| Code availability | required | inherited GitHub/Zenodo/MIT statement | compiled in `main.tex` |
| CRediT contributions | required | inherited from existing drafts | compiled in `main.tex` |
| Competing interests | required | inherited no-conflict statement | compiled in `main.tex` |
| Funding | required | inherited no-specific-grant statement | compiled in `main.tex` |
| Related/preprint disclosure | required | inherited originality/status statement | compiled cover letter; no separate related list found |
| Suggested reviewers | verify IPT requirement | candidate identities missing | do not invent |
| Graphical abstract | verify IPT requirement | none supplied | optional pending rule |

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

## Author checks

- Confirm Research Paper against the current IPT author guide.
- Confirm continued use of the extracted author order, affiliations, emails,
  ORCIDs, CRediT roles, no-specific-grant funding, and no-competing-interest
  statements for this new manuscript.
- Confirm that the cover-letter assertion “not previously published / not under
  consideration” is complete; the reviewed drafts contain no separate preprint
  identifier or related-manuscript list.
- Create a new Zenodo version for the CS-ECF submission. The existing DOI
  resolves to the preceding study.
- Confirm the expanded AI-assisted-work disclosure.
- Add reviewer suggestions only if the IPT portal requests them.

## Next actions

1. Verify the current IPT Guide for Authors, article type, abstract limit,
   Highlights character limit, figure limits, and declarations.
2. Expand the verified bibliography into a complete related-work synthesis.
3. Review the compiled Elsevier preprint and resolve remaining layout choices.
4. Expand the seed bibliography and incorporate verified dataset/method papers.
5. Confirm the inherited administrative declarations and create a new Zenodo
   version.
6. Re-run numerical consistency checks and compile both PDFs.
7. Only then generate the IPT submission zip.

