# Infrared Physics & Technology Submission Checklist

Date: 2026-10-28  
Target: Infrared Physics & Technology (Elsevier)

## Submission readiness

**Ready with author checks for the identified manuscript; journal submission is
still blocked by the incomplete domain bibliography, missing supplementary file,
and unverified current IPT instructions.**

Reasons:

- the bibliography is a verified 11-entry seed set rather than a complete domain
  review;
- exact current IPT author-instruction limits and declarations still need
  verification;
- the administrative declarations were inherited from the existing manuscript
  family and require final author confirmation;
- a supplementary file has not yet been assembled.

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
| Supplementary information | required if referenced | appendix plan outstanding | not started |
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
- detector-by-group hierarchical CI `[+0.0028,+0.0213]`;
- no-shield worst loss `-0.0584`, shield worst loss `-0.0040`;
- class-aware M3FD analysis showing a people-specific effect;
- thermal-top-hat stratification and thermal-mask/global ablation;
- standardized serial latency estimate `706.3 ms`;
- verified 2024–2026 IPT fusion/detection exemplars in `IPT_TARGETING.md`.

## Cross-file consistency checks

The following canonical values must remain identical everywhere:

| Metric | Canonical value |
|---|---:|
| Confirmatory checkpoints | 12 |
| Positive checkpoints | 9 |
| Dataset-balanced AP50 gain | +0.0133 |
| Model-by-group hierarchical CI | [+0.0028,+0.0213] |
| No-shield worst AP50 loss | -0.0584 |
| CS-ECF worst AP50 loss | -0.0040 |
| DroneVehicle best gain | +0.0251 |
| DroneVehicle last gain | +0.0293 |
| DroneVehicle group CI | [+0.0002,+0.0406] |
| Thermal-mask advantage over global | +0.0007 |
| Serial latency estimate | 706.3 ms |

## Author checks

- Confirm Research Paper against the current IPT author guide.
- Confirm continued use of the extracted author order, affiliations, emails,
  ORCIDs, CRediT roles, no-specific-grant funding, and no-competing-interest
  statements for this new manuscript.
- Confirm that the cover-letter assertion “not previously published / not under
  consideration” is complete; the reviewed drafts contain no separate preprint
  identifier or related-manuscript list.
- Update the Zenodo record from the IPT submission tag before submission.
- Confirm the expanded AI-assisted-work disclosure.
- Add reviewer suggestions only if the IPT portal requests them.

## Next actions

1. Verify the current IPT Guide for Authors, article type, abstract limit,
   Highlights character limit, figure limits, and declarations.
2. Expand the verified bibliography into a complete related-work synthesis.
3. Review the compiled Elsevier preprint and resolve remaining layout choices.
4. Expand the seed bibliography and incorporate verified dataset/method papers.
5. Confirm the inherited administrative declarations and update Zenodo.
6. Re-run numerical consistency checks and compile the submission PDF.
7. Only then generate the IPT submission zip.

