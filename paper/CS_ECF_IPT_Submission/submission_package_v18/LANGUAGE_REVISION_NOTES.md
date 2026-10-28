# Language Revision Notes

Date: 2026-10-28  
Skills applied: `jiefu-paper-advisor` and `nature-polishing`  
Scope: manuscript, figure captions, supplement, Highlights, and cover letter

## Editorial objective

Improve sentence flow and terminology consistency without changing experiments,
numbers, uncertainty definitions, or the scientific claim boundary.

## Canonical terminology

| Canonical term | Treatment |
|---|---|
| CS-ECF | Expand on first use, then use the acronym |
| CDF | Expand as consensus-detail fusion on first use |
| no-shield selector | Used for the top-five selector without the 1.5 margin gate |
| LLVIP-fine-tuned YOLOv9t | Used for the domain-fine-tuned evaluator; avoids the competing labels `LLVIP-trained` and `domain-trained` |
| training-run cluster | Used when same-run best/last checkpoints are treated as correlated |
| scene group / frame group | Retained as dataset-specific units rather than collapsed into a generic group |
| dataset-balanced | Hyphenated as a compound modifier and defined as equal dataset weighting after within-dataset averaging |
| AP50 / AP50--95 | Kept distinct; AP50--95 averages the ten IoU thresholds from 0.50 to 0.95 |

## Main prose changes

- Replaced dense multi-clause openings with direct statements in the abstract
  and Introduction.
- Split overloaded sentences in Methods, Results, and Discussion while retaining
  all quantitative values and confidence intervals.
- Changed the DroneVehicle discussion from "dataset-balanced average on
  DroneVehicle" to the correct quantity: the mean gain on DroneVehicle.
- Expressed worst-case change as a signed AP50 value when discussing the
  no-shield selector and as loss magnitude when comparing magnitudes.
- Clarified that the dataset-balanced estimand averages within each dataset
  before weighting datasets equally.
- Kept interpretation in the Discussion and reduced result paragraphs to the
  observations needed to support it.
- Standardized British spelling for manuscript prose (`behaviour`, `colour`,
  `favour`, `summarise`) while preserving the confirmed administrative wording.

## Verification

- Abstract: 250 words.
- Cover-letter body: 490 words.
- Main manuscript: 25 A4 pages.
- Supplementary material: 6 A4 pages.
- Cover letter: 1 A4 page.
- Figures: 11 figures and 11 figure-level source-data files.
- Bibliography: 46 entries.
- No unresolved citations, cross-references, placeholders, overfull boxes, or
  missing figure captions.

No scientific or administrative conclusion was changed by this language pass.
