# Figure and table layout audit

Audit date: 27 October 2026

- Manuscript: `PAPER_IPT_v3.pdf`
- Compiled length: 23 pages
- Figures: 11, numbered consecutively as 1-11
- Tables: 13, numbered consecutively as 1-13
- Figure captions: below the image, all on the same page
- Table captions: above the table body, all on the same page
- First textual reference: present for every figure and table

## Figure placement

| Figure | Caption page | First text citation |
|---|---:|---:|
| 1. Framework | 2 | 1 |
| 2. Clean-split evidence | 6 | 5 |
| 3. Regime sensitivity | 8 | 8 |
| 4. Seed/index robustness | 9 | 9 |
| 5. Fidelity transfer | 12 | 12 |
| 6. IoU criterion | 13 | 12 |
| 7. Recall/precision | 14 | 13 |
| 8. Temperature sweep | 14 | 13 |
| 9. Backbone/resolution | 15 | 15 |
| 10. Degraded inputs | 18 | 17 |
| 11. Conditional ranking | 19 | 18 |

## Table placement

| Table | Caption and body page | First text citation |
|---|---:|---:|
| 1. VIF comparison | 4 | 4 |
| 2. Full-benchmark fusion metrics | 5 | 4 |
| 3. Inference cost | 5 | 5 |
| 4. Primary independent test | 6 | 2 |
| 5. Pooled sensitivity audit | 6 | 5 |
| 6. Dual source conventions | 7 | 6 |
| 7. FLIR replication | 10 | 9 |
| 8. Cross-benchmark contrasts | 10 | 9 |
| 9. Architecture versus seed | 11 | 11 |
| 10. Fidelity and downstream contrast | 12 | 11 |
| 11. Criterion sweep | 13 | 12 |
| 12. Scale experiments | 15 | 15 |
| 13. Degraded inputs | 17 | 17 |

## Corrections made during the audit

- Table 2 is a non-splitting `table` float, with caption, header and all rows together
  on page 5.
- Table 5 is a non-splitting `table` float, with caption, header and all rows together
  on page 6.
- Figure 1 now has a textual citation in the Introduction before the float.
- Figure 2 now has a textual citation immediately before the float.

## Automated checks

- All 24 figure/table labels resolve.
- Figure numbers are consecutive 1-11.
- Table numbers are consecutive 1-13.
- All 11 figure captions follow the graphics.
- All 14 table captions precede the tabular body.
- No undefined references, undefined citations or overfull boxes.
- No caption is stranded on a page without its figure or table.
