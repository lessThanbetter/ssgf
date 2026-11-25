# Revision notes for v32

Date: 2026-10-29

## Figure 5c scientific elements

The deployment decision panel now includes compact visual encodings for each
validation gate:

- a train/validation/test partition glyph for the held-out evaluator;
- two matched switch-pattern rows for the switch-budget comparison;
- a zero-referenced interval glyph for AP50 versus AP50-95 relevance;
- a stopwatch glyph for the four-pass latency budget.

The four gates are labeled g1 through g4 and linked to the formal deployment
condition:

`D = 1 ⇔ g1 ∧ g2 ∧ g3 ∧ g4`.

The panel geometry, YES/NO decision paths, and data content remain unchanged.

## Verification

- Figure-source preflight passes with no blocking findings.
- PDF text audit passes the 5 pt floor.
- Panels a and b remain unchanged.
