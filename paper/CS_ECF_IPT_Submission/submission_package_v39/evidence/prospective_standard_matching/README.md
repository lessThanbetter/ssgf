# Standard-matching LLVIP sensitivity

This directory is the main source for the held-out LLVIP AP50 results in v24.

- `protocol.json` records detector paths, SHA-256 values, NMS, confidence
  floor, and the confidence-ordered matching rule.
- `llvip{050,060,070}/scores/` contains four candidate arrays for each of six
  final detector checkpoints.
- `analysis.json` contains the equal-sequence AP50 estimates,
  conditional intervals, per-checkpoint values, and AP50--95 sensitivity.
- `matched_switch_target_control.json` contains the 1,000-draw switch-budget
  and candidate-mix matched random control.

The original all-at-once maximum-cardinality score arrays remain under
`../prospective/` for sensitivity comparison.
