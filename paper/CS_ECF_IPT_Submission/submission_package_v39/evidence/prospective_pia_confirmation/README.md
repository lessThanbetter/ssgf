# Prospective visible-plus-PIAFusion confirmation

The policy was frozen before this holdout was scored. A resource amendment,
recorded before candidate or detector scoring, fixed 300 lexicographically
selected non-empty frames in each of LLVIP sequences 080, 090, and 100.

## Detection evidence

Under confidence-ordered one-to-one matching, the frozen visible-plus-PIA
policy gained `+0.01560` AP50 over visible input. The conditional bootstrap
interval was `[+0.00249,+0.02874]`. The gain was positive in all three
sequences and all six detector checkpoints.

| Scope | AP50 gain |
|---|---:|
| sequence 080 | `+0.02543` |
| sequence 090 | `+0.01924` |
| sequence 100 | `+0.00213` |
| worst checkpoint | `+0.00784` |

The AP50--95 gain was `+0.00946` `[+0.00113,+0.01769]`.

Only `5.22%` of image-checkpoint cases switched from visible to PIAFusion.
The selected output won on `4.22%` of cases, tied on `95.20%`, and lost on
`0.57%`.

## Selection boundary

Always using PIAFusion did not transfer across these sequences. The raw
PIAFusion-minus-visible AP50 change had an equal-sequence point estimate of
`-0.01237` with a wide conditional interval
`[-0.15111,+0.07787]`. Its per-sequence values were:

- 080: `+0.05415`;
- 090: `+0.06766`;
- 100: `-0.15891`.

The confidence gate retained visible input on most sequence-100 frames and
converted that large regression into `+0.00213`. The result therefore
supports the gate as a selective policy, not PIAFusion as a universally
superior fused representation.

## Scope

This is a post-review temporal external validation, not an externally
registered trial. The sample contains 900 images from three fixed sequences
and six fixed detector runs. The interval resamples within this observed set.
It does not remove the negative VEDAI and DroneVehicle transfer results.

## Files

- `PROTOCOL.md` and `PROTOCOL_AMENDMENT_20261126.md`: frozen design and sample
  amendment;
- `source_manifest.json`: 900 source pairs and 1,800 source-image hashes;
- `candidate_manifest.json`: PIAFusion candidate hashes. Candidate images are
  not redistributed;
- `scores/`: confidence-ordered per-image AP arrays;
- `analysis.json`: endpoints, intervals, per-sequence values, and run-level
  results;
- `end_to_end_latency.json` and `end_to_end_latency_summary.csv`: measured
  single-GPU latency.

Reproduction entry points are in `../../reproduction/`:

- `fetch_pia_holdout.py`;
- `prepare_pia_sources.py`;
- `generate_pia_candidates.py`;
- `score_pia_confirmation.py`;
- `analyze_pia_confirmation.py`;
- `assemble_pia_confirmation.py`;
- `benchmark_end_to_end.py` and `summarize_end_to_end.py`.
