# Resource amendment before scoring

Recorded: 2026-11-26T00:55:16+08:00

The public LLVIP archive is a 4.0 GB ZIP. Remote byte-range access for the
three new sequences was limited to approximately 50--70 KB/s, making the full
2,981-frame confirmation impractical within the revision run.

No PIAFusion candidate, detector prediction, match result, or AP outcome had
been generated when this amendment was recorded. Some source JPEGs had started
downloading, but no candidate or evaluator outcome had been inspected.

The confirmation sample is therefore fixed, before scoring, as the first 300
lexicographically ordered, non-empty labelled stems in each sequence:

- 080: `080001` through `080300` where labels exist;
- 090: the first 300 available non-empty labelled stems;
- 100: the first 300 available non-empty labelled stems.

The selection rule depends only on stem identifiers and label-file
non-emptiness. It does not use image pixels, detector scores, PIAFusion scores,
or AP outcomes.

All other elements of `PROTOCOL.md` remain unchanged, including the policy,
threshold, detector checkpoints, matching rule, endpoints, and resampling
procedure. The resulting 900-image analysis is a bounded external validation,
not a full-sequence estimate.
