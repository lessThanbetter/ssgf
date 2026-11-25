# Prospective evidence archive

The protocol Markdown files and their original SHA-256 sidecars are stored in
`protocol/`. The protocols were recorded locally before confirmation scoring
and were not externally registered.

The manuscript's standard-matching aggregate is stored separately in
`../prospective_standard_matching/`. The legacy
`canonical_prospective_evidence.json` is retained as the all-at-once
maximum-cardinality sensitivity record. It contains:

- LLVIP six-final-checkpoint and archived-eight-checkpoint results;
- CS-ECF contrasts against visible, Always CDF, and no shield;
- the sequence-040 margin audit;
- the matched image-statistic audit outcome;
- Native DroneVehicle OBB results for the public best/last checkpoints and two
  new independent runs;
- explicit independence and claim-status fields.

`llvip_primary_analysis.json` and `llvip_original8_analysis.json` are the
complete maximum-cardinality sensitivity outputs. Per-sequence score
directories contain compressed
NumPy arrays for every checkpoint and candidate:

- `llvip{040,050,060,070}/scores/`: six final checkpoints, four candidates;
- `llvip{040,050,060,070}/scores_original8/`: eight archived checkpoints, four
  candidates;
- `dronevehicle/native_scores/`: two new runs, four candidates;
- `dronevehicle/native_scores_public/`: public best/last, four candidates.

Each score file contains per-detection true-positive status, confidence, class,
image index, and ground-truth counts. Source images, detector weights, and
fusion-model weights are not redistributed.

The analysis scripts are in `../../reproduction/`. `assemble_prospective_evidence.py`
rebuilds the canonical JSON and the copy used by the figure driver.
