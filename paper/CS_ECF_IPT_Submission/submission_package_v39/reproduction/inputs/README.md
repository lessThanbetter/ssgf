# Figure-generation inputs

These aggregate JSON files support `../build_figures.py`. They are copied from
the retrospective and prospective analysis outputs on 2026-10-29 and are not
raw experimental observations.

`figure1_candidate_examples.npz` contains display-resolution LLVIP frame
070142 crops for the visible input, infrared input, CDF, PIAFusion, and
SPGFusion panels of Figure 1. The source frame and archived score arrays are
identified in `source_data/Figure1_example_images.csv` and
`source_data/Figure1_example_scores.csv`.

The LLVIP and DroneVehicle per-image score arrays are stored under
`../../evidence/prospective/`. Detector weights and source images are not
redistributed with the manuscript package. Set `SSGF_DATA_ROOT` to the public
dataset checkout when rerunning the underlying analyses.
