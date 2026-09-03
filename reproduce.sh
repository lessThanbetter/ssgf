#!/usr/bin/env bash
# One-shot reproduction of the main fusion table (VIF) on TNO / MSRS / RoadScene.
set -e
MSRS="${MSRS:?set MSRS=/path/to/MSRS-main}"
TNO="${TNO:?set TNO=/path/to/tno_trial}"
ROAD="${ROAD:?set ROAD=/path/to/RoadScene-master}"
python evaluate.py --ae data/msrs_ae.pt --probe data/msrs_probe.pt --prior tophat \
    --max_side 512 \
    --datasets "TNO::${TNO}::vis::ir" \
    --datasets "MSRS::${MSRS}/test::vi::ir" \
    --datasets "RoadScene::${ROAD}::crop_HR_visible::infrared" \
    --out data/cross_dataset.csv
python sota_runners/multiseed_vif.py
