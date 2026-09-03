#!/usr/bin/env bash
# Download and prepare the public datasets used by the paper.
# Usage: bash setup_datasets.sh [data_dir]
#   data_dir defaults to ./datasets (adjust paths in README / reproduce.sh).
set -e
D="${1:-datasets}"
mkdir -p "$D"

echo "== Public dataset sources =="
echo "  TNO       https://doi.org/10.6084/m9.figshare.1008029"
echo "  MSRS      https://github.com/Linfeng-Tang/Image-Fusion"
echo "  RoadScene https://github.com/hanna-xu/RoadScene"
echo "  LLVIP     https://github.com/bupt-ai-cz/LLVIP"
echo
echo "These benchmarks are publicly licensed; download the archives yourself and put them under:"
echo "  $D/tno.zip  $D/MSRS-main.zip  $D/RoadScene-master.zip  $D/LLVIP-main.zip"
echo "Then run the prep steps below (they convert raw folders into vis/ir pairs)."

# --- optional attempt: fetch the small, directly-hosted archives ---
# TNO (figshare direct download is usually click-to-accept; uncomment if you accept the licence)
# curl -L -o "$D/tno.zip" "https://figshare.com/ndownloader/files/1008029" || true
# RoadScene (github source zip)
if [ ! -d "$D/RoadScene-master" ]; then
  curl -L -o "$D/RoadScene.zip" "https://github.com/hanna-xu/RoadScene/archive/refs/heads/master.zip" || true
  [ -f "$D/RoadScene.zip" ] && (cd "$D" && unzip -o -q RoadScene.zip) || true
fi

echo
echo "== Prepare paired VIS/IR folders =="
# RoadScene: <root>/crop_HR_visible  +  <root>/infrared
python prepare_data.py --vis_root "$D/RoadScene-master/crop_HR_visible" \
    --ir_root "$D/RoadScene-master/infrared" --dest "$D/roadscene_pairs" --max_side 512 2>/dev/null || true
# TNO: <sequences>/visual + /thermal
if [ -d "$D/TNO_Image_Fusion_Dataset" ]; then
  python prepare_tno.py --tno "$D/TNO_Image_Fusion_Dataset" --dest "$D/tno_trial" || true
fi
# MSRS / LLVIP: raw vis+ir folders -> vis/ir pairs
#   prepare_data.py --vis_root <MSRS>/test/vi --ir_root <MSRS>/test/ir --dest "$D/msrs_pairs"

echo
echo "Done. Now point the dataset paths in README / reproduce.sh to the prepared folders."
