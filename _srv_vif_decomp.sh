#!/bin/bash
cd /root/autodl-tmp/ssgf
EF=/root/autodl-tmp/datasets/det/eval_fused
PY=/root/miniconda3/bin/python
nohup $PY -u /root/autodl-tmp/ssgf/_audit/vif_decomp2.py \
  --vis /root/autodl-tmp/datasets/det/test/visible \
  --ir /root/autodl-tmp/datasets/det/test/infrared \
  --fused mean=$EF/mean/images \
  --fused tophat=$EF/tophat/images \
  --fused probe=$EF/probe/images \
  --fused ensemble=$EF/ensemble/images \
  --fused ifcnn=$EF/ifcnn/images \
  --fused densefuse=$EF/densefuse/images \
  --fused piafusion=/root/autodl-tmp/ssgf/_audit/pia/images \
  --fused spgfusion=/root/autodl-tmp/ssgf/_audit/spg/images \
  --limit 30 \
  --target_hw 400x512 \
  --out /root/autodl-tmp/ssgf/_audit/vif_decomp3 \
  > /root/autodl-tmp/ssgf/_audit/vif_decomp3.log 2>&1 &
echo "launched"
sleep 3
tail -3 /root/autodl-tmp/ssgf/_audit/vif_decomp.log
