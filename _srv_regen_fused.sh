#!/bin/bash
set -e
cd /root/autodl-tmp/ssgf
PY=/root/miniconda3/bin/python
mkdir -p /root/autodl-tmp/ssgf/_restore
echo "=== checkpoints ==="
ls -la /root/autodl-tmp/ssgf/data/msrs_ae.pt /root/autodl-tmp/ssgf/data/msrs_probe.pt \
      /root/autodl-tmp/ssgf/data/msrs_csf.pt
echo "=== launching regeneration ==="
nohup $PY -u /root/autodl-tmp/ssgf/_restore_regen.py \
  > /root/autodl-tmp/ssgf/_restore/regen.log 2>&1 &
echo launched
