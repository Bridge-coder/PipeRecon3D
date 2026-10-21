#!/bin/bash
# docs/10 X5: code check of train_gs.py --pose-opt (300 iterations, no scoring)
export TORCH_HOME=/mnt/d/pipe3d_gs/torch_cache
PY=/opt/gsenv/bin/python
T=/mnt/c/Users/PC/Desktop/PipeRecon3D/pipesplat/train_gs.py
G=/mnt/d/pipe3d_gs/g0
$PY $T $G/inpipe_172011 /mnt/d/pipe3d_gs/review/smoke_pose_inpipe --anchor --lamp --lamp-ambient --pose-opt --iters 5000 --smoke 2>&1 | grep -v "Warn\|warn" | grep -E "iter|lamp:" | tail -12
