#!/bin/bash
# docs/10: code check of the new options (300 iterations, no scoring)
export TORCH_HOME=/mnt/d/pipe3d_gs/torch_cache
PY=/opt/gsenv/bin/python
T=/mnt/c/Users/PC/Desktop/PipeRecon3D/pipesplat/train_gs.py
G=/mnt/d/pipe3d_gs/g0
$PY $T $G/concrete_2 /mnt/d/pipe3d_gs/review/smoke_c2_raw_cap --anchor --raw --scale-cap 3 --iters 300 --smoke 2>&1 | grep -v "^  \|Warn\|warn"
$PY $T $G/concrete_2 /mnt/d/pipe3d_gs/review/smoke_c2_raw_lamp --anchor --raw --lamp --iters 300 --smoke 2>&1 | grep -v "^  \|Warn\|warn" | tail -6
