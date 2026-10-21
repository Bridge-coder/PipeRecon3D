#!/bin/bash
# docs/10 X4: code check of train_tex.py (a few hundred iterations, then scoring of a few frames is skipped by --iters)
export TORCH_HOME=/mnt/d/pipe3d_gs/torch_cache
PY=/opt/gsenv/bin/python
T=/mnt/c/Users/PC/Desktop/PipeRecon3D/pipesplat/train_tex.py
G=/mnt/d/pipe3d_gs/g0
$PY $T $G/concrete_2 /mnt/d/pipe3d_gs/review/smoke_tex_c2 --iters 400 --save 11 2>&1 | grep -v "^  held\|^  block\|Warn\|warn" | tail -30
