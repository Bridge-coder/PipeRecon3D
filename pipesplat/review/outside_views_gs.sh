#!/bin/bash
# docs/10: outside / cut-away / side views of the best Gaussian models (cameras from outside_views.py)
PY=/opt/gsenv/bin/python
S=/mnt/c/Users/PC/Desktop/PipeRecon3D/pipesplat/review/outside_views_gs.py
V=/mnt/d/pipe3d_gs/review/outside
G=/mnt/d/pipe3d_gs/g0
$PY $S $V/concrete_2 $G/concrete_2/run_g0a "$@" 2>&1 | grep -v Warn
$PY $S $V/concrete_lab $G/concrete_lab/run_g0a "$@" 2>&1 | grep -v Warn
$PY $S $V/inpipe $G/inpipe_vision/run_g0ala_s20 "$@" 2>&1 | grep -v Warn
