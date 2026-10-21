#!/bin/bash
# docs/10 section 5: the best Gaussian models as standard 3DGS .ply
PY=/opt/gsenv/bin/python
E=/mnt/c/Users/PC/Desktop/PipeRecon3D/pipesplat/export_ply.py
G=/mnt/d/pipe3d_gs/g0
for r in concrete_2/run_g0a concrete_lab/run_g0a inpipe_vision/run_g0ala_s20; do $PY $E $G/$r 2>&1 | grep -v Warn; done
