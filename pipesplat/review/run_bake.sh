#!/bin/bash
# docs/10 X8: bake a Gaussian model into the mesh texture (pipesplat/bake_gs_tex.py).
# usage: run_bake.sh DATA MODEL_RUN OUT_RUN [bake options]   (stdout/stderr -> /mnt/d/pipe3d_gs/review/bake_OUT_RUN.err)
export TORCH_HOME=/mnt/d/pipe3d_gs/torch_cache
PY=/opt/gsenv/bin/python
B=/mnt/c/Users/PC/Desktop/PipeRecon3D/pipesplat/bake_gs_tex.py
G=/mnt/d/pipe3d_gs/g0
$PY $B $G/$1 $G/$1/$2 $G/$1/$3 "${@:4}" > /mnt/d/pipe3d_gs/review/bake_$3.err 2>&1
tail -12 $G/$1/$3/bake.log
tail -5 /mnt/d/pipe3d_gs/review/bake_$3.err
