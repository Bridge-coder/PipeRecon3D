#!/bin/bash
# docs/10: texture model TX (pipesplat/train_tex.py). usage: run_tex.sh DATA RUN [options]
export TORCH_HOME=/mnt/d/pipe3d_gs/torch_cache
PY=/opt/gsenv/bin/python
T=/mnt/c/Users/PC/Desktop/PipeRecon3D/pipesplat/train_tex.py
G=/mnt/d/pipe3d_gs/g0
$PY $T $G/$1 $G/$1/$2 "${@:3}" > /dev/null 2>&1
tail -16 $G/$1/$2/train.log
