#!/bin/bash
# docs/10 X2/X3: G0-A with the scale cap (one change), concrete_2 then InPipe
export TORCH_HOME=/mnt/d/pipe3d_gs/torch_cache
PY=/opt/gsenv/bin/python
T=/mnt/c/Users/PC/Desktop/PipeRecon3D/pipesplat/train_gs.py
G=/mnt/d/pipe3d_gs/g0
$PY $T $G/$1 $G/$1/$2 "${@:3}" > /dev/null 2>&1
tail -25 $G/$1/$2/train.log
