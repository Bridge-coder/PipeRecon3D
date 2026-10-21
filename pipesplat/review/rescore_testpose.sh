#!/bin/bash
# docs/10 X6c: score with test-time pose refinement of each scored frame (model frozen), files *_r2tp
export TORCH_HOME=/mnt/d/pipe3d_gs/torch_cache
PY=/opt/gsenv/bin/python
T=/mnt/c/Users/PC/Desktop/PipeRecon3D/pipesplat/train_gs.py
G=/mnt/d/pipe3d_gs/g0
E="--eval-only --ruler2 --alpha-min 0.5 --suffix _r2tp --test-pose-iters 3"
run() { echo "=== $1 $2"; $PY $T $G/$1 $G/$1/$2 $E "${@:3}" > /dev/null 2>&1; grep -a -A3 "in-sequence\|block frames" $G/$1/$2/train.log | tail -10; }
run inpipe_172011 run_g0ala_s20_rc --anchor --lamp --lamp-ambient --lamp-smooth-mm 20 --surface surface_rc.npz --anchor-snap-mm 40
run inpipe_172011 run_g0ala_s20 --anchor --lamp --lamp-ambient --lamp-smooth-mm 20
run concrete_2 run_g0a --anchor
