#!/bin/bash
# docs/11: code check of train_gs.py --baseline 3dgs|mcmc|2dgs (300 iterations, no scoring) on concrete_2
export TORCH_HOME=/mnt/d/pipe3d_gs/torch_cache
PY=/opt/gsenv/bin/python
T=/mnt/c/Users/PC/Desktop/PipeRecon3D/pipesplat/train_gs.py
G=/mnt/d/pipe3d_gs/g0
for m in 3dgs mcmc 2dgs; do
  $PY $T $G/concrete_2 $G/concrete_2/_smoke_$m --baseline $m --iters 300 --smoke > /dev/null 2>&1
  echo "== $m exit $?"
  grep -E "iter|Gaussians|baseline|Error|error" $G/concrete_2/_smoke_$m/train.log | tail -4
done
