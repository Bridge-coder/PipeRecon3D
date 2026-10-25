#!/bin/bash
# docs/11: code check of train_gs.py --baseline 3dgs|mcmc|2dgs (300 iterations, no scoring) on concrete_2
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"   # REPO, P (pipesplat), PY, GS, TORCH_HOME from scripts/paths.py
T=$P/train_gs.py
G=$GS/g0
for m in 3dgs mcmc 2dgs; do
  $PY $T $G/concrete_2 $G/concrete_2/_smoke_$m --baseline $m --iters 300 --smoke > /dev/null 2>&1
  echo "== $m exit $?"
  grep -E "iter|Gaussians|baseline|Error|error" $G/concrete_2/_smoke_$m/train.log | tail -4
done
