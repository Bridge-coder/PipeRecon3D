#!/bin/bash
# docs/10 X5: code check of train_gs.py --pose-opt (300 iterations, no scoring)
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"   # REPO, P (pipesplat), PY, GS, TORCH_HOME from scripts/paths.py
T=$P/train_gs.py
G=$GS/g0
$PY $T $G/inpipe_172011 $GS/review/smoke_pose_inpipe --anchor --lamp --lamp-ambient --pose-opt --iters 5000 --smoke 2>&1 | grep -v "Warn\|warn" | grep -E "iter|lamp:" | tail -12
