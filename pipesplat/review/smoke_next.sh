#!/bin/bash
# docs/10: code check of the new options (300 iterations, no scoring)
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"   # REPO, P (pipesplat), PY, GS, TORCH_HOME from scripts/paths.py
T=$P/train_gs.py
G=$GS/g0
$PY $T $G/concrete_2 $GS/review/smoke_c2_raw_cap --anchor --raw --scale-cap 3 --iters 300 --smoke 2>&1 | grep -v "^  \|Warn\|warn"
$PY $T $G/concrete_2 $GS/review/smoke_c2_raw_lamp --anchor --raw --lamp --iters 300 --smoke 2>&1 | grep -v "^  \|Warn\|warn" | tail -6
