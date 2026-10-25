#!/bin/bash
# docs/10 X4: code check of train_tex.py (a few hundred iterations, then scoring of a few frames is skipped by --iters)
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"   # REPO, P (pipesplat), PY, GS, TORCH_HOME from scripts/paths.py
T=$P/train_tex.py
G=$GS/g0
$PY $T $G/concrete_2 $GS/review/smoke_tex_c2 --iters 400 --save 11 2>&1 | grep -v "^  held\|^  block\|Warn\|warn" | tail -30
