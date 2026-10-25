#!/bin/bash
# docs/10: texture model TX (pipesplat/train_tex.py). usage: run_tex.sh DATA RUN [options]
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"   # REPO, P (pipesplat), PY, GS, TORCH_HOME from scripts/paths.py
T=$P/train_tex.py
G=$GS/g0
$PY $T $G/$1 $G/$1/$2 "${@:3}" > /dev/null 2>&1
tail -16 $G/$1/$2/train.log
