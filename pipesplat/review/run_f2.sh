#!/bin/bash
# docs/10 X2/X3: G0-A with the scale cap (one change), concrete_2 then InPipe
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"   # REPO, P (pipesplat), PY, GS, TORCH_HOME from scripts/paths.py
T=$P/train_gs.py
G=$GS/g0
$PY $T $G/$1 $G/$1/$2 "${@:3}" > /dev/null 2>&1
tail -25 $G/$1/$2/train.log
