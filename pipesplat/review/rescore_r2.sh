#!/bin/bash
# docs/10 section 2: re-score the trained models of docs/08-09 with ruler v2 (eval only, nothing retrained).
# run: wsl -d pipe3d -u root -- bash <repo>/pipesplat/review/rescore_r2.sh
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"   # REPO, P (pipesplat), PY, GS, TORCH_HOME from scripts/paths.py
T=$P/train_gs.py
G=$GS/g0
E="--eval-only --ruler2 --alpha-min 0.5 --suffix _r2"
run() { echo "=== $1 $2"; $PY $T $G/$1 $G/$1/$2 $E "${@:3}" 2>&1 | grep -v "^  held\|^  block" | tail -14; }
run concrete_2 run_g0
run concrete_2 run_g0a --anchor
run concrete_2 run_g0al_s20 --anchor --lamp --lamp-smooth-mm 20
run inpipe_172011 run_g0
run inpipe_172011 run_g0a --anchor
run inpipe_172011 run_g0ala_s20 --anchor --lamp --lamp-ambient --lamp-smooth-mm 20
