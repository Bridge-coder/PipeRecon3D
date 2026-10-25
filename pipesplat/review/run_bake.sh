#!/bin/bash
# docs/10 X8: bake a Gaussian model into the mesh texture (pipesplat/bake_gs_tex.py).
# usage: run_bake.sh DATA MODEL_RUN OUT_RUN [bake options]   (stdout/stderr -> <GS_ROOT>/review/bake_OUT_RUN.err)
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"   # REPO, P (pipesplat), PY, GS, TORCH_HOME from scripts/paths.py
B=$P/bake_gs_tex.py
G=$GS/g0
$PY $B $G/$1 $G/$1/$2 $G/$1/$3 "${@:4}" > $GS/review/bake_$3.err 2>&1
tail -12 $G/$1/$3/bake.log
tail -5 $GS/review/bake_$3.err
