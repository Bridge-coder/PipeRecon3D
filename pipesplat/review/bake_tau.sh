#!/bin/bash
# docs/10 X8: bake variants with a different number of near views averaged (--near-tau), each baked and scored once
# (baked texture x lamp map, exact triangles), one after another.
# usage: bake_tau.sh NAME:TAU [NAME:TAU ...]   e.g. tau01:0.1 tau1:1.0 tau3:3 tau10:10  (-> run_bake_NAME)
# (run 1 first re-scored run_bake_near, whose first scoring ran out of memory while a diagnosis ran at the same time)
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"   # REPO, P (pipesplat), PY, GS, TORCH_HOME from scripts/paths.py
G=$GS/g0/concrete_2
R=$GS/review
for v in "$@"; do
  n=${v%%:*}; t=${v##*:}
  $PY $P/bake_gs_tex.py $G $G/run_g0a $G/run_bake_$n --near-tau $t > $R/bake_run_bake_$n.err 2>&1
  $PY $P/train_tex.py $G $G/run_bake_$n --eval-only --no-lamp --lamp-map --tex-pt $G/run_bake_$n/tex.pt \
      --surface $G/run_bake_$n/surface_bake.npz --mesh-raster $G/run_bake_$n/mesh_bake.npz > $R/score_run_bake_$n.err 2>&1
  echo "== run_bake_$n"; grep -A1 "^in-seq\|^block fr" $G/run_bake_$n/train.log | grep TX; tail -2 $R/score_run_bake_$n.err
done
