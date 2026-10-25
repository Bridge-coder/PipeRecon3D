#!/bin/bash
# docs/10 X8: the final bake (defaults: model surface, near-tau 1) and its scores, then TX's own OBJ drawn as exact
# triangles with its own lamp model (how its texture looks on the mesh in a viewer), one after another
export TORCH_HOME=/mnt/d/pipe3d_gs/torch_cache
PY=/opt/gsenv/bin/python
P=/mnt/c/Users/PC/Desktop/PipeRecon3D/pipesplat
G=/mnt/d/pipe3d_gs/g0/concrete_2
R=/mnt/d/pipe3d_gs/review
bash $P/review/run_bake.sh concrete_2 run_g0a run_bake
bash $P/review/score_bake.sh run_bake
$PY $P/train_tex.py $G $G/run_tx_log_tri --eval-only --tex-pt $G/run_tx_log/tex.pt --mesh-raster $G/mesh.npz > $R/score_run_tx_log_tri.err 2>&1
echo "== run_tx_log_tri"; grep -A1 "^in-seq\|^block fr" $G/run_tx_log_tri/train.log | grep TX; tail -2 $R/score_run_tx_log_tri.err
