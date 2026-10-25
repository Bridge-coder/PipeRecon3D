#!/bin/bash
# docs/10 X8: ruler-v2 scores of textures baked from G0-A (train_tex.py --eval-only --no-lamp --tex-pt), one after another:
#   BAKE_RUN       baked texture x lamp map (lampmap's recipe, as B)
#   BAKE_RUN_flat  baked texture alone (what a viewer shows without lighting), per-frame gain/offset only
#   BAKE_RUN_near  the near-view mean only (no brightness normalisation) x lamp map (ablation)
# the mesh is drawn as exact triangles (--mesh-raster, as a viewer draws the OBJ): BAKE_RUN/mesh_bake.npz + its
# surface_bake.npz (texture layout) when they exist (the mesh moved onto the model's surface), else the scene's mesh.npz
# (the first scoring drew one disc per triangle and is kept as *_disc: wrong for a texture on the triangles)
# usage: score_bake.sh [BAKE_RUN=run_bake]
export TORCH_HOME=/mnt/d/pipe3d_gs/torch_cache
PY=/opt/gsenv/bin/python
T=/mnt/c/Users/PC/Desktop/PipeRecon3D/pipesplat/train_tex.py
G=/mnt/d/pipe3d_gs/g0/concrete_2
R=/mnt/d/pipe3d_gs/review
B=${1:-run_bake}
S="--mesh-raster $G/mesh.npz"
[ -f $G/$B/surface_bake.npz ] && S="--surface $G/$B/surface_bake.npz --mesh-raster $G/$B/mesh_bake.npz"
$PY $T $G $G/$B --eval-only --no-lamp --lamp-map --tex-pt $G/$B/tex.pt $S > $R/score_$B.err 2>&1
$PY $T $G $G/${B}_flat --eval-only --no-lamp --tex-pt $G/$B/tex.pt $S > $R/score_${B}_flat.err 2>&1
$PY $T $G $G/${B}_near --eval-only --no-lamp --lamp-map --tex-pt $G/$B/tex_near.pt $S > $R/score_${B}_near.err 2>&1
for r in $B ${B}_flat ${B}_near; do echo "== $r"; grep -A1 "^in-seq\|^block fr" $G/$r/train.log | grep TX | tail -2; tail -2 $R/score_$r.err; done
