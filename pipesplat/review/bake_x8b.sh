#!/bin/bash
# docs/10 X8b: variants of the G0-A bake (pipesplat/bake_gs_tex.py), each baked and scored once (baked texture x lamp
# map, exact triangles of its own mesh_bake.npz), one after another:
#   rep     the X8 bake again (same options as run_bake: run-to-run noise of bake + scoring)
#   endref  end region (cap + wall beyond u 4.26 m) from the ref views only
#   endtau  end region from many views (near-tau 10)
#   endrob  end region from many views with lamp highlights down-weighted (robust)
#   all     the X8 bake on the mesh with the faces OpenMVS left without an image (mesh_all.npz: no holes)
#   allwall those faces added only where their centre lies at x < 4.26 m, not in the end cap / socket mesh
#           (mesh_all_wall.npz; adopted in docs/10 X8b)
# meshes (Windows): python -I pipesplat/review/dump_mesh.py D:/pipe3d/concrete_2 D:/pipe3d_gs/g0/concrete_2/mesh_all.npz
#           --with-blank   (and mesh_all_wall.npz with --with-blank --blank-max-x 4.26)
# usage: bake_x8b.sh NAME [NAME ...]   (-> run_bake_NAME; logs /mnt/d/pipe3d_gs/review/{bake,score}_run_bake_NAME.err)
export TORCH_HOME=/mnt/d/pipe3d_gs/torch_cache
PY=/opt/gsenv/bin/python
P=/mnt/c/Users/PC/Desktop/PipeRecon3D/pipesplat
G=/mnt/d/pipe3d_gs/g0/concrete_2
R=/mnt/d/pipe3d_gs/review
for n in "$@"; do
  case $n in
    rep)       o="" ;;
    endref)    o="--end-x 4.26 --end-mode ref" ;;
    endtau)    o="--end-x 4.26 --end-mode tau --end-tau 10" ;;
    endrob)    o="--end-x 4.26 --end-mode robust --end-tau 10 --end-hl 0.15" ;;
    all)       o="--mesh mesh_all.npz" ;;
    allwall)   o="--mesh mesh_all_wall.npz" ;;   # control: those faces only on the wall (x < 4.26 m), not at the end
    allendrob) o="--mesh mesh_all.npz --end-x 4.26 --end-mode robust --end-tau 10 --end-hl 0.15" ;;
    allendref) o="--mesh mesh_all.npz --end-x 4.26 --end-mode ref" ;;
    allendtau) o="--mesh mesh_all.npz --end-x 4.26 --end-mode tau --end-tau 10" ;;
    *) echo "unknown variant $n"; continue ;;
  esac
  $PY $P/bake_gs_tex.py $G $G/run_g0a $G/run_bake_$n $o > $R/bake_run_bake_$n.err 2>&1
  $PY $P/train_tex.py $G $G/run_bake_$n --eval-only --no-lamp --lamp-map --tex-pt $G/run_bake_$n/tex.pt \
      --surface $G/run_bake_$n/surface_bake.npz --mesh-raster $G/run_bake_$n/mesh_bake.npz > $R/score_run_bake_$n.err 2>&1
  echo "== run_bake_$n ($o)"; grep -A1 "^in-seq\|^block fr" $G/run_bake_$n/train.log | grep TX; tail -2 $R/score_run_bake_$n.err
done
