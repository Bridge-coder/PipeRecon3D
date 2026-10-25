#!/bin/bash
# docs/11 Y2/Y3: short GPU jobs run while the training queue is paused: Pipe1 bakes (colours from the model / from the
# video, 1 mm texels: 11 m of 442 mm pipe), and outside views of this round's Gaussian models (cameras from outside_views.py)
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"   # REPO, P (pipesplat), PY, GS, TORCH_HOME from scripts/paths.py
G=$GS/g0
V=$GS/review/outside
for name in run_bake_vhf_model run_bake_vhf; do
  rm -rf $G/sewer_pipe1/$name
  src=video; [ $name = run_bake_vhf_model ] && src=model
  $PY $P/bake_vhf.py $G/sewer_pipe1 $G/sewer_pipe1/run_vhf $G/sewer_pipe1/$name --texel-mm 1.0 --source $src > /dev/null 2>&1
  echo "== Pipe1 $name (source $src)"
  grep -E "model surface|flat field|texture " $G/sewer_pipe1/$name/bake.log | cut -c1-170
  tail -13 $G/sewer_pipe1/$name/bake.log | grep -E "baked mesh|lampmap B" | cut -c1-170
done
$PY $P/review/outside_views_gs.py $V/sewer_pipe1 $G/sewer_pipe1/run_vhf 2>&1 | grep -v Warn | tail -2
mkdir -p $V/inpipe_y1 && cp $V/inpipe/cams.json $V/inpipe/old_*.png $V/inpipe_y1/
$PY $P/review/outside_views_gs.py $V/inpipe_y1 $G/inpipe_vision/run_vhf 2>&1 | grep -v Warn | tail -2
