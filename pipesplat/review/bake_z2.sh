#!/bin/bash
# docs/13 Z2: InPipe Y1 baked with the two light sources split (bake_vhf.py --two-light); prints the scores
# usage: bake_z2.sh [OUT_RUN=run_bake_vhf_2l] [bake_vhf.py options]
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"   # REPO, P (pipesplat), PY, GS, TORCH_HOME from scripts/paths.py
G=$GS/g0/inpipe_vision
OUT=${1:-run_bake_vhf_2l}
rm -rf $G/$OUT
$PY $P/bake_vhf.py $G $G/run_vhf $G/$OUT --two-light "${@:2}" > $GS/review/bake_$OUT.err 2>&1
echo "== $OUT ${*:2} (exit $?)"
grep -a -E "two-light bake|model surface|^texture " $G/$OUT/bake.log | cut -c1-220
grep -a -A3 "median, ruler v2" $G/$OUT/bake.log | grep -a -E "ruler v2|baked mesh|lampmap B" | cut -c1-200; grep -a "camera-fixed map on top" $G/$OUT/bake.log | cut -c1-160
tail -3 $GS/review/bake_$OUT.err | grep -a -i -E "error|Traceback" || true
