#!/bin/bash
# docs/11 Y2: bake_vhf.py variants on InPipe Y1 (run_vhf), one after another; prints the scores
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"   # REPO, P (pipesplat), PY, GS, TORCH_HOME from scripts/paths.py
S=$P/bake_vhf.py
G=$GS/g0/inpipe_vision
run() {
  name=$1; shift
  rm -rf $G/$name
  $PY $S $G $G/run_vhf $G/$name "$@" > /dev/null 2>&1
  echo "== $name $*"
  tail -13 $G/$name/bake.log | grep -E "baked mesh" | cut -c1-170
}
for v in "$@"; do
  case $v in
    flat2) run run_bake_vhf_flat2 --flat-iters 2 ;;
    noref) run run_bake_vhf_noref --lp-mm 0 ;;
    noref2) run run_bake_vhf_noref2 --lp-mm 0 --flat-iters 2 ;;
    tau3) run run_bake_vhf_tau3 --near-tau 3 ;;
  esac
done
