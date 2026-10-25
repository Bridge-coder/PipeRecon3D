#!/bin/bash
# docs/10: outside / cut-away / side views of the best Gaussian models (cameras from outside_views.py)
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"   # REPO, P (pipesplat), PY, GS, TORCH_HOME from scripts/paths.py
S=$P/review/outside_views_gs.py
V=$GS/review/outside
G=$GS/g0
$PY $S $V/concrete_2 $G/concrete_2/run_g0a "$@" 2>&1 | grep -v Warn
$PY $S $V/concrete_lab $G/concrete_lab/run_g0a "$@" 2>&1 | grep -v Warn
$PY $S $V/inpipe $G/inpipe_vision/run_g0ala_s20 "$@" 2>&1 | grep -v Warn
