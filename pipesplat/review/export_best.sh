#!/bin/bash
# docs/10 section 5: the best Gaussian models as standard 3DGS .ply
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"   # REPO, P (pipesplat), PY, GS, TORCH_HOME from scripts/paths.py
E=$P/export_ply.py
G=$GS/g0
for r in concrete_2/run_g0a concrete_lab/run_g0a inpipe_vision/run_g0ala_s20; do $PY $E $G/$r 2>&1 | grep -v Warn; done
