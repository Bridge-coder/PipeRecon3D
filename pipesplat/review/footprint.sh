#!/bin/bash
# docs/10: on-screen size of the trained Gaussians (capacity check)
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"   # REPO, P (pipesplat), PY, GS, TORCH_HOME from scripts/paths.py
D=$P/diag_footprint.py
G=$GS/g0
for r in concrete_2/run_g0 concrete_2/run_g0a inpipe_172011/run_g0a inpipe_172011/run_g0ala_s20; do
  echo "=== $r"; $PY $D $G/${r%%/*} $G/$r 2>&1 | grep -v Warning
done
