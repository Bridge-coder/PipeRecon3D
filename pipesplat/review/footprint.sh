#!/bin/bash
# docs/10: on-screen size of the trained Gaussians (capacity check)
export TORCH_HOME=/mnt/d/pipe3d_gs/torch_cache
PY=/opt/gsenv/bin/python
D=/mnt/c/Users/PC/Desktop/PipeRecon3D/pipesplat/diag_footprint.py
G=/mnt/d/pipe3d_gs/g0
for r in concrete_2/run_g0 concrete_2/run_g0a inpipe_172011/run_g0a inpipe_172011/run_g0ala_s20; do
  echo "=== $r"; $PY $D $G/${r%%/*} $G/$r 2>&1 | grep -v Warning
done
