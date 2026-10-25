#!/bin/bash
# docs/11: train_gs.py runs strictly one after another (one GPU job at a time). Each argument is one job,
# "DATA RUN [train_gs.py options]" (DATA = folder under /mnt/d/pipe3d_gs/g0); a job whose RUN already has
# summary_r2.json is skipped, so the queue can be started again after an interruption.
# usage (from PowerShell): wsl -d pipe3d -u root -- bash .../pipesplat/review/queue.sh "concrete_2 run_3dgs --baseline 3dgs" ...
export TORCH_HOME=/mnt/d/pipe3d_gs/torch_cache
PY=/opt/gsenv/bin/python
T=/mnt/c/Users/PC/Desktop/PipeRecon3D/pipesplat/train_gs.py
G=/mnt/d/pipe3d_gs/g0
for job in "$@"; do
  set -- $job
  if [ -f "$G/$1/$2/summary_r2.json" ]; then echo "skip $1/$2 (done)"; continue; fi
  # pause between jobs while /mnt/d/pipe3d_gs/queue.pause exists (a short GPU job can run in the gap)
  while [ -f /mnt/d/pipe3d_gs/queue.pause ]; do sleep 20; done
  echo "$(date +%F_%T) start $job"
  $PY $T "$G/$1" "$G/$1/$2" "${@:3}" > /dev/null 2>&1
  echo "$(date +%F_%T) end $1/$2 exit $?"
  tail -4 "$G/$1/$2/train.log"
done
echo "$(date +%F_%T) queue done"
