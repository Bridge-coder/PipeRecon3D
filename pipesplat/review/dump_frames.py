"""docs/11: render every held-out and block frame of the given runs (train_gs.py --eval-only --ruler2 --dump), one
run after another, with each run's own model flags taken from its summary file - the frames for compare_video.py.
Runs in WSL.

usage: /opt/gsenv/bin/python pipesplat/review/dump_frames.py DATA/RUN [DATA/RUN ...]
writes /mnt/d/pipe3d_gs/review/video/<DATA>/<RUN>/{held,block}_<frame>_{G,real,B}.jpg (+ the run's summary_dump.json)
"""
import json
import os
import subprocess
import sys

G = "/mnt/d/pipe3d_gs/g0"
OUT = "/mnt/d/pipe3d_gs/review/video"
T = "/mnt/c/Users/PC/Desktop/PipeRecon3D/pipesplat/train_gs.py"
FLAGS = ("anchor", "lamp", "lamp_ambient", "raw", "pose_opt", "opaque")          # switches that change the renderer
VALUES = ("lamp_smooth_mm", "surface", "points_every", "baseline", "thin_mm")

for item in sys.argv[1:]:
    ds, run = item.split("/")
    rd = os.path.join(G, ds, run)
    summ = next(os.path.join(rd, f) for f in ("summary.json", "summary_r2.json") if os.path.exists(os.path.join(rd, f)))
    args = json.load(open(summ))["args"]
    cmd = ["/opt/gsenv/bin/python", T, os.path.join(G, ds), rd, "--eval-only", "--ruler2", "--suffix", "_dump",
           "--dump", os.path.join(OUT, ds, run)]
    cmd += [f"--{k.replace('_', '-')}" for k in FLAGS if args.get(k)]
    for k in VALUES:
        if args.get(k) is not None and not (k == "points_every" and args[k] == 1):
            cmd += [f"--{k.replace('_', '-')}", str(args[k])]
    print(" ".join(cmd[2:]), flush=True)
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                       env=dict(os.environ, TORCH_HOME="/mnt/d/pipe3d_gs/torch_cache"))
    n = len([f for f in os.listdir(os.path.join(OUT, ds, run)) if f.endswith("_G.jpg")]) if os.path.isdir(os.path.join(OUT, ds, run)) else 0
    print(f"  {ds}/{run}: exit {r.returncode}, {n} frames" + ("" if r.returncode == 0 else "\n" + r.stderr[-1500:]), flush=True)
