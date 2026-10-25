"""docs/11: render every held-out and block frame of the given runs (train_gs.py --eval-only --ruler2 --dump), one
run after another, with each run's own model flags taken from its summary file - the frames for compare_video.py.
Runs in WSL. (The same as pipesplat/rescore.py --suffix _dump --dump, without the score table.)

usage: /opt/gsenv/bin/python pipesplat/review/dump_frames.py DATA/RUN [DATA/RUN ...]
writes GS_ROOT/review/video/<DATA>/<RUN>/{held,block}_<frame>_{G,real,B}.jpg (+ the run's summary_dump.json)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rescore  # noqa: E402

OUT = os.path.join(rescore.paths.GS_ROOT, "review", "video")

for item in sys.argv[1:]:
    ds, run = item.split("/")
    code, err = rescore.score(ds, run, "_dump", os.path.join(OUT, ds, run))
    d = os.path.join(OUT, ds, run)
    n = len([f for f in os.listdir(d) if f.endswith("_G.jpg")]) if os.path.isdir(d) else 0
    print(f"  {ds}/{run}: exit {code}, {n} frames" + ("" if code == 0 else "\n" + err), flush=True)
