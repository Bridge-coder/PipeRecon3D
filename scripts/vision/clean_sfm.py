"""Remove degenerate 3-D points from an OpenMVG reconstruction before it is handed to OpenMVS.

Usage: clean_sfm.py CONVERTER_EXE SFM_DATA_BIN OUT_JSON

A point that lies behind a camera observing it, or almost at that camera's centre (closer than 5% of the
median depth of the points seen by that camera), is a triangulation error. One such point is enough to
make OpenMVS 2.4.0 compute an absurd size ratio between two views ("619(5pts,242666.28scl)" in its log)
and DensifyPointCloud then stops with exit code 0xC0000409. Poses and all other points are kept as they are.
"""
import json
import os
import subprocess
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
converter, sfm_bin, out_json = sys.argv[1:4]
tmp = os.path.splitext(out_json)[0] + "_all.json"
subprocess.run([converter, "-i", sfm_bin, "-o", tmp, "-V", "-I", "-E", "-S"], check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
d = json.load(open(tmp, encoding="utf-8"))
os.remove(tmp)
pose_of = {v["value"]["ptr_wrapper"]["data"]["id_view"]: v["value"]["ptr_wrapper"]["data"]["id_pose"] for v in d["views"]}
pose = {e["key"]: (np.array(e["value"]["rotation"], float), np.array(e["value"]["center"], float)) for e in d["extrinsics"]}

depth = {}                                   # view id -> list of (point index, depth along the viewing direction)
for k, s in enumerate(d["structure"]):
    x = np.array(s["value"]["X"], float)
    for o in s["value"]["observations"]:
        r, c = pose[pose_of[o["key"]]]
        depth.setdefault(o["key"], []).append((k, float((r @ (x - c))[2])))
bad, views_hit = set(), []
for view, items in depth.items():
    z = np.array([b for _, b in items])
    limit = 0.05 * np.median(z)
    hit = [k for k, b in items if b <= limit]
    if hit:
        bad.update(hit)
        views_hit.append((view, len(hit), float(z.min()), float(np.median(z))))
for view, n, zmin, zmed in sorted(views_hit, key=lambda h: h[2])[:5]:        # the five worst cases
    print(f"   view {view}: {n} point(s) too close or behind (smallest depth {zmin:.4f}, median depth {zmed:.3f} units)")
d["structure"] = [s for k, s in enumerate(d["structure"]) if k not in bad]
json.dump(d, open(out_json, "w", encoding="utf-8"))
print(f"removed {len(bad)} of {len(bad) + len(d['structure'])} points seen from {len(views_hit)} view(s); written {out_json}")
