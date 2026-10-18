"""How much of the OpenMVS dense cloud lies on the pipe wall, stretch by stretch along the (bent) camera path.

For every stretch of SEG cameras: local direction = chord of the smoothed camera path; dense points between
the stretch start and a little beyond its end, within 2.5x the median distance from that line; cylinder fit
on a subsample; share of the slab's points within +-15% of the fitted radius, wall scatter (rms inside
the band) and circumference coverage (share of 10-degree sectors holding wall points).

usage: python -I eval_dense.py WORK_DIR OUT_CSV
"""
import csv
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipe_geometry import fit_cylinder  # noqa: E402
from ply_io import read_ply_vertices  # noqa: E402

WORK, OUT = sys.argv[1:3]
from paths import CONVERTER as CONV  # noqa: E402
js = os.path.join(WORK, "sfm", "recon", "sfm_dense_eval.json")
subprocess.run([CONV, "-i", os.path.join(WORK, "sfm", "recon", "sfm_data.bin"), "-o", js, "-V", "-I", "-E"],
               check=True, stdout=subprocess.DEVNULL)
d = json.load(open(js))
os.remove(js)
frames = {r["file"]: r for r in csv.DictReader(open(os.path.join(WORK, "frames.csv")))}
view = {v["value"]["ptr_wrapper"]["data"]["id_view"]: v["value"]["ptr_wrapper"]["data"] for v in d["views"]}
pose = {e["key"]: np.array(e["value"]["center"]) for e in d["extrinsics"]}
C = np.array([c for _, c in sorted((int(frames[v["filename"]]["color_idx"]), pose[v["id_pose"]])
                                   for v in view.values() if v["id_pose"] in pose)])
k5 = np.ones(5) / 5
Cs = np.column_stack([np.convolve(np.pad(C[:, j], 2, mode="edge"), k5, mode="valid") for j in range(3)])
s_cam = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(Cs, axis=0), axis=1))])

V = read_ply_vertices(os.path.join(WORK, "mvs", "scene_dense.ply"))
X = np.column_stack([V["x"], V["y"], V["z"]]).astype(float)
print(f"{len(X):,} dense points, {len(C)} cameras, path {s_cam[-1]:.3f} units")
SEG = 40
rows = []
rng = np.random.default_rng(0)
for k0 in range(0, len(C) - SEG // 2, SEG // 2):
    k1 = min(len(C), k0 + SEG) - 1
    dvec = Cs[k1] - Cs[k0]
    L = np.linalg.norm(dvec)
    if L < 1e-9:
        continue
    dvec /= L
    q = X - Cs[k0]
    sx = q @ dvec
    rx = np.linalg.norm(q - np.outer(sx, dvec), axis=1)
    slab = (sx > 0) & (sx < L)
    if slab.sum() < 1000:
        continue
    cand = slab & (rx < 2.5 * np.median(rx[slab]))
    sub = np.flatnonzero(cand)
    sub = sub if len(sub) <= 20000 else rng.choice(sub, 20000, replace=False)
    try:
        f = fit_cylinder(X[sub], d0=dvec)
    except ValueError:
        continue
    P = X[slab]
    W = P - f["point"]
    t = W @ f["dir"]
    radial = W - np.outer(t, f["dir"])
    r = np.linalg.norm(radial, axis=1)
    on = np.abs(r - f["radius"]) < 0.15 * f["radius"]
    e1 = np.cross(f["dir"], [0.0, 0, 1] if abs(f["dir"][2]) < 0.9 else [1.0, 0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(f["dir"], e1)
    ang = np.degrees(np.arctan2(radial[on] @ e2, radial[on] @ e1))
    cover = len(np.unique(np.floor((ang + 180) / 10))) / 36
    rows.append({"cams": f"{k0}-{k1}", "s0_units": s_cam[k0], "s1_units": s_cam[k1], "points": int(slab.sum()),
                 "on_wall_share": float(on.mean()), "radius_units": f["radius"],
                 "wall_rms_pct_of_R": 100 * float(np.sqrt(np.mean((r[on] - f["radius"]) ** 2)) / f["radius"]),
                 "circumference_cover": cover})
with open(OUT, "w", newline="") as fo:
    w = csv.DictWriter(fo, list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(f"cams {r['cams']:>8}: {r['points']:7d} pts, on wall {r['on_wall_share']:.0%}, R {r['radius_units']:.3f}, "
          f"scatter {r['wall_rms_pct_of_R']:.1f}% of R, circumference {r['circumference_cover']:.0%}")
tot = sum(r["points"] for r in rows)
print(f"overall on-wall share (point-weighted): {sum(r['points'] * r['on_wall_share'] for r in rows) / tot:.0%}")
