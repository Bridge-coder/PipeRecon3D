"""Radius of the vision sparse model per 30-degree sector around the pipe axis, for one stretch of cameras.

Checks two things against the LiDAR: whether the visual points cover the circumference evenly (if they sit
mostly on the sides of a horizontally wide pipe, a circle fit overestimates the radius and the scale taken
from it is too small), and whether vision also sees the pipe wider than tall (ratio is scale-free).

usage: python -I vision_sector_radius.py WORK_DIR K0 K1
"""
import csv
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipe_geometry import fit_cylinder, fit_ellipse  # noqa: E402

WORK, K0, K1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
from paths import CONVERTER as CONV  # noqa: E402
js = os.path.join(WORK, "sfm", "recon", "sfm_sector.json")
subprocess.run([CONV, "-i", os.path.join(WORK, "sfm", "recon", "sfm_data.bin"), "-o", js, "-V", "-I", "-E", "-S"],
               check=True, stdout=subprocess.DEVNULL)
d = json.load(open(js))
os.remove(js)
frames = {r["file"]: r for r in csv.DictReader(open(os.path.join(WORK, "frames.csv")))}
view = {v["value"]["ptr_wrapper"]["data"]["id_view"]: v["value"]["ptr_wrapper"]["data"] for v in d["views"]}
pose = {e["key"]: (np.array(e["value"]["rotation"]), np.array(e["value"]["center"])) for e in d["extrinsics"]}
cams = sorted((int(frames[v["filename"]]["color_idx"]), pose[v["id_pose"]]) for v in view.values() if v["id_pose"] in pose)
C = np.array([c[1][1] for c in cams])
Rc = np.array([c[1][0] for c in cams])
X = np.array([s["value"]["X"] for s in d["structure"]])
a = C[K1] - C[K0]
a /= np.linalg.norm(a)
q = X - C[K0]
sx = q @ a
L = (C[K1] - C[K0]) @ a
rx = np.linalg.norm(q - np.outer(sx, a), axis=1)
m = (sx > 0) & (sx < 1.3 * L + 0.1 * L)
m &= rx < 3 * np.median(rx[m])
f = fit_cylinder(X[m], d0=a)
P = X[m][f["inliers"]]
down = Rc[K0:K1 + 1, 1, :].mean(0)          # camera +y = image down, used as the vertical reference
down -= (down @ f["dir"]) * f["dir"]
down /= np.linalg.norm(down)
side = np.cross(f["dir"], -down)
W = P - f["point"]
u, v = W @ side, W @ (-down)
th = np.degrees(np.arctan2(u, v))           # 0 = up, +-180 = bottom
r = np.hypot(u, v)
print(f"cameras {K0}..{K1}: {len(P)} wall points, circle radius {f['radius']:.4f} units")
print("sector (deg, 0 = up)   points   median r / circle R")
for lo in range(-180, 180, 30):
    s = (th >= lo) & (th < lo + 30)
    print(f"  {lo:5d}..{lo + 30:4d}   {s.sum():6d}   {np.median(r[s]) / f['radius'] if s.sum() > 5 else float('nan'):.3f}")
x0, y0, A, B, ang = fit_ellipse(u, v)
print(f"ellipse: major/minor = {A / B:.3f}, major axis {np.degrees(ang):.0f} deg from horizontal; "
      f"(A+B)/2 / circle R = {(A + B) / 2 / f['radius']:.3f}")
