"""Quick views of the coloured LiDAR point cloud in the fused metric frame (top view in the plane of the bend,
and a side view), plus the fused camera path. Sanity check that the bend and the colours come out right.

usage: python -I f5_views.py RESULTS_DIR
"""
import csv
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = sys.argv[1]
OUT = os.path.join(RES, "fusion")
legs = json.load(open(os.path.join(OUT, "vision_legs.json")))
a1, a2 = np.array(legs["legs"][0]["dir"]), np.array(legs["legs"][1]["dir"])
n = np.cross(a1, a2)
n /= np.linalg.norm(n)
e1 = a1
e2 = np.cross(n, a1)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import FUSED_BIG  # noqa: E402
with open(os.path.join(FUSED_BIG, "lidar_coloured_world.ply"), "rb") as f:
    while f.readline().strip() != b"end_header":
        pass
    rec = np.frombuffer(f.read(), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
rng = np.random.default_rng(0)
s = rng.choice(len(rec), min(400000, len(rec)), replace=False)
X = np.column_stack([rec["x"][s], rec["y"][s], rec["z"][s]]).astype(float)
C = np.column_stack([rec["r"][s], rec["g"][s], rec["b"][s]]) / 255.0
traj = np.array([[float(r["x_m"]), float(r["y_m"]), float(r["z_m"])] for r in csv.DictReader(open(os.path.join(OUT, "fused_trajectory.csv")))])
o = traj[0]
fig, ax = plt.subplots(2, 1, figsize=(15, 9))
for k, (u, v, lab) in enumerate(((e1, e2, "across, in the bend plane (m)"), (e1, n, "normal to the bend plane (m)"))):
    q = X - o
    order = np.argsort(q @ (n if k == 0 else e2))          # draw far points first
    ax[k].scatter((q @ u)[order], (q @ v)[order], c=C[order], s=0.3, linewidths=0)
    ax[k].plot((traj - o) @ u, (traj - o) @ v, "r-", lw=1.5, label="fused camera path")
    ax[k].set_aspect("equal")
    ax[k].set_xlabel("along leg 1 (m)")
    ax[k].set_ylabel(lab)
    ax[k].grid(alpha=.3)
    ax[k].legend(loc="upper left")
ax[0].set_title("LiDAR wall points coloured by the camera, fused metric frame - view onto the bend plane")
ax[1].set_title("side view")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "f5_views.png"), dpi=90)
