"""docs/10 X8 diagnosis: where the anchored Gaussians of a train_gs.py --anchor run sit relative to the measured mesh:
signed offset along the radius (+ = outward, behind the mesh as seen from the axis) of every Gaussian centre from the
plane of its nearest surface triangle, weighted by opacity.

usage: /opt/gsenv/bin/python pipesplat/review/gs_offset.py SCENE_DIR RUN_DIR
"""
import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

scene, run = sys.argv[1], sys.argv[2]
sf = np.load(os.path.join(scene, "surface.npz"))
c, n = sf["centres"].astype(np.float64), sf["normals"].astype(np.float64)
sv = torch.load(os.path.join(run, "model.pt"))
mu = sv["means"].numpy().astype(np.float64)
op = torch.sigmoid(sv["opacities"]).numpy()
sc = torch.exp(sv["scales"]).numpy()
_, idx = cKDTree(c).query(mu, k=1)
off = ((mu - c[idx]) * n[idx]).sum(1)
rad = np.stack([np.zeros(len(c)), c[:, 1], c[:, 2]], 1)
rad /= np.linalg.norm(rad, axis=1, keepdims=True)
sgn = np.sign((n * rad).sum(1))[idx]
offr = off * sgn * 1000                        # mm, + outward
wall = mu[:, 0] < 4.3


def wq(x, wt, qs):
    o = np.argsort(x)
    cw = np.cumsum(wt[o]) / wt.sum()
    return [round(float(x[o][np.searchsorted(cw, q)]), 2) for q in qs]


print(f"{len(mu):,} Gaussians; opacity-weighted radial offset from the mesh (mm, + outward) p10/25/50/75/90: "
      f"{wq(offr[wall], op[wall], [0.1, 0.25, 0.5, 0.75, 0.9])}")
print(f"  unweighted p10/50/90 {np.percentile(offr[wall], [10, 50, 90]).round(2).tolist()}; at the clamp (|off| > 1.95 mm): "
      f"{(np.abs(offr[wall]) > 1.95).mean():.1%}")
for lo, hi in ((0.0, 0.3), (0.3, 0.7), (0.7, 0.99), (0.99, 1.01)):
    m = wall & (op >= lo) & (op < hi)
    print(f"  opacity {lo}-{hi}: {m.mean():.1%} of the Gaussians, median offset {np.median(offr[m]):+.2f} mm")
th = np.degrees(np.arctan2(-mu[:, 1], mu[:, 2]))
for lo in range(-180, 180, 45):
    m = wall & (th >= lo) & (th < lo + 45) & (op > 0.3)
    print(f"  theta {lo:+4d}..{lo + 45:+4d} deg: median offset {np.median(offr[m]):+.2f} mm (opacity > 0.3)")
for x0 in (0.5, 1.5, 2.5, 3.5):
    m = wall & (mu[:, 0] >= x0) & (mu[:, 0] < x0 + 1) & (op > 0.3)
    print(f"  x {x0}..{x0 + 1} m: median offset {np.median(offr[m]):+.2f} mm")
