"""F4, LiDAR-only test for a range bias.

In a straight stretch the wall is at the same distance R from the pipe axis everywhere. A constant range
bias b moves every point by b along its beam; seen from (near) the axis, a wall point at range rho then
appears at radius  r = R + b * R / rho  (near points: steep beams, most of b goes radial; far points:
grazing beams, little of it does). Fitting r against R/rho over many scans gives R and b from the LiDAR
alone. Ring-dependent offsets and incidence-angle effects also vary with rho, so the result is checked
per ring group and against the registration (f3_register.py).

usage: python -I f4_lidar_range.py EXTRACT_DIR RESULTS_DIR
"""
import csv
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from timebase import load_offsets, stream_times  # noqa: E402

EX, RES = sys.argv[1:3]
OUT = os.path.join(RES, "fusion")
off = load_offsets(os.path.join(OUT, "time_offsets.json"))
legs = json.load(open(os.path.join(OUT, "vision_legs.json")))
fits = {int(r["idx"]): r for r in csv.DictReader(open(os.path.join(EX, "lidar_fit_short.csv"))) if r.get("diameter_mm")}
D_med = np.median([float(r["diameter_mm"]) for r in fits.values()])
tS, rS = stream_times(EX, off, "lidar")
tE, rE = stream_times(EX, off, "encoders")
sE = np.array([(float(r["v6"]) - float(r["v7"])) / 2 for r in rE])
sE -= sE[0]
s_bend = legs["bend_point_path_units"] * 1.730 / legs["path_units"]       # camera travel at the bend (m)
UP = np.array([0.0, -1, 0])
rows = []
for k, (t, row) in enumerate(zip(tS, rS)):
    if k % 4:
        continue
    i = int(row["idx"])
    f = fits.get(i)
    if f is None or abs(float(f["diameter_mm"]) - D_med) > 12 or float(f["inlier_share"]) < 0.5:
        continue
    s_now = np.interp(t, tE, sE)
    # straight legs only: fit window (and the stretch used here) clear of the bend
    if not (s_now < s_bend - 0.9 or s_now > s_bend + 0.1):
        continue
    a = np.load(os.path.join(EX, "lidar", f"{i:06d}.npy"))
    P = np.column_stack([a["x"], a["y"], a["z"]]).astype(float)
    ok = np.isfinite(P).all(1) & (np.abs(P).sum(1) > 0)
    P, ring = P[ok], a["ring"][ok]
    d = np.array([float(f["dir_x"]), float(f["dir_y"]), float(f["dir_z"])])
    up = UP - (UP @ d) * d
    up /= np.linalg.norm(up)
    side = np.cross(d, up)
    p0 = (float(f["off_side_mm"]) * side + float(f["off_up_mm"]) * up) / -1000.0
    Q = P - p0
    s = Q @ d
    r = np.linalg.norm(Q - np.outer(s, d), axis=1)
    rho = np.linalg.norm(P, axis=1)
    R = float(f["diameter_mm"]) / 2000
    m = (s > 0.08) & (s < 0.6) & (np.abs(r - R) < 0.025) & (rho > 0.1)
    rows.append(np.column_stack([r[m], rho[m], s[m], ring[m], np.full(m.sum(), k)]))
A = np.vstack(rows)
r, rho, s, ring = A[:, 0], A[:, 1], A[:, 2], A[:, 3].astype(int)
x = 0.1 / rho
M = np.column_stack([np.ones_like(x), x])


def robust_fit(M, y, it=5):
    w = np.ones(len(y))
    for _ in range(it):
        sol, *_ = np.linalg.lstsq(M * w[:, None], y * w, rcond=None)
        res = y - M @ sol
        sig = 1.4826 * np.median(np.abs(res))
        w = np.where(np.abs(res) < 2.5 * sig, 1.0, 0.0)
    return sol, sig


(R0, c), sig = robust_fit(M, r)
# r = R + b * R / rho = R + (b * R / 0.1) * (0.1 / rho)  ->  b = c * 0.1 / R
b = c * 0.1 / R0
print(f"{len(rows)} straight-leg scans, {len(r):,} wall points: R = {R0 * 1000:.1f} mm (D = {R0 * 2000:.1f} mm), "
      f"range bias b = {b * 1000:+.1f} mm (residual sigma {sig * 1000:.1f} mm)")
groups = []
for lo, hi in ((0, 32), (32, 64), (64, 96), (96, 128)):
    m = (ring >= lo) & (ring < hi)
    if m.sum() > 5000:
        (Rg, cg), sg = robust_fit(M[m], r[m])
        groups.append((lo, hi, int(m.sum()), Rg * 2000, cg * 0.1 / Rg * 1000, float(np.median(rho[m]))))
        print(f"  rings {lo:3d}-{hi - 1:3d}: {m.sum():8,} points, median range {np.median(rho[m]):.2f} m -> D {Rg * 2000:.1f} mm, b {cg * 0.1 / Rg * 1000:+.1f} mm")
json.dump({"points": int(len(r)), "D_mm": R0 * 2000, "range_bias_mm": b * 1000, "sigma_mm": sig * 1000,
           "ring_groups": groups}, open(os.path.join(OUT, "f4_lidar_range.json"), "w"), indent=1)
bins = np.linspace(0.1, 0.6, 26)
k = np.digitize(rho, bins)
med = np.array([np.median(r[k == j]) if (k == j).sum() > 200 else np.nan for j in range(1, len(bins))])
cen = (bins[1:] + bins[:-1]) / 2
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(cen, med * 1000, "o-", label="median radius per range bin")
ax.plot(cen, (R0 + c * 0.1 / cen) * 1000, "r-", label=f"fit R + b R/range: D {R0 * 2000:.1f} mm, b {b * 1000:+.1f} mm")
ax.set_xlabel("LiDAR range (m)"); ax.set_ylabel("distance of wall point from pipe axis (mm)"); ax.grid(alpha=.3); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "f4_lidar_range.png"), dpi=85)
