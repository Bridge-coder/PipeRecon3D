"""F3 refinement: translation t_cl (mainly its axial part) from colour consistency.

Each LiDAR wall point is placed on the pipe wall by the LiDAR alone (scan's own axis fit) plus the wheel
encoders: s = encoder travel + distance ahead of the LiDAR, theta = angle from the slot. That placement does
not depend on t_cl. The same wall cell (3 mm x 1.5 deg) is hit by many scans as the robot creeps forward,
each time at a different distance from the camera. Every hit is projected into the colour frame closest in
time with the candidate t_cl (x_cam = R_cl x_lidar + t_cl, recorded K) and the high-passed grey value is
read there. With the right t_cl all hits of a cell read the same texture; with a wrong one the error in
the image position changes with distance, so the readings of a cell disagree. Objective = pooled variance
of the readings within cells (cells with >= 4 hits).

Images are high-passed (grey minus a 10 px Gaussian blur) because the robot's own lamp makes the same wall
patch brighter when the robot is close to it.

usage: python -I f3c_photometric.py EXTRACT_DIR VISION_WORK RESULTS_DIR [--every N]
writes RESULTS_DIR/fusion/extrinsic_translation_photo.json and f3c_photometric.png
"""
import csv
import json
import os
import sys
import time

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from timebase import load_offsets, stream_times  # noqa: E402
from vision_model import load_cameras  # noqa: E402

args = sys.argv[1:]
EVERY = 4
if "--every" in args:
    i = args.index("--every")
    EVERY = int(args[i + 1])
    del args[i:i + 2]
EX, VW, RES = args
OUT = os.path.join(RES, "fusion")
off = load_offsets(os.path.join(OUT, "time_offsets.json"))
R_cl = np.array(json.load(open(os.path.join(OUT, "extrinsic_rotation.json")))["R_cl"])
t_geo = np.array(json.load(open(os.path.join(OUT, "extrinsic_translation.json")))["t_cl_m"])
legs = json.load(open(os.path.join(OUT, "vision_legs.json")))
cam = load_cameras(EX, VW, off)
t_split = cam["t"][legs["split_camera"]]
ci = json.load(open(os.path.join(EX, "bag_info.json")))["camera_info"]["color"]
K = np.array(ci["k"]).reshape(3, 3)
W, H = ci["width"], ci["height"]
RLO, RHI, DS, DT = 0.20, 0.60, 0.003, 1.5

tE, rE = stream_times(EX, off, "encoders")
sE = np.array([(float(r["v6"]) - float(r["v7"])) / 2 for r in rE])
tS, rS = stream_times(EX, off, "lidar")
tC, rC = stream_times(EX, off, "color")
fits = {int(r["idx"]): r for r in csv.DictReader(open(os.path.join(EX, "lidar_fit_short.csv"))) if r.get("diameter_mm")}
D_med = np.median([float(r["diameter_mm"]) for r in fits.values()])
UP = np.array([0.0, -1, 0])

t0 = time.time()
scans = []          # (x_lidar points, cell ids, image index, leg)
images = []
img_of_frame = {}
for k, (t, row) in enumerate(zip(tS, rS)):
    i = int(row["idx"])
    f = fits.get(i)
    if k % EVERY or f is None or t < cam["t"][0] or t > cam["t"][-1] - 8e9:      # leave out the exit
        continue
    if abs(float(f["diameter_mm"]) - D_med) > 12 or float(f["inlier_share"]) < 0.5:
        continue
    if np.interp(t, tE, sE) - np.interp(tE[0], tE, sE) < 0.005:                  # robot not yet moving
        continue
    a = np.load(os.path.join(EX, "lidar", f"{i:06d}.npy"))
    P = np.column_stack([a["x"], a["y"], a["z"]]).astype(float)
    P = P[np.isfinite(P).all(1) & (np.abs(P).sum(1) > 0)]
    rho = np.linalg.norm(P, axis=1)
    d = np.array([float(f["dir_x"]), float(f["dir_y"]), float(f["dir_z"])])
    up = UP - (UP @ d) * d
    up /= np.linalg.norm(up)
    side = np.cross(d, up)
    p0 = (float(f["off_side_mm"]) * side + float(f["off_up_mm"]) * up) / -1000.0
    Q = P - p0
    s_loc = Q @ d
    rad = np.linalg.norm(Q - np.outer(s_loc, d), axis=1)
    m = (rho > RLO) & (rho < RHI) & (s_loc > 0.05) & (np.abs(rad - float(f["diameter_mm"]) / 2000) < 0.025)
    if m.sum() < 200:
        continue
    th = np.degrees(np.arctan2(Q[m] @ side, Q[m] @ up))
    th = (th - float(f["slot_centre_deg"]) + 360) % 360           # slot at 0 / 360
    keep = (th > 20) & (th < 340)                                   # not the slot itself
    s = np.interp(t, tE, sE) + s_loc[m][keep]
    cell = (np.floor(s / DS).astype(np.int64) * 1000 + np.floor(th[keep] / DT).astype(np.int64))
    j = int(np.argmin(np.abs(tC - t)))
    if j not in img_of_frame:
        g = cv2.imread(os.path.join(EX, "images", "color", f"{int(rC[j]['idx']):06d}.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        hp = g - cv2.GaussianBlur(g, (0, 0), 10)
        img_of_frame[j] = len(images)
        images.append(np.clip(hp, -127, 127).astype(np.int8))
    scans.append((P[m][keep], cell, img_of_frame[j], 0 if t < t_split else 1))
u_cells, inv_all = np.unique(np.concatenate([s[1] for s in scans]), return_inverse=True)
offs = np.cumsum([0] + [len(s[1]) for s in scans])
for n, sc in enumerate(scans):
    scans[n] = (sc[0], inv_all[offs[n]:offs[n + 1]], sc[2], sc[3])
NC = len(u_cells)
print(f"{len(scans)} scans, {offs[-1]:,} wall hits in {NC:,} cells, {len(images)} colour frames cached "
      f"({sum(im.nbytes for im in images) / 1e6:.0f} MB) in {time.time() - t0:.0f} s")


def readings(t_cl, R=R_cl, leg=None):
    """grey readings and their cell ids for one candidate extrinsic"""
    vals, cells = [], []
    for P, cell, ii, lg in scans:
        if leg is not None and lg != leg:
            continue
        xc = P @ R.T + t_cl
        uvw = xc @ K.T
        ok = uvw[:, 2] > 0.03
        u = uvw[ok, 0] / uvw[ok, 2]
        v = uvw[ok, 1] / uvw[ok, 2]
        ok2 = (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
        u, v = u[ok2], v[ok2]
        img = images[ii]
        x0, y0 = u.astype(int), v.astype(int)
        fx, fy = u - x0, v - y0
        g = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy) +
             img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)
        vals.append(g)
        cells.append(cell[ok][ok2])
    return np.concatenate(vals), np.concatenate(cells)


def objective(t_cl, R=R_cl, leg=None):
    g, c = readings(t_cl, R, leg)
    n = np.bincount(c, minlength=NC).astype(float)
    s1 = np.bincount(c, weights=g, minlength=NC)
    s2 = np.bincount(c, weights=g * g, minlength=NC)
    ok = n >= 4
    within = (s2[ok] - s1[ok] ** 2 / n[ok]).sum() / n[ok].sum()
    total = g.var()
    return within / total            # share of the texture variance not explained by the cell -> lower is better


# 1. axial scan with the geometric lateral part
tz = np.arange(-0.02, 0.1201, 0.01)
curves = {}
for name, leg in (("all", None), ("leg 1", 0), ("leg 2", 1)):
    curves[name] = np.array([objective(np.array([t_geo[0], t_geo[1], z]), leg=leg) for z in tz])
    print(f"axial scan ({name}): best t_z = {tz[np.argmin(curves[name])] * 1000:.0f} mm, objective "
          f"{curves[name].min():.4f} (at the geometric 23 mm: {np.interp(t_geo[2], tz, curves[name]):.4f})", flush=True)


def refine(t, leg=None, steps=(0.008, 0.004, 0.002, 0.001)):
    t = t.copy()
    best = objective(t, leg=leg)
    for h in steps:
        improved = True
        while improved:
            improved = False
            for ax in range(3):
                for sgn in (1, -1):
                    c = t.copy()
                    c[ax] += sgn * h
                    v = objective(c, leg=leg)
                    if v < best - 1e-6:
                        best, t, improved = v, c, True
    return t, best


# 2. joint refinement of all three components, from the best axial value
starts = {}
for name, leg in (("all", None), ("leg 1", 0), ("leg 2", 1)):
    t_start = np.array([t_geo[0], t_geo[1], tz[np.argmin(curves[name])]])
    starts[name] = refine(t_start, leg=leg)
    print(f"joint refinement ({name}): t_cl = ({starts[name][0][0] * 1000:.1f}, {starts[name][0][1] * 1000:.1f}, "
          f"{starts[name][0][2] * 1000:.1f}) mm, objective {starts[name][1]:.4f}", flush=True)
t_best = starts["all"][0]

# 3. sharpness of the minimum along each axis (half-width where the objective rises by 2% of its range)
prof = {}
for ax, lab in enumerate("xyz"):
    grid = np.arange(-0.03, 0.0301, 0.003)
    vals = []
    for dlt in grid:
        c = t_best.copy()
        c[ax] += dlt
        vals.append(objective(c))
    prof[lab] = (grid, np.array(vals))
fine_z = np.arange(t_best[2] - 0.03, t_best[2] + 0.0301, 0.002)
fine_vals = np.array([objective(np.array([t_best[0], t_best[1], z])) for z in fine_z])

out = {"t_cl_m": t_best.tolist(), "objective": starts["all"][1],
       "geometric_t_cl_m": t_geo.tolist(),
       "per_leg": {k: {"t_cl_m": v[0].tolist(), "objective": v[1]} for k, v in starts.items()},
       "axial_scan_mm": (tz * 1000).tolist(), "axial_scan": {k: v.tolist() for k, v in curves.items()},
       "scans": len(scans), "hits": int(offs[-1]), "cells": int(NC)}
json.dump(out, open(os.path.join(OUT, "extrinsic_translation_photo.json"), "w"), indent=1)

fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
for name, cv in curves.items():
    ax[0].plot(tz * 1000, cv, "o-", ms=3, label=name)
ax[0].axvline(t_geo[2] * 1000, color="k", ls="--", lw=.8, label="geometric estimate")
ax[0].set_xlabel("axial t_z (mm)"); ax[0].set_ylabel("unexplained texture variance"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[0].set_title("axial scan (lateral = geometric)")
ax[1].plot(fine_z * 1000, fine_vals, "o-", ms=3)
ax[1].set_xlabel("axial t_z (mm), lateral at the refined values"); ax[1].grid(alpha=.3); ax[1].set_title("fine axial profile")
for lab, (grid, vals) in prof.items():
    ax[2].plot(grid * 1000, vals, label=f"t_{lab} offset")
ax[2].set_xlabel("offset from the refined t_cl (mm)"); ax[2].legend(); ax[2].grid(alpha=.3); ax[2].set_title("profiles per component")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "f3c_photometric.png"), dpi=85)
print(f"final t_cl = ({t_best[0] * 1000:.1f}, {t_best[1] * 1000:.1f}, {t_best[2] * 1000:.1f}) mm; total {time.time() - t0:.0f} s")
