"""R1: LiDAR wall points of every scan in the fused world frame, with what the bias calibration (R2) needs.

Per scan: the wall is picked with the scan's own short-window cylinder fit (lidar_fit_short.csv) but with a
generous band (radius 60-160 mm from the local axis), so points past the bend, where the local straight axis
no longer fits, are kept; R2 picks the wall again around the real centerline. Range 0.22-0.50 m (R2 gates
to 0.25-0.45 m after calibrating; the margin shows the bias trend). Every scan is used (f5_fuse used every 2nd).
Placement as f5_fuse.py: x_cam = R_cl x_lidar + t_cl, world = C(t) + R(t)^T x_cam with the fused pose.

output: RECON_BIG/wall_points.npz (X world f4, ring, range, intensity, scan) and the per-scan
LiDAR origin in the world frame.
usage: python -I r1_wall_points.py
"""
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recon_common import BIG, EX, R_cl, t_cl, off, FusedPoses, stream_times  # noqa: E402

RLO, RHI = 0.22, 0.50
fp = FusedPoses()
cam = fp.cam
tS, rS = stream_times(EX, off, "lidar")
fits = {int(r["idx"]): r for r in csv.DictReader(open(os.path.join(EX, "lidar_fit_short.csv"))) if r.get("diameter_mm")}
D_med = np.median([float(r["diameter_mm"]) for r in fits.values()])
UP = np.array([0.0, -1, 0])
acc = {k: [] for k in ("X", "ring", "rho", "inten", "scan")}
origins, scans, times = [], [], []
t0 = time.time()
n_used = 0
for k, (t, row) in enumerate(zip(tS, rS)):
    i = int(row["idx"])
    f = fits.get(i)
    if f is None or t < cam["t"][0] or t > cam["t"][-1]:
        continue
    if abs(float(f["diameter_mm"]) - D_med) > 12 or float(f["inlier_share"]) < 0.5:
        continue
    a = np.load(os.path.join(EX, "lidar", f"{i:06d}.npy"))
    P = np.column_stack([a["x"], a["y"], a["z"]]).astype(float)
    ok = np.isfinite(P).all(1) & (np.abs(P).sum(1) > 0)
    rho = np.linalg.norm(P, axis=1)
    d = np.array([float(f["dir_x"]), float(f["dir_y"]), float(f["dir_z"])])
    up = UP - (UP @ d) * d
    up /= np.linalg.norm(up)
    side = np.cross(d, up)
    p0 = (float(f["off_side_mm"]) * side + float(f["off_up_mm"]) * up) / -1000.0
    Q = P - p0
    s_loc = Q @ d
    rad = np.linalg.norm(Q - np.outer(s_loc, d), axis=1)
    m = ok & (rho > RLO) & (rho < RHI) & (s_loc > 0.05) & (rad > 0.06) & (rad < 0.16)
    if m.sum() < 100:
        continue
    Rw, Cw, _ = fp.pose(t)
    xc = P[m] @ R_cl.T + t_cl
    acc["X"].append((Cw + xc @ Rw).astype(np.float32))
    acc["ring"].append(a["ring"][m].astype(np.uint8))
    acc["rho"].append(rho[m].astype(np.float32))
    acc["inten"].append(a["intensity"][m].astype(np.float32))
    acc["scan"].append(np.full(m.sum(), n_used, np.int32))
    origins.append(Cw + t_cl @ Rw)
    scans.append(i)
    times.append(t)
    n_used += 1
    if n_used % 200 == 0:
        print(f"scan {i}: {n_used} scans, {sum(len(x) for x in acc['rho']):,} points [{time.time() - t0:.0f} s]", flush=True)
for kk in acc:
    acc[kk] = np.concatenate(acc[kk])
np.savez(os.path.join(BIG, "wall_points.npz"), origin=np.array(origins), scan_idx=np.array(scans),
         scan_t=np.array(times, np.int64), **acc)
print(f"{n_used} scans, {len(acc['rho']):,} points -> {os.path.join(BIG, 'wall_points.npz')} [{time.time() - t0:.0f} s]")
