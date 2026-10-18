"""LiDAR single-modal step 1: fit the pipe in every LiDAR scan.

InPipe-SLAM sequence 172011: Hesai 128-line LiDAR (frame "hesai_lidar"), z axis pointing along the pipe,
gravity along +y (the IMU reads ay = -9.78 at rest, so -y is up). The pipe is about 200 mm across and has
a slot along the bottom through which the floor (~0.28 m below the LiDAR) is visible.

For each scan:
  1. keep finite points with lo < z' < hi along the current axis estimate and within RMAX of it
     (drops the robot body, the floor seen through the slot and whatever lies beyond the pipe end)
  2. cylinder fit (pipe_geometry.fit_cylinder): first with a 30 mm inlier band, then 20 mm (the pipe is
     about 206 x 185 mm, so a 10 mm band would cut off the ends of the ellipse and hide the ovality)
  3. from the fit: diameter, rms, inlier share, axis direction in the LiDAR frame (tilt), the offset of
     the LiDAR origin from the axis (where the sensor sits in the cross-section)
  4. the slot: largest empty angular gap of the inliers around the axis (angle 0 = up, positive towards
     +x side), its centre is a roll reference that vision and LiDAR both see
  5. ellipse (Fitzgibbon direct fit) of the inlier cross-section -> major/minor diameter, ovality
The next scan starts from this scan's axis.

usage: python -I lidar_scan_fit.py EXTRACT_DIR [--first A] [--last B] [--every N] [--lo M] [--hi M] [--out NAME]
writes EXTRACT_DIR/lidar_fit.csv (or NAME). --lo/--hi: fit window along the axis in metres (default 0.15-1.0;
0.15-0.45 keeps the fit inside one straight leg of a bent pipe most of the time)
"""
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipe_geometry import fit_cylinder, fit_ellipse  # noqa: E402

LO, HI, RMAX = 0.15, 1.0, 0.2   # metres along / around the axis used for the fit

args = sys.argv[1:]


def opt(name, default):
    if name in args:
        i = args.index(name)
        v = int(args[i + 1])
        del args[i:i + 2]
        return v
    return default


first, last, every = opt("--first", 0), opt("--last", 10 ** 9), opt("--every", 1)
for flag in ("--lo", "--hi"):
    if flag in args:
        i = args.index(flag)
        if flag == "--lo":
            LO = float(args[i + 1])
        else:
            HI = float(args[i + 1])
        del args[i:i + 2]
out_name = "lidar_fit.csv"
if "--out" in args:
    i = args.index("--out")
    out_name = args[i + 1]
    del args[i:i + 2]
EX = args[0]

with open(os.path.join(EX, "lidar_index.csv")) as f:
    index = list(csv.DictReader(f))


def largest_gap(theta):
    t = np.sort(theta)
    gaps = np.diff(np.concatenate([t, [t[0] + 2 * np.pi]]))
    k = int(np.argmax(gaps))
    centre = t[k] + gaps[k] / 2
    return (centre + np.pi) % (2 * np.pi) - np.pi, gaps[k]


axis_p, axis_d = np.zeros(3), np.array([0.0, 0, 1])
UP = np.array([0.0, -1, 0])
out_rows = []
t0 = time.time()
sel_idx = [r for r in index if first <= int(r["idx"]) <= last and int(r["idx"]) % every == 0]
for n, row in enumerate(sel_idx):
    i = int(row["idx"])
    a = np.load(os.path.join(EX, "lidar", f"{i:06d}.npy"))
    P = np.column_stack([a["x"], a["y"], a["z"]]).astype(float)
    P = P[np.isfinite(P).all(1) & (np.abs(P).sum(1) > 0)]
    Q = P - axis_p
    s = Q @ axis_d
    rad = np.linalg.norm(Q - np.outer(s, axis_d), axis=1)
    sel = (P @ axis_d > LO) & (P @ axis_d < HI) & (rad < RMAX)
    rec = {"idx": i, "log_time_ns": row["log_time_ns"], "stamp_ns": row["stamp_ns"], "n_valid": len(P),
           "n_sel": int(sel.sum())}
    try:
        f1 = fit_cylinder(P[sel], d0=axis_d, thresh=0.03, max_points=6000)
        f = fit_cylinder(P[sel], d0=f1["dir"], thresh=0.02, max_points=6000)
    except (ValueError, np.linalg.LinAlgError) as e:
        rec["error"] = str(e)
        out_rows.append(rec)
        continue
    d = f["dir"] if f["dir"][2] > 0 else -f["dir"]
    # axis point where it crosses the plane through the LiDAR origin perpendicular to the axis
    p0 = f["point"] - (f["point"] @ d) * d
    # cross-section frame: up = gravity-up projected, side = d x up
    up = UP - (UP @ d) * d
    up /= np.linalg.norm(up)
    side = np.cross(d, up)
    W = P[sel][f["inliers"]] - p0
    u, v = W @ side, W @ up
    th = np.arctan2(u, v)  # 0 = up, +90 deg = side (+x-ish)
    gap_c, gap_w = largest_gap(th)
    # ellipse on the wall points (the slot leaves a gap, the fit still closes the curve)
    try:
        ex, ey, ea, eb, eang = fit_ellipse(u, v)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        ex = ey = ea = eb = eang = np.nan
    rec.update({
        "diameter_mm": 2000 * f["radius"], "rms_mm": 1000 * f["rms"], "inlier_share": f["n_inliers"] / sel.sum(),
        "dir_x": d[0], "dir_y": d[1], "dir_z": d[2],
        # tilt of the LiDAR z axis relative to the pipe axis: yaw about up (x), pitch about side (y)
        "yaw_deg": np.degrees(np.arctan2(d[0], d[2])), "pitch_deg": np.degrees(np.arctan2(-d[1], d[2])),
        # LiDAR origin relative to the axis, in the cross-section frame (mm): side (+x-ish) and up
        "off_side_mm": -1000 * (p0 @ side), "off_up_mm": -1000 * (p0 @ up),
        "slot_centre_deg": np.degrees(gap_c), "slot_width_deg": np.degrees(gap_w),
        "ell_major_mm": 2000 * ea, "ell_minor_mm": 2000 * eb, "ell_angle_deg": np.degrees(eang),
        "ovality_pct": 100 * (ea - eb) / f["radius"] if np.isfinite(ea) else np.nan,
        "ell_dx_mm": 1000 * ex, "ell_dy_mm": 1000 * ey,
    })
    out_rows.append(rec)
    axis_p, axis_d = p0, d
    if n % 50 == 0:
        print(f"scan {i}: D {rec['diameter_mm']:.1f} mm, rms {rec['rms_mm']:.2f}, inl {rec['inlier_share']:.0%}, "
              f"yaw {rec['yaw_deg']:.2f} pitch {rec['pitch_deg']:.2f} deg, off {rec['off_side_mm']:.1f}/"
              f"{rec['off_up_mm']:.1f} mm, slot {rec['slot_centre_deg']:.1f} deg w {rec['slot_width_deg']:.1f}, "
              f"oval {rec['ovality_pct']:.1f}%  [{time.time() - t0:.0f} s]", flush=True)

keys = []
for r in out_rows:
    keys += [k for k in r if k not in keys]
with open(os.path.join(EX, out_name), "w", newline="") as f:
    w = csv.DictWriter(f, keys)
    w.writeheader()
    w.writerows(out_rows)
print(f"{len(out_rows)} scans in {time.time() - t0:.0f} s (window {LO}-{HI} m) -> {os.path.join(EX, out_name)}")
