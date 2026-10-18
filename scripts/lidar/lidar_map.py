"""LiDAR single-modal step 2: build the pipe map from the per-scan fits (lidar_scan_fit.py) and wheel odometry.

"LiDAR only" here means LiDAR + wheel odometry: in a straight pipe the LiDAR cannot tell how far the robot
moved along the axis, so the axial position of every scan comes from the wheel encoders (default; see the
note at the odometry block on why not /tf), interpolated at the scan's receive time.

Every scan's wall points are expressed in pipe coordinates of that scan's own fitted axis:
  s = s_odom(t_scan) + distance along the axis from the LiDAR origin
  theta = angle around the axis, 0 = up, 180 deg = bottom; with --roll slot the angle is measured from the
          slot instead (slot = 180 deg), which removes roll of the robot inside the pipe
  r = distance from the axis
Only the near part of each scan (NEAR_LO..NEAR_HI along the axis) is used: closest to the sensor, and the
same stretch of wall is then seen by ~(NEAR_HI-NEAR_LO)/speed/0.1 s scans.

Outputs (OUT_DIR):
  wall_points.npz         s, theta, r, intensity, scan index of all accumulated wall points
  profile.csv             per 2 cm slice: n points, circle diameter, ellipse major/minor, ovality, rms
  unrolled_radius.png     r(s, theta) minus the overall median radius
  unrolled_intensity.png  intensity(s, theta)
  profile.png             diameter / ovality / offsets along the pipe

usage: python -I lidar_map.py EXTRACT_DIR OUT_DIR [--every N] [--roll up|slot] [--odom encoders|tf]
"""
import csv
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipe_geometry import fit_circle_2d, fit_ellipse  # noqa: E402


NEAR_LO, NEAR_HI = 0.15, 0.45   # m along the axis from the LiDAR
BAND = 0.02                      # m around the fitted radius
SLICE = 0.02                     # m, slice length of the diameter profile
DS, DTH = 0.005, 2.0             # unrolled map bins: 5 mm x 2 deg

args = sys.argv[1:]
every = 1
if "--every" in args:
    i = args.index("--every")
    every = int(args[i + 1])
    del args[i:i + 2]
roll = "up"
if "--roll" in args:
    i = args.index("--roll")
    roll = args[i + 1]
    del args[i:i + 2]
odom = "encoders"
if "--odom" in args:
    i = args.index("--odom")
    odom = args[i + 1]
    del args[i:i + 2]
EX, OUT = args
os.makedirs(OUT, exist_ok=True)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# --- wheel odometry against receive time.
# The recorded /tf odom_wheels -> base_link is unusable in sequence 172011: the two wheel encoders count
# with opposite signs, and the odometry node turns straight driving into rotation on the spot (x, y stay 0,
# yaw runs through 180 deg while /cmd_vel only asks for forward speed). /encoders carries 8 values per
# message; v6 and v7 are the cumulative distances of the two wheels in metres (v4/v5 = distance per
# message = ticks * pi * 0.1 m / 4096, i.e. 0.1 m wheels, 4096 ticks per turn; 0.019 m/s matches /cmd_vel).
# Travelled distance = (v6 - v7) / 2.
if odom == "encoders":
    enc = read_csv(os.path.join(EX, "encoders.csv"))
    t_odom = np.array([int(r["log_time_ns"]) for r in enc], dtype=np.int64)
    s_odom = np.array([(float(r["v6"]) - float(r["v7"])) / 2 for r in enc])
    s_odom -= s_odom[0]
else:
    tf = [r for r in read_csv(os.path.join(EX, "tf.csv")) if r["parent"] == "odom_wheels" and r["child"] == "base_link"]
    t_odom = np.array([int(r["log_time_ns"]) for r in tf], dtype=np.int64)
    xy = np.array([[float(r["x"]), float(r["y"])] for r in tf])
    s_odom = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])

fits = [r for r in read_csv(os.path.join(EX, "lidar_fit.csv")) if r.get("diameter_mm")]
# drop scans whose fit is off (e.g. near the far pipe end, where the fit window reaches out of the pipe)
D_all = np.array([float(r["diameter_mm"]) for r in fits])
D_med = np.median(D_all)
good = [r for r in fits if abs(float(r["diameter_mm"]) - D_med) < 10 and float(r["inlier_share"]) > 0.5]
print(f"{len(good)} of {len(fits)} scan fits kept (|D - {D_med:.1f}| < 10 mm, inlier share > 0.5)")
fits = good[::every]
t_scan = np.array([int(r["log_time_ns"]) for r in fits], dtype=np.int64)
s_scan = np.interp(t_scan - t_odom[0], t_odom - t_odom[0], s_odom)
slot = np.array([float(r["slot_centre_deg"]) for r in fits])
# the slot angle per scan is noisy (gap edges); smooth it with a running median over 9 scans
slot_s = np.array([np.median(slot[max(0, k - 4):k + 5]) for k in range(len(slot))])

UP = np.array([0.0, -1, 0])
S, TH, R, INT, IDX = [], [], [], [], []
for k, r in enumerate(fits):
    i = int(r["idx"])
    a = np.load(os.path.join(EX, "lidar", f"{i:06d}.npy"))
    P = np.column_stack([a["x"], a["y"], a["z"]]).astype(float)
    ok = np.isfinite(P).all(1) & (np.abs(P).sum(1) > 0)
    P, inten = P[ok], a["intensity"][ok]
    d = np.array([float(r["dir_x"]), float(r["dir_y"]), float(r["dir_z"])])
    up = UP - (UP @ d) * d
    up /= np.linalg.norm(up)
    side = np.cross(d, up)
    # axis point in the plane through the LiDAR origin: the LiDAR sits at (off_side, off_up) from it
    p0 = (float(r["off_side_mm"]) * side + float(r["off_up_mm"]) * up) / -1000.0
    Q = P - p0
    s_loc = Q @ d
    u, v = Q @ side, Q @ up
    rad = np.hypot(u, v)
    R_fit = float(r["diameter_mm"]) / 2000
    m = (s_loc > NEAR_LO) & (s_loc < NEAR_HI) & (np.abs(rad - R_fit) < BAND)
    th = np.degrees(np.arctan2(u[m], v[m]))
    if roll == "slot":
        th = th - slot_s[k] + 180.0
    S.append(s_scan[k] + s_loc[m])
    TH.append((th + 180.0) % 360.0 - 180.0)
    R.append(rad[m])
    INT.append(inten[m])
    IDX.append(np.full(m.sum(), i, np.int32))
S, TH, R, INT, IDX = map(np.concatenate, (S, TH, R, INT, IDX))
np.savez_compressed(os.path.join(OUT, "wall_points.npz"), s=S, theta=TH, r=R, intensity=INT, scan=IDX)
print(f"{len(fits)} scans, {len(S):,} wall points, s from {S.min():.3f} to {S.max():.3f} m; "
      f"odometry path length {s_odom[-1]:.3f} m")

# --- diameter / ovality per slice
rows = []
edges = np.arange(np.floor(S.min() / SLICE) * SLICE, S.max() + SLICE, SLICE)
for lo in edges[:-1]:
    m = (S >= lo) & (S < lo + SLICE)
    if m.sum() < 500:
        continue
    th = np.radians(TH[m])
    u, v = R[m] * np.sin(th), R[m] * np.cos(th)
    cx, cy, rc = fit_circle_2d(np.column_stack([u, v]))
    res = np.hypot(u - cx, v - cy) - rc
    try:
        ex, ey, ea, eb, eang = fit_ellipse(u, v)
    except (np.linalg.LinAlgError, ValueError):
        ea = eb = eang = np.nan
    rows.append({"s_m": lo + SLICE / 2, "n": int(m.sum()), "diameter_mm": 2000 * rc,
                 "rms_mm": 1000 * np.sqrt(np.mean(res ** 2)), "ell_major_mm": 2000 * ea, "ell_minor_mm": 2000 * eb,
                 "ell_angle_deg": np.degrees(eang), "ovality_pct": 100 * (ea - eb) / rc})
with open(os.path.join(OUT, "profile.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0]))
    w.writeheader()
    w.writerows(rows)
D = np.array([r["diameter_mm"] for r in rows])
print(f"diameter over {len(rows)} slices: median {np.median(D):.1f} mm, 5-95% {np.percentile(D, 5):.1f}-"
      f"{np.percentile(D, 95):.1f} mm")

# --- unrolled maps
r_med = np.median(R)
ns = int(np.ceil((S.max() - S.min()) / DS))
nt = int(360 / DTH)
iS = np.clip(((S - S.min()) / DS).astype(int), 0, ns - 1)
iT = np.clip(((TH + 180) / DTH).astype(int), 0, nt - 1)
flat = iT * ns + iS
order = np.argsort(flat)
flat_s, R_s, I_s = flat[order], R[order], INT[order]
uniq, start = np.unique(flat_s, return_index=True)
med_r = np.full(ns * nt, np.nan)
med_i = np.full(ns * nt, np.nan)
for u_, a_, b_ in zip(uniq, start, np.append(start[1:], len(flat_s))):
    med_r[u_] = np.median(R_s[a_:b_])
    med_i[u_] = np.median(I_s[a_:b_])
ext = [S.min(), S.max(), -180, 180]
for name, img, cmap, lim, label in (
        ("unrolled_radius.png", (med_r.reshape(nt, ns) - r_med) * 1000, "RdBu_r", (-10, 10), f"r - {r_med * 1000:.1f} mm (mm)"),
        ("unrolled_intensity.png", med_i.reshape(nt, ns), "gray", tuple(np.nanpercentile(med_i, [2, 98])), "intensity")):
    fig, ax = plt.subplots(figsize=(16, 4.5))
    im = ax.imshow(img, origin="lower", aspect="auto", extent=ext, cmap=cmap, vmin=lim[0], vmax=lim[1],
                   interpolation="nearest")
    ax.set_xlabel("s along the pipe (m, wheel odometry)")
    ax.set_ylabel("angle (deg, 0 = up" + (", slot at 180)" if roll == "slot" else ", 180 = bottom)"))
    plt.colorbar(im, ax=ax, label=label)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), dpi=90)
    plt.close(fig)

fig, ax = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
sp = np.array([r["s_m"] for r in rows])
ax[0].plot(sp, D, ".-", ms=3, label="circle")
ax[0].plot(sp, [r["ell_major_mm"] for r in rows], ".", ms=2, label="ellipse major")
ax[0].plot(sp, [r["ell_minor_mm"] for r in rows], ".", ms=2, label="ellipse minor")
ax[0].set_ylabel("diameter (mm)"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(sp, [r["ovality_pct"] for r in rows], ".-", ms=3); ax[1].set_ylabel("ovality (%)"); ax[1].grid(alpha=.3)
ax[2].plot(s_scan + (NEAR_LO + NEAR_HI) / 2, [float(r["off_side_mm"]) for r in fits], label="side")
ax[2].plot(s_scan + (NEAR_LO + NEAR_HI) / 2, [float(r["off_up_mm"]) for r in fits], label="up")
ax[2].set_ylabel("LiDAR offset from axis (mm)"); ax[2].legend(); ax[2].grid(alpha=.3)
ax[3].plot(s_scan + (NEAR_LO + NEAR_HI) / 2, [float(r["yaw_deg"]) for r in fits], label="yaw")
ax[3].plot(s_scan + (NEAR_LO + NEAR_HI) / 2, [float(r["pitch_deg"]) for r in fits], label="pitch")
ax[3].plot(s_scan + (NEAR_LO + NEAR_HI) / 2, (slot_s - 180 + 180) % 360 - 180, label="slot - 180 (roll)")
ax[3].set_ylabel("deg"); ax[3].set_xlabel("s (m)"); ax[3].legend(); ax[3].grid(alpha=.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "profile.png"), dpi=90)
print(f"outputs in {OUT}")
