"""F1: put every stream of an extracted InPipe-SLAM bag on one clock.

Common clock = RealSense stamps (the realsense driver keeps them on host epoch time; colour, depth and the
RealSense IMU share it). The other streams:
  - LiDAR points + LiDAR IMU: stamped by the LiDAR's own clock (seconds since power-on). First mapping =
    lower envelope of (receive time - stamp) over time (a line: offset + clock drift), fitted on the
    high-rate LiDAR IMU. Then refined by cross-correlating the angular-rate magnitude |w| of the LiDAR IMU
    with that of the RealSense IMU (rotation-invariant, so no extrinsics are needed yet).
  - rear camera (usbcam): its stamps come from another clock (~1 s off); lower envelope only, then
    refined against the encoders through the rear AprilTag.
  - wheel encoders: no stamp, only receive time; lag found by fitting the encoder travel to the front
    AprilTag distance seen by the colour camera (colour clock), checked against the vision SfM travel.

usage: python -I f1_time.py EXTRACT_DIR RESULTS_DIR      (RESULTS_DIR holds tag/ and vision/ from earlier steps)
writes RESULTS_DIR/fusion/time_offsets.json and RESULTS_DIR/fusion/f1_time.png
"""
import csv
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from timebase import to_common  # noqa: E402

EX, RES = sys.argv[1:3]
OUT = os.path.join(RES, "fusion")
os.makedirs(OUT, exist_ok=True)


def rows_of(name):
    with open(os.path.join(EX, name)) as f:
        return list(csv.DictReader(f))


def envelope(stamp, log, bin_s=2.0):
    """Line through the lower envelope of (log - stamp) against stamp: offset at t0 and drift."""
    off = (log - stamp).astype(float)
    t = (stamp - stamp[0]) / 1e9
    b = np.floor(t / bin_s).astype(int)
    tb, ob = [], []
    for k in np.unique(b):
        m = b == k
        j = np.argmin(off[m])
        tb.append(t[m][j])
        ob.append(off[m][j])
    tb, ob = np.array(tb), np.array(ob)
    keep = np.ones(len(tb), bool)
    for _ in range(3):
        p = np.polyfit(tb[keep], ob[keep], 1)
        res = ob - np.polyval(p, tb)
        keep = res < 2e6            # bins whose minimum is > 2 ms above the line had no low-latency message
    lat = off - np.polyval(p, t)
    return {"kind": "stamp", "offset_ns": float(p[1]), "drift": float(p[0]) / 1e9, "t0_ns": int(stamp[0]),
            "latency_ms_p50": float(np.median(lat) / 1e6), "latency_ms_p99": float(np.percentile(lat, 99) / 1e6)}


def stamped(name):
    r = rows_of(name)
    return r, np.array([int(x["stamp_ns"]) for x in r], np.int64), np.array([int(x["log_time_ns"]) for x in r], np.int64)


offsets = {}
report = []
# --- RealSense streams define the clock
for key, name in (("imu_camera", "imu_camera.csv"), ("color", "images/color_index.csv"), ("depth", "images/depth_index.csv")):
    r, st, lg = stamped(name)
    env = envelope(st, lg)
    offsets[key] = {"kind": "stamp", "offset_ns": 0.0, "drift": 0.0, "t0_ns": int(st[0])}
    report.append(f"{key}: RealSense clock (reference); receive - stamp: floor {env['offset_ns'] / 1e6:.1f} ms, "
                  f"drift {env['drift'] * 1e6:.1f} ppm, latency above floor p50 {env['latency_ms_p50']:.1f} ms")

# --- LiDAR clock: envelope on the LiDAR IMU, then refine with the gyro correlation
ri, st_i, lg_i = stamped("imu_lidar.csv")
env_l = envelope(st_i, lg_i)
G_l = np.array([[float(x[k]) for k in ("gx", "gy", "gz")] for x in ri])
rc, st_c, lg_c = stamped("imu_camera.csv")
G_c = np.array([[float(x[k]) for k in ("gx", "gy", "gz")] for x in rc])


def binned(t_ns, G, dt=5e6):
    """Average samples in 5 ms bins (the LiDAR driver updates the three gyro axes one at a time)."""
    b = np.floor((t_ns - t_ns[0]) / dt).astype(np.int64)
    u, inv = np.unique(b, return_inverse=True)
    S = np.zeros((len(u), 3))
    np.add.at(S, inv, G)
    n = np.bincount(inv).astype(float)
    return t_ns[0] + (u * dt + dt / 2).astype(np.int64), S / n[:, None]


tl, Gl = binned(to_common(env_l, st_i), G_l)
tc, Gc = st_c, G_c
# gyro bias from the first 20 s (robot standing still)
t_start = max(tl[0], tc[0])
Gl -= Gl[(tl - t_start) < 20e9].mean(0)
Gc -= Gc[(tc - t_start) < 20e9].mean(0)
grid = np.arange(t_start + 21e9, min(tl[-1], tc[-1]) - 1e9, 5e6).astype(np.int64)   # 200 Hz, moving part
mc = np.interp(grid, tc, np.linalg.norm(Gc, axis=1))
lags = np.arange(-400, 401) * 1e6     # +-400 ms in 1 ms steps
corr = []
for lag in lags:
    ml = np.interp(grid + lag, tl, np.linalg.norm(Gl, axis=1))
    corr.append(np.corrcoef(mc, ml)[0, 1])
corr = np.array(corr)
k = int(np.argmax(corr))
# parabolic refinement of the peak
if 0 < k < len(corr) - 1:
    y0, y1, y2 = corr[k - 1], corr[k], corr[k + 1]
    frac = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
else:
    frac = 0.0
best_lag = (lags[k] + frac * 1e6)
# The LiDAR IMU is dominated by drive vibration, so the raw |w| correlation peak is broad (checked on
# 2026-10-15: 0.62 at -7 ms against 0.59 elsewhere). The envelope mapping is kept as the LiDAR clock; the
# correlation (raw and after a 0.5 s moving average) is only reported as a consistency check.
k50 = np.ones(100) / 100
mc_s = np.convolve(mc, k50, mode="same")
corr_s = []
for lag in lags:
    ml = np.convolve(np.interp(grid + lag, tl, np.linalg.norm(Gl, axis=1)), k50, mode="same")
    corr_s.append(np.corrcoef(mc_s, ml)[0, 1])
corr_s = np.array(corr_s)
ks = int(np.argmax(corr_s))
lidar_rule = dict(env_l)
offsets["imu_lidar"] = lidar_rule
offsets["lidar"] = dict(lidar_rule)        # point clouds share the LiDAR clock
report.append(f"LiDAR clock: envelope offset {env_l['offset_ns'] / 1e9:.6f} s, drift {env_l['drift'] * 1e6:.1f} ppm "
              f"(used as is). Check: raw |w| correlation peak {corr[k]:.3f} at {best_lag / 1e6:+.1f} ms "
              f"(best outside +-30 ms {np.max(corr[np.abs(lags - lags[k]) > 30e6]):.3f}); 0.5 s smoothed peak "
              f"{corr_s[ks]:.3f} at {lags[ks] / 1e6:+.0f} ms")

# --- encoders: lag from the front AprilTag (colour clock), check with the vision travel
enc = rows_of("encoders.csv")
lg_e = np.array([int(r["log_time_ns"]) for r in enc], np.int64)
s_e = np.array([(float(r["v6"]) - float(r["v7"])) / 2 for r in enc])
s_e -= s_e[0]


def fit_lag(t_ref, y_ref, lag_grid):
    """y_ref ~ k * s_enc(t_ref + lag) + c ; return best lag (ns), k, rms residual (m) and the curve."""
    out = []
    for lag in lag_grid:
        x = np.interp(t_ref + lag, lg_e, s_e)
        A = np.column_stack([x, np.ones_like(x)])
        (kk, cc), *_ = np.linalg.lstsq(A, y_ref, rcond=None)
        out.append((np.sqrt(np.mean((y_ref - kk * x - cc) ** 2)), kk, cc))
    j = int(np.argmin([o[0] for o in out]))
    return lag_grid[j], out[j][1], out[j][0], np.array([o[0] for o in out])


lag_grid = np.arange(-1500, 1501, 5) * 1e6
tagf = list(csv.DictReader(open(os.path.join(RES, "tag", "tag_track_color.csv"))))
col_rows = {int(r["idx"]): r for r in rows_of("images/color_index.csv")}
t_tagf = to_common(offsets["color"], np.array([int(col_rows[int(r["idx"])]["stamp_ns"]) for r in tagf], np.int64))
d_tagf = -0.1 * np.array([float(r["dist_unit"]) for r in tagf])          # 100 mm tag, distance shrinks
lag_f, k_f, rms_f, curve_f = fit_lag(t_tagf, d_tagf, lag_grid)
# encoder sample with receive time T describes the robot at common time T - lag_enc; fit_lag found
# y(t) ~ s_enc(t + lag), i.e. the encoder value received at t + lag belongs to t  ->  lag_enc = lag_f
report.append(f"encoders: lag {lag_f / 1e6:+.0f} ms from the front tag (fit scale {k_f:.4f}, rms {rms_f * 1000:.1f} mm)")

vt = list(csv.DictReader(open(os.path.join(RES, "vision", "vision_travel.csv"))))
t_vis = to_common(offsets["color"], np.array([int(col_rows[int(r["color_idx"])]["stamp_ns"]) for r in vt], np.int64))
s_vis = np.array([float(r["s_vision_m"]) for r in vt])
lag_v, k_v, rms_v, curve_v = fit_lag(t_vis, s_vis, lag_grid)
report.append(f"encoders: lag {lag_v / 1e6:+.0f} ms from the vision SfM travel (fit scale {k_v:.4f}, rms {rms_v * 1000:.1f} mm)")
# both residual curves are flat near their minima (the robot moves slowly and steadily); take the minimum
# of their sum, and as uncertainty the lag range where that sum stays within 5% of its minimum
comb = curve_f ** 2 / curve_f.min() ** 2 + curve_v ** 2 / curve_v.min() ** 2
j = int(np.argmin(comb))
lag_f = lag_grid[j]
within = lag_grid[comb < 1.05 * comb[j]]
offsets["encoders"] = {"kind": "log", "lag_ns": float(lag_f)}
report.append(f"encoders: lag used {lag_f / 1e6:+.0f} ms (combined minimum; within 5% of it: {within.min() / 1e6:+.0f} .. "
              f"{within.max() / 1e6:+.0f} ms, i.e. about +-{0.02 * (within.max() - within.min()) / 2e9 * 1000:.1f} mm at 2 cm/s)")

# --- rear camera: envelope, then refine through the rear tag against the (now timed) encoders
ru, st_u, lg_u = stamped("images/usbcam_index.csv")
env_u = envelope(st_u, lg_u)
tagr = list(csv.DictReader(open(os.path.join(RES, "tag", "tag_track_usbcam.csv"))))
t_tagr_env = to_common(env_u, np.array([int(r["stamp_ns"]) for r in tagr], np.int64))
d_tagr = 0.1 * np.array([float(r["dist_unit"]) for r in tagr])
# with the encoder lag known, the remaining shift belongs to the rear camera's mapping
lag_r, k_r, rms_r, curve_r = fit_lag(t_tagr_env, d_tagr, lag_grid + lag_f)
usb_rule = dict(env_u)
usb_rule["offset_ns"] = env_u["offset_ns"] + (lag_r - lag_f)
offsets["usbcam"] = usb_rule
report.append(f"rear camera: envelope offset {env_u['offset_ns'] / 1e9:.4f} s, drift {env_u['drift'] * 1e6:.1f} ppm; "
              f"extra shift from the rear tag {(lag_r - lag_f) / 1e6:+.0f} ms (fit scale {k_r:.4f}, rms {rms_r * 1000:.1f} mm)")

with open(os.path.join(OUT, "time_offsets.json"), "w") as f:
    json.dump(offsets, f, indent=1)
with open(os.path.join(OUT, "f1_time.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(report) + "\n")
print("\n".join(report))

fig, ax = plt.subplots(3, 1, figsize=(12, 10))
ax[0].plot(lags / 1e6, corr); ax[0].axvline(best_lag / 1e6, color="r", lw=.8)
ax[0].set_xlabel("lag (ms)"); ax[0].set_ylabel("corr |w| LiDAR IMU vs RealSense IMU"); ax[0].grid(alpha=.3)
ax[1].plot(lag_grid / 1e6, curve_f * 1000, label="front tag (colour clock)")
ax[1].plot(lag_grid / 1e6, curve_v * 1000, label="vision SfM travel (colour clock)")
ax[1].plot((lag_grid + lag_f - lag_f) / 1e6, curve_r * 1000, label="rear tag (rear-camera clock, after envelope)")
ax[1].set_xlabel("encoder lag (ms)"); ax[1].set_ylabel("rms residual (mm)"); ax[1].set_ylim(0, 30); ax[1].legend(); ax[1].grid(alpha=.3)
sl = slice(None)
tt = (grid - grid[0]) / 1e9
ax[2].plot(tt, mc, lw=.6, label="|w| RealSense IMU")
ax[2].plot(tt, np.interp(grid + best_lag, tl, np.linalg.norm(Gl, axis=1)), lw=.6, label="|w| LiDAR IMU, aligned")
ax[2].set_xlabel("time from 21 s after start (s)"); ax[2].set_ylabel("rad/s"); ax[2].legend(); ax[2].grid(alpha=.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "f1_time.png"), dpi=85)
