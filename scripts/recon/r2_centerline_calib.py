"""R2: pipe centerline through the bend, and self-calibration of the LiDAR range biases.

1. centerline: short cylinder fits (+-3 cm) every 1 cm along the vision centerline, smoothed over 6 cm, with a
   rotation-minimising frame; theta = 0 up, the bottom slot at +-180 deg.
2. wall: points within 20 mm of the median radius of their 2 cm x 4 deg cell.
3. range biases: a wall spot is seen by many beams at many ranges while the robot drives, so
       r_measured = r_wall(cell) + g * (b_ring + f(range)),   g = beam direction . outward radial direction
   (a range error moves the point along the beam; g is how much of that is radial). Solved by alternating cell
   medians and least squares; the overall offset is fixed by median(b_ring) = 0, so the mean radius stays as the
   LiDAR measured it. Fitted on even scans and checked on odd scans (wall thickness = p5-p95 of r minus its
   1 cm x 4 deg cell median), per pipe section (black leg before the bend, grey leg after it).
4. the correction is applied along the beam (x <- x - delta * e, which also removes the axial part of the
   error), the range is gated to 0.25-0.45 m and the corrected wall is written for R3.

output: RECON_BIG/centerline.npz, wall_corrected.npz; results/inpipe_172011/recon/r2_calib.json, r2_*.png
usage: python -I r2_centerline_calib.py
"""
import json
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recon_common import BIG, OUT, B_units, a1, a2, lam, Centerline  # noqa: E402
from pipe_geometry import fit_cylinder  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
t0 = time.time()
z = np.load(os.path.join(BIG, "wall_points.npz"))
X = z["X"].astype(np.float64)
ring, rho, inten, scan = z["ring"].astype(int), z["rho"], z["inten"], z["scan"]
O = z["origin"]
print(f"{len(X):,} points, {len(O)} scans [{time.time() - t0:.0f} s]", flush=True)


def group_median(key, val):
    """median of val per key (lower median), returned per element, plus the count per element"""
    order = np.lexsort((val, key))
    ks = key[order]
    start = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    cnt = np.diff(np.r_[start, len(ks)])
    med = val[order][start + (cnt - 1) // 2]
    gid = np.empty(len(key), np.int64)
    gid[order] = np.repeat(np.arange(len(start)), cnt)
    return med[gid], cnt[gid]


def thickness(res):
    p5, p95 = np.percentile(res, [5, 95])
    return 1000 * 1.4826 * np.median(np.abs(res - np.median(res))), 1000 * (p95 - p5)


# ---- 1. centerline -----------------------------------------------------------------------------------------
B = B_units * lam
u1, u2 = (X - B) @ a1, (X - B) @ a2
on2 = u2 > 0
u_leg = np.where(on2, u2, np.minimum(u1, 0))
Q = np.where(on2[:, None], B + np.outer(u_leg, a2), B + np.outer(u_leg, a1))
dax = np.where(on2[:, None], a2, a1)
q = X - Q
r_leg = np.linalg.norm(q - np.einsum("ij,ij->i", q, dax)[:, None] * dax, axis=1)
near = np.abs(r_leg - np.median(r_leg)) < 0.03
rng = np.random.default_rng(0)
stations, axis_pts = [], []
order = np.argsort(u_leg)
us = u_leg[order]
for uc in np.arange(np.percentile(u_leg, 0.5) + 0.03, np.percentile(u_leg, 99.5) - 0.03, 0.01):
    a, b = np.searchsorted(us, [uc - 0.03, uc + 0.03])
    idx = order[a:b]
    idx = idx[near[idx]]
    if len(idx) < 2000:
        continue
    idx = rng.choice(idx, min(len(idx), 8000), replace=False)
    try:
        f = fit_cylinder(X[idx], d0=a2 if uc > 0 else a1, thresh=0.02, max_points=8000)
    except (ValueError, np.linalg.LinAlgError):
        continue
    d = f["dir"] if f["dir"] @ (a2 if uc > 0 else a1) > 0 else -f["dir"]
    c_st = B + uc * (a2 if uc > 0 else a1)                 # put the axis point in the station's plane
    p = f["point"] + ((c_st - f["point"]) @ d) * d
    stations.append(uc)
    axis_pts.append(p)
stations, axis_pts = np.array(stations), np.array(axis_pts)
print(f"centerline: {len(stations)} local cylinder fits, u {stations[0]:+.2f} .. {stations[-1]:+.2f} m [{time.time() - t0:.0f} s]", flush=True)
cam_up = json.load(open(os.path.join(OUT, "cam_up.json")))["up"] if os.path.exists(os.path.join(OUT, "cam_up.json")) else None
if cam_up is None:
    from recon_common import FusedPoses  # noqa: E402
    up = -np.mean(FusedPoses().cam["R"][:, 1], axis=0)
    json.dump({"up": up.tolist()}, open(os.path.join(OUT, "cam_up.json"), "w"))
else:
    up = np.array(cam_up)
uu, Cc, T, N, Bn = Centerline.build(stations, axis_pts, smooth_pts=0.06, up=up)
np.savez(os.path.join(BIG, "centerline.npz"), u=uu, C=Cc, T=T, N=N, Bn=Bn, th0=0.0)
cl = Centerline(os.path.join(BIG, "centerline.npz"))
u, th, r = cl.coords(X, u_init=u_leg)                     # arc along the legs ~ arc along the centerline
# slot: the angle bin with the fewest wall points, then theta shifted so that it sits at 180
hist = np.bincount(((th + 180) // 2).astype(int) % 180, weights=near, minlength=180)
k = int(np.argmin(np.convolve(np.r_[hist, hist[:5]], np.ones(5), "valid")[:180])) + 2
slot = (k * 2 - 180 + 1) % 360
th0 = float(slot - 180)
np.savez(os.path.join(BIG, "centerline.npz"), u=uu, C=Cc, T=T, N=N, Bn=Bn, th0=th0)
cl = Centerline(os.path.join(BIG, "centerline.npz"))
th = (th - th0 + 540) % 360 - 180
bend_u = float(uu[np.argmin(np.linalg.norm(Cc - B, axis=1))])
print(f"slot centre at {slot:.0f} deg of the transported frame -> theta shifted by {th0:+.0f}; bend at u = {bend_u:+.3f} m "
      f"[{time.time() - t0:.0f} s]", flush=True)

# ---- 2. wall ------------------------------------------------------------------------------------------------
key2 = ((u - uu[0]) / 0.02).astype(np.int64) * 100 + ((th + 180) // 4).astype(np.int64)
med2, _ = group_median(key2, r)
wall = np.abs(r - med2) < 0.020
leg2 = u > bend_u
e = X - O[scan]
e /= np.linalg.norm(e, axis=1, keepdims=True)
i_c = np.clip(np.round((u - cl.u[0]) / cl.ds).astype(int), 0, len(cl.u) - 1)
nr = X - cl.C[i_c]
nr -= np.einsum("ij,ij->i", nr, cl.T[i_c])[:, None] * cl.T[i_c]
nr /= np.linalg.norm(nr, axis=1, keepdims=True)
g = np.einsum("ij,ij->i", e, nr)
print(f"wall: {wall.sum():,} of {len(X):,} points; radial share of the beam g: median {np.median(g[wall]):.2f} "
      f"(p10 {np.percentile(g[wall], 10):.2f}, p90 {np.percentile(g[wall], 90):.2f}) [{time.time() - t0:.0f} s]", flush=True)

# ---- 3. self-calibration ------------------------------------------------------------------------------------
RB = np.arange(0.22, 0.505, 0.01)
ib = np.clip(np.digitize(rho, RB) - 1, 0, len(RB) - 2)
key1 = ((u - uu[0]) / 0.01).astype(np.int64) * 100 + ((th + 180) // 4).astype(np.int64)


def calibrate(sel, iters=6):
    b, f = np.zeros(128), np.zeros(len(RB) - 1)
    for _ in range(iters):
        delta = b[ring[sel]] + f[ib[sel]]
        med, _ = group_median(key1[sel], r[sel] - g[sel] * delta)
        res = r[sel] - med                                   # = g * delta + noise
        sig = 1.4826 * np.median(np.abs(res - g[sel] * delta))
        ok = np.abs(res - g[sel] * delta) < 3 * sig
        gs, rs, rg, ibs = g[sel][ok], res[ok], ring[sel][ok], ib[sel][ok]
        b = np.bincount(rg, gs * (rs - gs * f[ibs]), 128) / np.maximum(np.bincount(rg, gs * gs, 128), 1e-9)
        n_r = np.bincount(rg, minlength=128)
        b[n_r < 200] = 0
        b -= np.median(b[n_r >= 200])
        f = np.bincount(ibs, gs * (rs - gs * b[rg]), len(f)) / np.maximum(np.bincount(ibs, gs * gs, len(f)), 1e-9)
        f[np.bincount(ibs, minlength=len(f)) < 500] = 0
    return b, f, n_r


def check(sel_eval, b, f, gate):
    """wall thickness of the odd scans (cell medians from the odd scans only)"""
    s_ = sel_eval & gate
    rc = r[s_] - g[s_] * (b[ring[s_]] + f[ib[s_]])
    med, cnt = group_median(key1[s_], rc)
    return thickness((rc - med)[cnt >= 5])


even, odd = scan % 2 == 0, scan % 2 == 1
g_all = (rho > 0.20) & (rho < 0.50)
g_band = (rho > 0.25) & (rho < 0.45)
zero_b, zero_f = np.zeros(128), np.zeros(len(RB) - 1)
report = {}
fits = {}
for name, legsel in (("leg1", ~leg2), ("leg2", leg2)):
    base = wall & legsel
    b_l, f_l, n_l = calibrate(base & even)
    fits[name] = (b_l, f_l, n_l)
    rows = {
        "raw, 0.22-0.50 m": check(base & odd, zero_b, zero_f, g_all),
        "raw, 0.25-0.45 m": check(base & odd, zero_b, zero_f, g_band),
        "calibrated (own section), 0.22-0.50 m": check(base & odd, b_l, f_l, g_all),
        "calibrated (own section), 0.25-0.45 m": check(base & odd, b_l, f_l, g_band),
    }
    report[name] = {k: {"robust_sigma_mm": round(v[0], 2), "band_p5_p95_mm": round(v[1], 1)} for k, v in rows.items()}
    print(f"{name}: ring bias (along the beam) std {1000 * np.std(b_l[n_l >= 200]):.1f} mm over {(n_l >= 200).sum()} rings")
    for kk, v in rows.items():
        print(f"   {kk:42s} sigma {v[0]:5.2f} mm   wall thickness p5-p95 {v[1]:5.1f} mm", flush=True)
# does the other section's calibration transfer? (black vs grey surface)
for name, other, legsel in (("leg1", "leg2", ~leg2), ("leg2", "leg1", leg2)):
    v = check(wall & legsel & odd, fits[other][0], fits[other][1], g_band)
    report[name][f"calibrated with {other}'s biases, 0.25-0.45 m"] = {"robust_sigma_mm": round(v[0], 2), "band_p5_p95_mm": round(v[1], 1)}
    print(f"{name} with {other}'s biases, 0.25-0.45 m: sigma {v[0]:.2f}, thickness {v[1]:.1f} mm")
print(f"[{time.time() - t0:.0f} s]", flush=True)

# final: per section, fitted on all scans
final = {}
delta = np.zeros(len(X))
for name, legsel in (("leg1", ~leg2), ("leg2", leg2)):
    b_l, f_l, n_l = calibrate(wall & legsel)
    final[name] = {"ring_bias_mm": (1000 * b_l).round(2).tolist(), "ring_points": n_l.tolist(),
                   "range_bins_m": RB.round(3).tolist(), "range_bias_mm": (1000 * f_l).round(2).tolist()}
    delta[legsel] = b_l[ring[legsel]] + f_l[ib[legsel]]

# ---- 4. corrected wall -------------------------------------------------------------------------------------
Xc = X - delta[:, None] * e
keep = g_band & (np.abs(r - g * delta - med2) < 0.03)
uc_, thc, rc = cl.coords(Xc[keep], u_init=u[keep])
key2c = ((uc_ - uu[0]) / 0.02).astype(np.int64) * 100 + ((thc + 180) // 4).astype(np.int64)
m2c, _ = group_median(key2c, rc)
w2 = np.abs(rc - m2c) < 0.012
idx = np.flatnonzero(keep)[w2]
np.savez(os.path.join(BIG, "wall_corrected.npz"), X=Xc[idx].astype(np.float32), u=uc_[w2].astype(np.float32),
         th=thc[w2].astype(np.float32), r=rc[w2].astype(np.float32), scan=scan[idx], ring=ring[idx].astype(np.uint8),
         rho=rho[idx], inten=inten[idx])
# the uncorrected wall the same way, for the before/after model in R3 (0.20-0.60 m would be f5_fuse's band;
# here 0.22-0.50 m, all that R1 kept)
keep0 = g_all & wall
np.savez(os.path.join(BIG, "wall_raw.npz"), u=u[keep0].astype(np.float32), th=th[keep0].astype(np.float32),
         r=r[keep0].astype(np.float32))
json.dump({"bend_u_m": bend_u, "slot_theta_shift_deg": th0, "n_points_corrected": int(len(idx)),
           "check_odd_scans": report, "final_biases": final},
          open(os.path.join(OUT, "r2_calib.json"), "w"), indent=1)
print(f"corrected wall: {len(idx):,} points (0.25-0.45 m) -> wall_corrected.npz [{time.time() - t0:.0f} s]")

# ---- figure: biases ---------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
for name, col in (("leg1", "k"), ("leg2", "tab:red")):
    b_l = np.array(final[name]["ring_bias_mm"])
    n_l = np.array(final[name]["ring_points"])
    used = n_l >= 200
    ax[0].plot(np.flatnonzero(used), b_l[used], "o-", ms=3, color=col, label={"leg1": "第一节（黑）", "leg2": "第二节（灰）"}[name])
    ax[1].plot(RB[:-1] + 0.005, final[name]["range_bias_mm"], "o-", ms=3, color=col)
ax[0].set_xlabel("激光线编号 ring"); ax[0].set_ylabel("沿光束的测距偏差 mm"); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[0].set_title("每条激光线的固定偏差（自标定）")
ax[1].set_xlabel("测距 m"); ax[1].set_ylabel("mm"); ax[1].grid(alpha=0.3)
ax[1].set_title("随距离变化的偏差（自标定，叠加在每线偏差上）")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "r2_biases.png"), dpi=100)
print("figure ->", os.path.join(OUT, "r2_biases.png"))
