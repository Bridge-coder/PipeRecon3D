"""R2b: scan-to-map refinement of the LiDAR poses in the 4 directions a pipe constrains well.

Against the median wall map (1 cm x 4 deg cells), each scan's radial residual is fitted with
    res = c + a cos(theta) + b sin(theta) + du (a' cos(theta) + b' sin(theta)),   du = u - (median u of the scan)
i.e. a sideways shift (a, b) and a tilt (a', b') of the scan relative to the map; c (a radius offset) is fitted
but not applied. Along the axis (encoders) and about it (roll) nothing is changed: there a straight pipe gives
the LiDAR no information. A constant tilt of every scan (extrinsic rotation error) shows up here as well,
because a wall spot is seen from 0.23-0.44 m ahead: an angle error moves it by angle x distance.
The correction of scan k is a small rigid motion about its LiDAR origin O_k:  X <- X + d_k + w_k x (X - O_k),
smoothed over +-3 scans, repeated 3 times.

output: RECON_BIG/wall_refined.npz, scan_corrections.npz (t, O, d, w per scan, to correct the
camera poses in R4); results/inpipe_172011/recon/r2b_refine.json, r2b_refine.png
usage: python -I r2b_scan_refine.py
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
from recon_common import BIG, OUT, Centerline  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
t0 = time.time()
cl = Centerline(os.path.join(BIG, "centerline.npz"))
zc = np.load(os.path.join(BIG, "wall_corrected.npz"))
zp = np.load(os.path.join(BIG, "wall_points.npz"))
O, scan_t = zp["origin"], zp["scan_t"]
X = zc["X"].astype(np.float64)
u, th, r = zc["u"].astype(np.float64), zc["th"].astype(np.float64), zc["r"].astype(np.float64)
scan = zc["scan"]
bend = json.load(open(os.path.join(OUT, "r2_calib.json")))["bend_u_m"]
ns = len(O)
order = np.argsort(scan, kind="stable")
starts = np.searchsorted(scan[order], np.arange(ns + 1))


def group_median(key, val):
    o = np.lexsort((val, key))
    ks = key[o]
    st = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    cnt = np.diff(np.r_[st, len(ks)])
    med = val[o][st + (cnt - 1) // 2]
    gid = np.empty(len(key), np.int64)
    gid[o] = np.repeat(np.arange(len(st)), cnt)
    return med[gid], cnt[gid]


def residuals():
    key = ((u - cl.u[0]) / 0.01).astype(np.int64) * 100 + ((th + 180) // 4).astype(np.int64)
    med, cnt = group_median(key, r)
    return r - med, cnt


def thickness(res, m):
    p5, p95 = np.percentile(res[m], [5, 95])
    return round(1000 * 1.4826 * np.median(np.abs(res[m] - np.median(res[m]))), 2), round(1000 * (p95 - p5), 1)


def smooth(a, k=7):
    ker = np.ones(k) / k
    return np.column_stack([np.convolve(np.pad(a[:, j], k // 2, mode="edge"), ker, mode="valid") for j in range(a.shape[1])])


us = np.array([np.median(u[order[starts[k]:starts[k + 1]]]) if starts[k + 1] > starts[k] else np.nan for k in range(ns)])
leg_of_scan = np.where(us > bend, 2, 1)
report = {"before": {}}
res, cnt = residuals()
for name, L in (("leg1", 1), ("leg2", 2)):
    m = (leg_of_scan[scan] == L) & (cnt >= 5)
    report["before"][name] = thickness(res, m)
print(f"before: {report['before']}  (robust sigma mm, wall thickness p5-p95 mm) [{time.time() - t0:.0f} s]", flush=True)
D_tot, W_tot = np.zeros((ns, 3)), np.zeros((ns, 3))
fit_hist = []
for it in range(3):
    t_rad = np.radians(th)
    D, Wv = np.zeros((ns, 3)), np.zeros((ns, 3))
    prm = np.full((ns, 5), np.nan)
    for k in range(ns):
        idx = order[starts[k]:starts[k + 1]]
        if len(idx) < 500:
            continue
        uc = np.median(u[idx])
        du = u[idx] - uc
        c_, s_ = np.cos(t_rad[idx]), np.sin(t_rad[idx])
        A = np.column_stack([np.ones(len(idx)), c_, s_, du * c_, du * s_])
        ok = np.abs(res[idx]) < 0.008
        if ok.sum() < 300:
            continue
        x, *_ = np.linalg.lstsq(A[ok], res[idx][ok], rcond=None)
        prm[k] = x
        # directions of theta = 0 and 90 deg at uc, and the tangent there
        i = int(np.clip(round((uc - cl.u[0]) / cl.ds), 0, len(cl.u) - 1))
        tt = np.radians(cl.th0)
        e0 = np.cos(tt) * cl.N[i] + np.sin(tt) * cl.Bn[i]
        e90 = -np.sin(tt) * cl.N[i] + np.cos(tt) * cl.Bn[i]
        T = cl.T[i]
        Pc = cl.C[i]
        d0 = -(x[1] * e0 + x[2] * e90)                     # undo the shift at the scan's median u
        wv = np.cross(x[3] * e0 + x[4] * e90, T)            # undo the tilt: w x T = -(a' e0 + b' e90)
        D[k] = d0 + np.cross(wv, O[k] - Pc)                 # same motion, expressed about the LiDAR origin
        Wv[k] = wv
    good = np.isfinite(prm[:, 0])
    for arr in (D, Wv):
        arr[~good] = np.nan
        for j in range(3):
            arr[~good, j] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), arr[good, j])
    D, Wv = smooth(D), smooth(Wv)
    D_tot += D
    W_tot += Wv
    X = X + D[scan] + np.cross(Wv[scan], X - O[scan])
    u, th, r = cl.coords(X, u_init=u)
    res, cnt = residuals()
    fit_hist.append(prm)
    rep = {}
    for name, L in (("leg1", 1), ("leg2", 2)):
        m = (leg_of_scan[scan] == L) & (cnt >= 5)
        rep[name] = thickness(res, m)
    report[f"iteration {it + 1}"] = rep
    print(f"iteration {it + 1}: {rep}; this step: shift median {1000 * np.nanmedian(np.hypot(prm[:, 1], prm[:, 2])):.2f} mm, "
          f"tilt median {np.degrees(np.nanmedian(np.hypot(prm[:, 3], prm[:, 4]))):.2f} deg [{time.time() - t0:.0f} s]", flush=True)

wall = np.abs(res) < 0.010
np.savez(os.path.join(BIG, "wall_refined.npz"), X=X[wall].astype(np.float32), u=u[wall].astype(np.float32),
         th=th[wall].astype(np.float32), r=r[wall].astype(np.float32), scan=scan[wall], ring=zc["ring"][wall],
         rho=zc["rho"][wall], inten=zc["inten"][wall])
np.savez(os.path.join(BIG, "scan_corrections.npz"), t=scan_t, O=O, d=D_tot, w=W_tot, u=us)
ang = np.degrees(np.linalg.norm(W_tot, axis=1))
sh = 1000 * np.linalg.norm(D_tot, axis=1)
report["total correction"] = {"rotation_deg_median": round(float(np.median(ang)), 2), "rotation_deg_p90": round(float(np.percentile(ang, 90)), 2),
                              "shift_at_lidar_mm_median": round(float(np.median(sh)), 2), "shift_at_lidar_mm_p90": round(float(np.percentile(sh, 90)), 2),
                              "rotation_mean_vector_deg": np.degrees(W_tot.mean(0)).round(3).tolist()}
json.dump(report, open(os.path.join(OUT, "r2b_refine.json"), "w"), indent=1)
print(f"total correction: rotation median {np.median(ang):.2f} deg (p90 {np.percentile(ang, 90):.2f}), shift at the LiDAR "
      f"median {np.median(sh):.2f} mm; {wall.sum():,} wall points -> wall_refined.npz [{time.time() - t0:.0f} s]")

fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
for j, lab in enumerate("xyz"):
    ax[0].plot(us, np.degrees(W_tot[:, j]), label=f"绕世界 {lab} 轴")
    ax[1].plot(us, 1000 * D_tot[:, j], label=f"世界 {lab}")
ax[0].set_ylabel("旋转修正 °"); ax[0].legend(ncol=3); ax[0].grid(alpha=0.3)
ax[0].set_title("每帧雷达位姿修正（扫描对地图，沿管与滚转不动）")
ax[1].set_ylabel("雷达原点平移修正 mm"); ax[1].set_xlabel("该帧管壁点的中位沿管位置 u（m，0 = 弯头）"); ax[1].grid(alpha=0.3)
for a in ax:
    a.axvline(bend, color="k", ls="--", lw=0.8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "r2b_refine.png"), dpi=100)
