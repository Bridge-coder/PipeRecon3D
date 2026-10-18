"""R3: the pipe wall as a surface: height field r(u, theta) on a 5 mm x 1 deg grid, and its triangle mesh.

Each cell takes the median radius of the refined wall points in it (R2b); cells with < MIN_PTS points take the
value of a grid twice as coarse (10 mm x 2 deg, the top of the pipe is where the LiDAR samples least); cells
more than 6 mm off the median of their 2 cm x 3 deg neighbourhood are dropped (spikes, mostly where the beam
grazes the cut edge of the slot) and the slot gap is widened by 2 deg on each side; isolated empty cells are
filled from their neighbours; wide gaps (the bottom slot, the ends) stay open, so the
mesh shows them as openings. Two deviation maps:
  dev        r minus the mean cross-section of its pipe section (median over u per angle): ellipse and offset
             removed, slow shape changes along the pipe kept
  dev_local  r minus the running median over +-5 cm along u (per angle): only local features (steps, dents,
             holes, deposits) - the map to look for defects in
The same is done with the raw wall (no bias correction, no pose refinement, 0.22-0.50 m) for comparison.

output: RECON_BIG/heightfield.npz, heightfield_raw.npz; results/inpipe_172011/recon/r3_heightfield.png
usage: python -I r3_heightfield.py
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recon_common import BIG, OUT  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
DU, DT, MIN_PTS = 0.005, 1.0, 5
bend = json.load(open(os.path.join(OUT, "r2_calib.json")))["bend_u_m"]
t0 = time.time()


def cell_median(u, th, r, u0, nu, du, dt):
    nt = int(round(360 / dt))
    iu = ((u - u0) / du).astype(np.int64)
    it = ((th + 180) / dt).astype(np.int64) % nt
    ok = (iu >= 0) & (iu < nu)
    key = iu[ok] * nt + it[ok]
    val = r[ok]
    o = np.lexsort((val, key))
    ks, vs = key[o], val[o]
    st = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    cnt = np.diff(np.r_[st, len(ks)])
    R = np.full(nu * nt, np.nan)
    N = np.zeros(nu * nt, np.int64)
    R[ks[st]] = vs[st + (cnt - 1) // 2]
    N[ks[st]] = cnt
    R[N < MIN_PTS] = np.nan
    return R.reshape(nu, nt), N.reshape(nu, nt)


def drop_outliers(R, lim=0.006, ku=4, kt=3, slot_margin=2):
    """cells more than lim from the median of their +-ku x +-kt neighbourhood (spikes, mostly on the slot edges,
    where the beam grazes the cut edge) are emptied, and slot_margin cells on each side of the slot gap too.
    A step around the pipe (a joint) survives: along theta the neighbourhood follows it."""
    P = np.pad(R, ((ku, ku), (kt, kt)), mode="constant", constant_values=np.nan)
    P[ku:-ku, :kt], P[ku:-ku, -kt:] = R[:, -kt:], R[:, :kt]  # theta wraps
    win = np.lib.stride_tricks.sliding_window_view(P, (2 * ku + 1, 2 * kt + 1))
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ref = np.nanmedian(win.reshape(*win.shape[:2], -1), axis=2)
    out = R.copy()
    out[np.abs(R - ref) > lim] = np.nan
    nt = R.shape[1]
    j180 = np.arange(nt)
    for i in range(R.shape[0]):                             # widen the gap around +-180 deg (the slot)
        row = np.isnan(out[i])
        if not row[0] and not row[-1]:
            continue
        a = np.argmax(~row) if row[0] else 0                # first filled cell after -180
        b = nt - np.argmax(~row[::-1]) if row[-1] else nt   # one past the last filled cell before +180
        out[i, j180[:a + slot_margin]] = np.nan
        out[i, j180[max(b - slot_margin, 0):]] = np.nan
    return out


def heightfield(u, th, r, u0, nu):
    R, N = cell_median(u, th, r, u0, nu, DU, DT)
    Rc, _ = cell_median(u, th, r, u0, (nu + 1) // 2, 2 * DU, 2 * DT)
    Rc = np.repeat(np.repeat(Rc, 2, axis=0), 2, axis=1)[:nu]
    hole = np.isnan(R)
    R[hole] = Rc[hole]
    R = drop_outliers(R)
    # fill isolated holes: an empty cell with >= 6 of its 8 neighbours filled takes their median (2 passes)
    for _ in range(2):
        P = np.pad(R, ((1, 1), (0, 0)), constant_values=np.nan)
        nb = np.stack([np.roll(P, (du, dt), axis=(0, 1))[1:-1] for du in (-1, 0, 1) for dt in (-1, 0, 1) if du or dt])
        nfill = np.isfinite(nb).sum(0)
        with np.errstate(all="ignore"):
            fill = np.isnan(R) & (nfill >= 6)
            R[fill] = np.nanmedian(nb[:, fill], axis=0)
    return R, N


def deviation(R, uc):
    dev = np.full_like(R, np.nan)
    for sel, away in ((uc < bend, uc < bend - 0.05), (uc >= bend, uc > bend + 0.05)):
        with np.errstate(all="ignore"):
            prof = np.nanmedian(R[away], axis=0)
        dev[sel] = R[sel] - prof
    return dev


def deviation_local(R, half=0.05):
    k = int(round(half / DU))
    P = np.pad(R, ((k, k), (0, 0)), constant_values=np.nan)
    win = np.lib.stride_tricks.sliding_window_view(P, 2 * k + 1, axis=0)   # nu x nt x (2k+1)
    with np.errstate(all="ignore"):
        ref = np.nanmedian(win, axis=2)
    return R - ref


z = np.load(os.path.join(BIG, "wall_refined.npz"))
u, th, r = z["u"].astype(float), z["th"].astype(float), z["r"].astype(float)
u0 = np.floor(np.percentile(u, 0.02) / DU) * DU
u1 = np.ceil(np.percentile(u, 99.98) / DU) * DU
nu = int(round((u1 - u0) / DU))
uc = u0 + (np.arange(nu) + 0.5) * DU
tc = -180 + (np.arange(int(360 / DT)) + 0.5) * DT
R, N = heightfield(u, th, r, u0, nu)
dev = deviation(R, uc)
dev_local = deviation_local(R)
inten = z["inten"].astype(float)
Ri, _ = heightfield(u, th, inten, u0, nu)                    # LiDAR intensity per cell (median), for reference


def close_gaps(R, max_deg=40.0, keep_out=150.0):
    """for the mesh only: gaps up to max_deg wide away from the slot (|theta| < keep_out) interpolated along theta
    (the top of the first leg, where the LiDAR samples least); returns the filled copy and the mask of filled cells"""
    Rf = R.copy()
    inner = np.abs(tc) < keep_out
    for i in range(R.shape[0]):
        row = R[i]
        ok = np.isfinite(row) & inner
        if ok.sum() < 10:
            continue
        holes = np.flatnonzero(~np.isfinite(row) & inner)
        if not len(holes):
            continue
        runs = np.split(holes, np.flatnonzero(np.diff(holes) > 1) + 1)
        for run in runs:
            if len(run) * DT <= max_deg and run[0] > 0 and run[-1] < len(row) - 1 and ok[run[0] - 1] and ok[run[-1] + 1]:
                Rf[i, run] = np.interp(tc[run], [tc[run[0] - 1], tc[run[-1] + 1]], [row[run[0] - 1], row[run[-1] + 1]])
    return Rf, np.isnan(R) & np.isfinite(Rf)


R_mesh, interp = close_gaps(R)
fill_row = np.isfinite(R_mesh[:, np.abs(tc) < 150]).mean(1)
rows = np.flatnonzero(fill_row >= 0.8)
mesh_rows = (int(rows.min()), int(rows.max()) + 1)          # u range of the mesh: rows that are >= 80 % covered
np.savez(os.path.join(BIG, "heightfield.npz"), u=uc, th=tc, r=R, r_mesh=R_mesh, interp=interp, n=N, dev=dev,
         dev_local=dev_local, inten=Ri, du=DU, dt=DT, mesh_rows=np.array(mesh_rows))
filled = np.isfinite(R)
print(f"mesh: rows {mesh_rows[0]}..{mesh_rows[1]} (u {uc[mesh_rows[0]]:+.3f} .. {uc[mesh_rows[1] - 1]:+.3f} m); "
      f"{interp[mesh_rows[0]:mesh_rows[1]].mean():.1%} of its cells interpolated along theta", flush=True)
print(f"height field: {nu} x {len(tc)} cells ({DU * 1000:.0f} mm x {DT:.0f} deg), u {u0:+.3f} .. {u1:+.3f} m, "
      f"{filled.mean():.1%} filled, median {np.median(N[N > 0]):.0f} points per cell [{time.time() - t0:.0f} s]", flush=True)
# precision of a cell: robust scatter of its points / sqrt(n) (from the R2b residual sigma ~1.2-1.5 mm)
zr = np.load(os.path.join(BIG, "wall_raw.npz"))
Rr, Nr = heightfield(zr["u"].astype(float), zr["th"].astype(float), zr["r"].astype(float), u0, nu)
devr = deviation(Rr, uc)
devr_local = deviation_local(Rr)
Rr_mesh, interp_r = close_gaps(Rr)
np.savez(os.path.join(BIG, "heightfield_raw.npz"), u=uc, th=tc, r=Rr, r_mesh=Rr_mesh, interp=interp_r, n=Nr, dev=devr,
         dev_local=devr_local, du=DU, dt=DT, mesh_rows=np.array(mesh_rows))


def roughness(D):
    """cell-to-cell scatter of the deviation map: median absolute second difference along u and theta"""
    d2u = D[2:, :] - 2 * D[1:-1, :] + D[:-2, :]
    d2t = D[:, 2:] - 2 * D[:, 1:-1] + D[:, :-2]
    return 1000 * np.nanmedian(np.abs(d2u)) / np.sqrt(6), 1000 * np.nanmedian(np.abs(d2t)) / np.sqrt(6)


stats = {"grid": f"{DU * 1000:.0f} mm x {DT:.0f} deg", "u_range_m": [float(u0), float(u1)],
         "filled_share": float(filled.mean()), "filled_share_raw": float(np.isfinite(Rr).mean()),
         "points_per_cell_median": float(np.median(N[N > 0])),
         "cell_noise_mm_refined (along u, along theta)": [round(x, 2) for x in roughness(dev)],
         "cell_noise_mm_raw (along u, along theta)": [round(x, 2) for x in roughness(devr)],
         "deviation_p5_p95_mm_refined": (1000 * np.nanpercentile(dev, [5, 95])).round(1).tolist(),
         "deviation_p5_p95_mm_raw": (1000 * np.nanpercentile(devr, [5, 95])).round(1).tolist(),
         "local_deviation_p5_p95_mm_refined": (1000 * np.nanpercentile(dev_local, [5, 95])).round(1).tolist(),
         "local_deviation_p5_p95_mm_raw": (1000 * np.nanpercentile(devr_local, [5, 95])).round(1).tolist()}
for name, sel in (("leg1", uc < bend - 0.05), ("leg2", uc > bend + 0.05)):
    with np.errstate(all="ignore"):
        prof = np.nanmedian(R[sel], axis=0)
    stats[f"{name}_mean_radius_mm"] = round(1000 * float(np.nanmedian(prof)), 1)
    stats[f"{name}_major_minor_diameter_mm"] = [round(2000 * float(np.nanpercentile(prof, q)), 1) for q in (95, 5)]
json.dump(stats, open(os.path.join(OUT, "r3_heightfield.json"), "w"), indent=1)
print(json.dumps(stats, indent=1, ensure_ascii=False))

fig, ax = plt.subplots(4, 1, figsize=(16, 16), sharex=True)
for a, D, title, lim in ((ax[0], devr, "原始：0.22–0.50 m 全部点，无偏差校正、无位姿细化——减去每节平均截面", 5),
                         (ax[1], dev, "改进后：每线 + 距离偏差自标定，0.25–0.45 m，扫描对地图位姿细化——减去每节平均截面", 5),
                         (ax[2], devr_local, "原始——局部偏差：减去沿管 ±5 cm 滑动中位数", 2),
                         (ax[3], dev_local, "改进后——局部偏差：减去沿管 ±5 cm 滑动中位数", 2)):
    im = a.imshow(1000 * D.T, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                  extent=[u0, u1, -180, 180], interpolation="nearest")
    a.set_ylabel("转角 °（0 = 顶，±180 = 底缝）")
    a.set_title(f"{title}（高度场 5 mm × 1°，±{lim} mm）", fontsize=11)
    a.axvline(bend, color="k", ls="--", lw=0.8)
    plt.colorbar(im, ax=a, label="mm", fraction=0.03, pad=0.01)
ax[1].set_xlabel("沿管位置 u（m，0 = 弯头）")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "r3_heightfield.png"), dpi=90)
print("figure ->", os.path.join(OUT, "r3_heightfield.png"))
