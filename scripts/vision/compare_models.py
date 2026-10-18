"""Do two reconstructions of the same pipe agree in length scale? No ground truth is used.

Usage: compare_models.py CONVERTER_EXE WORK_A WORK_B OUT_PNG

For each work folder the dense cloud (mvs/scene_dense.ply) is put into pipe coordinates (axis and radius
from the sparse model, one scale for the whole model from the 300 mm diameter) and the brightness of the
wall points is averaged around the circumference, giving one brightness profile along the pipe. Joints
and stains are the same physical marks in both videos, so the stretch factor and shift that best align
profile B with profile A show how much the two models differ along the pipe:
  position_A = factor * position_B + shift      (factor 1.000 = same length scale)
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as ev  # noqa: E402

converter, work_a, work_b, out_png = sys.argv[1:5]
BIN = 0.01          # metres
X = np.arange(0.5, 4.6, BIN)


def wall_profile(work):
    """Mean grey value of the wall points per 1 cm along the pipe, metres from the first camera."""
    views, _, _, poses, spts, nobs, _ = ev.load_sfm(converter, os.path.join(work, "sfm", "recon", "sfm_data.bin"))
    frames = {r["file"]: int(r["source_frame"]) for r in csv.DictReader(open(os.path.join(work, "frames.csv"), encoding="utf-8"))}
    cams = np.array([p for _, p in sorted((frames[name], poses[pid]) for name, pid in views.values() if pid in poses)])
    mean_c = cams.mean(axis=0)
    _, _, vt = np.linalg.svd(cams - mean_c)
    d0 = vt[0] * np.sign((cams[-1] - cams[0]) @ vt[0])
    use = spts[nobs >= 3]
    d, axis_pt, _, _, u, v = ev.fit_pipe(use, d0)
    d *= np.sign(d @ d0)
    s_cam, s_sp = (cams - axis_pt) @ d, (use - axis_pt) @ d
    rows = []
    edges = np.linspace(s_cam.min(), np.percentile(s_sp, 99), 13)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (s_sp >= a) & (s_sp < b)
        if m.sum() < 80:
            continue
        c, r_k, k_in = ev.fit_circle(np.c_[(use[m] - axis_pt) @ u, (use[m] - axis_pt) @ v])
        if k_in.mean() >= 0.4:
            rows.append((0.5 * (a + b), r_k, c[0], c[1]))
    sp = np.array(rows)
    vtx = ev.read_ply_vertices(os.path.join(work, "mvs", "scene_dense.ply"))
    pts = np.c_[vtx["x"], vtx["y"], vtx["z"]].astype(float)
    grey = (vtx["red"].astype(float) + vtx["green"] + vtx["blue"]) / 3
    s = (pts - axis_pt) @ d
    pu, pv = (pts - axis_pt) @ u, (pts - axis_pt) @ v
    rr = np.hypot(pu - np.interp(s, sp[:, 0], sp[:, 2]), pv - np.interp(s, sp[:, 0], sp[:, 3])) / np.interp(s, sp[:, 0], sp[:, 1])
    half = ev.PIPE_DIAMETER_M / 2
    wall = (rr > 0.85) & (rr < 1.15)
    # position along the pipe in metres from the first camera, with two ways of setting the scale:
    x_one = (s - s_cam[0]) * half / np.median(sp[:, 1])                      # one scale for the whole model
    grid = np.linspace(min(s.min(), s_cam.min()), max(s.max(), s_cam.max()), 4000)
    metric = np.r_[0, np.cumsum(np.diff(grid) * half / np.interp(0.5 * (grid[1:] + grid[:-1]), sp[:, 0], sp[:, 1]))]
    x_loc = np.interp(s, grid, metric) - np.interp(s_cam[0], grid, metric)   # scale re-set locally from the diameter
    profs = []
    for x in (x_one, x_loc):
        k = np.floor((x[wall] - X[0]) / BIN).astype(int)
        ok = (k >= 0) & (k < len(X))
        total = np.bincount(k[ok], weights=grey[wall][ok], minlength=len(X))
        count = np.bincount(k[ok], minlength=len(X))
        prof = np.where(count > 50, total / np.maximum(count, 1), np.nan)
        profs.append(np.interp(X, X[~np.isnan(prof)], prof[~np.isnan(prof)]))
    return profs[0], profs[1], float((s_cam[-1] - s_cam[0]) * half / np.median(sp[:, 1]))


def detrend(p, width=0.30):
    """Remove the slow brightness change along the pipe, keep marks narrower than `width` metres."""
    n = int(width / BIN) | 1
    smooth = np.convolve(np.pad(p, n // 2, mode="edge"), np.ones(n) / n, mode="valid")
    q = p - smooth
    return (q - q.mean()) / q.std()


core = (X > 0.8) & (X < 4.3)          # part of A that must be covered by B for every stretch and shift tried


def align(da, db):
    """Stretch factor and shift of profile B that correlate best with profile A."""
    best = (-2.0, 1.0, 0.0)
    for factor in np.arange(0.90, 1.1001, 0.002):
        for shift in np.arange(-0.30, 0.3001, 0.005):
            c = float(np.corrcoef(da[core], np.interp((X[core] - shift) / factor, X, db))[0, 1])
            if c > best[0]:
                best = (c, float(factor), float(shift))
    return best, float(np.corrcoef(da[core], db[core])[0, 1])


pa, pa_loc, len_a = wall_profile(work_a)
pb, pb_loc, len_b = wall_profile(work_b)
da, db = detrend(pa), detrend(pb)
best, c0 = align(da, db)
best_loc, c0_loc = align(detrend(pa_loc), detrend(pb_loc))
print(f"camera path length: A {len_a:.3f} m, B {len_b:.3f} m")
print(f"brightness profiles along the pipe, correlation as reconstructed (factor 1, shift 0): {c0:.2f}")
print(f"best alignment: position_A = {best[1]:.3f} * position_B {best[2]:+.3f} m   (correlation {best[0]:.2f})")
print(f"with the scale re-set locally from the diameter in both models: correlation as reconstructed {c0_loc:.2f}; "
      f"best alignment position_A = {best_loc[1]:.3f} * position_B {best_loc[2]:+.3f} m   (correlation {best_loc[0]:.2f})")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
ax[0].plot(X, da, "b-", lw=0.9, label=os.path.basename(work_a))
ax[0].plot(X, db, "r-", lw=0.9, label=os.path.basename(work_b))
ax[0].set(title=f"wall brightness along the pipe, as reconstructed (correlation {c0:.2f})", ylabel="detrended brightness")
ax[0].legend()
ax[1].plot(X, da, "b-", lw=0.9, label=os.path.basename(work_a))
ax[1].plot(best[1] * X + best[2], db, "r-", lw=0.9, label=f"{os.path.basename(work_b)}: x * {best[1]:.3f} {best[2]:+.3f} m")
ax[1].set(title=f"after the best stretch and shift (correlation {best[0]:.2f})", xlabel="along the pipe from the first camera [m]",
          ylabel="detrended brightness", xlim=(X[0], X[-1]))
ax[1].legend()
fig.tight_layout()
fig.savefig(out_png, dpi=110)
print("saved", out_png)
