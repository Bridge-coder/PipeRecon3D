"""Evaluate a reconstruction against per-frame ground-truth distances when the pipe diameter is NOT known.

usage: python -I eval_gt_scale.py WORK_DIR OUT_PREFIX [--part NAME] [--colmap SPARSE_TXT_DIR] [--window M]
       WORK_DIR   needs frames.csv and counter.csv (source_frame, counter_m = ground-truth distance)
       --colmap   evaluate a COLMAP model (baselines/run_colmap.py) on the same frames instead of WORK/sfm/recon
       --window   length of the windows for the local-scale check (default 2 m)
       --from-m   only frames whose ground truth is at least this (default 0.25 m: odometers lag while the robot
                  starts, as the Sheffield counters do; evaluate.py uses the same 0.25 m)

A monocular model has no scale. Here one global scale (and offset) is fitted between the camera positions along
the pipe axis and the ground-truth distances, by least squares over all registered frames; what remains is the
localisation error of the model itself (how the robot-localisation literature scores monocular VO such as ORB-SLAM3):
  MAE / RMSE / max of |estimated - true| position, in metres and in % of the distance covered
  local scale: the same fit in consecutive windows of M metres; their spread (%) is the scale drift
  implied bore: 2 x (cylinder radius of the sparse points in model units) x scale
Registered frames, number of continuous runs and the longest run are reported as well (a run that breaks is a
failure of the pipeline even when the pieces are accurate).
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baselines"))
import evaluate as ev  # noqa: E402
import paths  # noqa: E402

args = sys.argv[1:]


def opt(flag, default=None):
    if flag in args:
        i = args.index(flag)
        v = args[i + 1]
        del args[i:i + 2]
        return v
    return default


part, colmap_dir, window = opt("--part"), opt("--colmap"), float(opt("--window", "2.0"))
from_m = float(opt("--from-m", "0.25"))
work, out_prefix = args
frames_csv = os.path.join(work, "frames.csv")
frames = {r["file"]: int(r["source_frame"]) for r in csv.DictReader(open(frames_csv, encoding="utf-8"))}
gt = {int(r["source_frame"]): float(r["counter_m"]) for r in csv.DictReader(open(os.path.join(work, "counter.csv"), encoding="utf-8"))}
if colmap_dir:
    import colmap_eval
    ev.load_sfm = colmap_eval.colmap_loader(colmap_dir, frames_csv)
    sfm_bin, label = None, f"COLMAP {colmap_dir}"
else:
    sfm_bin = os.path.join(work, "sfm", f"recon_{part}" if part else "recon", "sfm_data.bin")
    label = sfm_bin
views, model, intr, poses, pts, nobs, obs = ev.load_sfm(paths.CONVERTER, sfm_bin)
reg_all = sorted((frames[name], poses[pid]) for vid, (name, pid) in views.items() if pid in poses)
reg = [r for r in reg_all if r[0] in gt and gt[r[0]] >= from_m]          # scored frames
src = np.array([r[0] for r in reg])
cams = np.array([r[1] for r in reg])
order = sorted(frames.values())
pos = {f: k for k, f in enumerate(order)}
idx = np.array([pos[f] for f, _ in reg_all])
runs = np.split(np.arange(len(idx)), np.where(np.diff(idx) > 1)[0] + 1)
longest = max(runs, key=len)
out = {"model": label, "frames_total": len(frames), "frames_registered": int(len(reg_all)), "runs": len(runs),
       "longest_run": int(len(longest)), "points": int(len(pts)), "focal": intr.get("focal_length"),
       "k1": (intr.get("disto_k1") or [None])[0], "frames_scored": int(len(src)), "scored_from_m": from_m}
print(f"{label}\nregistered {len(reg_all)} of {len(frames)} frames; {len(runs)} run(s), longest {len(longest)}; "
      f"{len(pts)} points; focal {intr.get('focal_length'):.1f} px, k1 {out['k1']}; scored {len(src)} frames "
      f"(ground truth >= {from_m:g} m)")
if len(src) < 10:
    json.dump(out, open(out_prefix + ".json", "w"), indent=1)
    sys.exit("too few registered frames to evaluate")

# axis: principal direction of the camera path (a straight stretch of sewer); position = projection on it
mu = cams.mean(axis=0)
_, _, vt = np.linalg.svd(cams - mu)
d0 = vt[0] * np.sign((cams[-1] - cams[0]) @ vt[0])
s = (cams - mu) @ d0
g = np.array([gt[f] for f in src])
A = np.c_[s, np.ones_like(s)]
(scale, off), *_ = np.linalg.lstsq(A, g, rcond=None)
err = scale * s + off - g
length = float(g.max() - g.min())
out.update(scale_m_per_unit=float(scale), stretch_m=length, mae_m=float(np.abs(err).mean()),
           rmse_m=float(np.sqrt((err ** 2).mean())), max_abs_m=float(np.abs(err).max()),
           mae_pct=float(np.abs(err).mean() / length * 100), max_pct=float(np.abs(err).max() / length * 100))
# local scale in windows of the true distance
loc = []
for a in np.arange(g.min(), g.max() - window / 2, window):
    m = (g >= a) & (g < a + window)
    if m.sum() >= 10 and np.ptp(g[m]) > window / 2:
        loc.append(float(np.polyfit(s[m], g[m], 1)[0]))
loc = np.array(loc)
out["local_scale_rel"] = (loc / scale).round(4).tolist()
out["local_scale_spread_pct"] = float(np.std(loc / scale) * 100) if len(loc) > 1 else None
# implied bore from the sparse points: a real sewer is not straight over metres, so one cylinder does not fit; instead
# cross-sections every ~0.5 m, each perpendicular to the local camera-path direction (+-0.5 m of cameras), circle fit
good = nobs >= 3
use = pts[good] if good.sum() >= 200 else pts
order_c = np.argsort(s)
cs, cc = s[order_c], cams[order_c]
step = 0.5 / scale
radii = []
for a in np.arange(cs[0] + step, cs[-1] - step, step):
    near = np.abs(cs - a) < step
    if near.sum() < 5:
        continue
    q = cc[near]
    _, _, vv = np.linalg.svd(q - q.mean(0))
    dl = vv[0] * np.sign(vv[0] @ d0)
    c0 = q.mean(0)
    sp = (use - c0) @ dl
    m = np.abs(sp) < step / 2
    if m.sum() < 60:
        continue
    e1 = np.cross(dl, [0.0, 0.0, 1.0] if abs(dl[2]) < 0.9 else [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(dl, e1)
    uv = np.c_[(use[m] - c0) @ e1, (use[m] - c0) @ e2]
    try:
        _, r_k, k_in = ev.fit_circle(uv)
    except Exception:  # noqa: BLE001
        continue
    if k_in.mean() >= 0.4:
        radii.append(r_k)
if radii:
    radii = np.array(radii) * scale * 2000
    out["implied_diameter_mm"] = float(np.median(radii))
    out["implied_diameter_p10_p90_mm"] = [float(np.percentile(radii, 10)), float(np.percentile(radii, 90))]
    out["cross_sections"] = int(len(radii))
else:
    out["implied_diameter_mm"] = None
print(f"ground truth {g.min():.2f} .. {g.max():.2f} m ({length:.2f} m); one global scale {scale:.4f} m/unit")
print(f"localisation error: MAE {out['mae_m'] * 100:.1f} cm ({out['mae_pct']:.2f} % of the stretch), RMSE "
      f"{out['rmse_m'] * 100:.1f} cm, max {out['max_abs_m'] * 100:.1f} cm ({out['max_pct']:.2f} %)")
if out["local_scale_spread_pct"] is not None:
    print(f"local scale in {window:g} m windows relative to the global one: {' '.join(f'{x:.3f}' for x in loc / scale)} "
          f"(spread {out['local_scale_spread_pct']:.1f} %)")
if out.get("implied_diameter_mm"):
    print(f"implied pipe diameter {out['implied_diameter_mm']:.0f} mm (median of {out['cross_sections']} cross-sections, "
          f"p10-p90 {out['implied_diameter_p10_p90_mm'][0]:.0f}-{out['implied_diameter_p10_p90_mm'][1]:.0f} mm)")
json.dump(out, open(out_prefix + ".json", "w"), indent=1)
with open(out_prefix + "_track.csv", "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f)
    wr.writerow(["source_frame", "gt_m", "est_m", "error_cm"])
    wr.writerows((int(a), f"{b:.3f}", f"{c:.3f}", f"{(c - b) * 100:.1f}") for a, b, c in zip(src, g, scale * s + off))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(g, scale * s + off, "b.", ms=2, label="重建（一个全局尺度）")
ax[0].plot([g.min(), g.max()], [g.min(), g.max()], "k--", lw=0.8, label="理想")
ax[0].set(xlabel="真值距离 [m]", ylabel="重建位置 [m]", title="沿管位置：重建 vs 真值")
ax[0].legend()
ax[1].plot(g, err * 100, "r.", ms=2)
ax[1].axhline(0, color="k", lw=0.8)
ax[1].set(xlabel="真值距离 [m]", ylabel="误差 [cm]", title=f"定位误差：MAE {out['mae_pct']:.2f} % / 最大 {out['max_pct']:.2f} %")
fig.tight_layout()
fig.savefig(out_prefix + ".png", dpi=110)
print("saved", out_prefix + ".json / .png / _track.csv")
