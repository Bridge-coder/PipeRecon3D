"""F3/F4: put the LiDAR scans into the vision model through the camera poses and solve for the translation
t_cl, the vision scale and a LiDAR range bias.

  x_lidar' = x_lidar * (1 + b / |x_lidar|)                 range bias b (m) along every beam
  x_cam    = R_cl x_lidar' + t_cl                           metres, R_cl from F2 (optionally refined here)
  X_world  = C(t) + R(t)^T x_cam / lam                      vision model units; lam = metres per unit
Residual = point-to-plane distance to the nearest OpenMVS dense point (its normal), in metres; ICP with
fixed correspondences per iteration, Gauss-Newton with numeric Jacobian, Huber weights (5 mm).

Variants (F4 - which explanation of the vision/LiDAR scale conflict fits the data):
  A  lam fixed to the encoder scale (vision path length = encoder travel), b = 0   -> t_cl only
  B  lam free, b = 0
  C  lam fixed to the encoder scale, b free
  D  lam and b free
plus every variant with a small rotation refinement of R_cl. A LiDAR range bias shows as radial residuals
that change with range (b * R / range), a pure scale problem as radial residuals independent of range.

Options (added after F4 showed that LiDAR and vision only agree in the second, grey pipe section and that
LiDAR points closer than ~0.2 m are unreliable): --leg 1|2 keeps scans taken with the camera before/after
the bend, --range LO HI the LiDAR range band, --scale tag|encoder the fixed scale of variants A/C,
--tag NAME the output suffix.

usage: python -I f3_register.py EXTRACT_DIR VISION_WORK RESULTS_DIR [--every N] [--per-scan N] [--leg 1|2]
       [--range LO HI] [--scale tag|encoder] [--tag NAME]
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
sys.path.insert(0, os.path.dirname(HERE))
from timebase import load_offsets, stream_times  # noqa: E402
from vision_model import load_cameras, pose_at, _expmap  # noqa: E402
from ply_io import read_ply_vertices  # noqa: E402

args = sys.argv[1:]
EVERY, PER_SCAN = 8, 1500
LEG, RLO, RHI, SCALE, TAG = None, 0.12, 0.9, "encoder", ""
for flag, n in (("--leg", 1), ("--range", 2), ("--scale", 1), ("--tag", 1)):
    if flag in args:
        i = args.index(flag)
        vals = args[i + 1:i + 1 + n]
        del args[i:i + 1 + n]
        if flag == "--leg":
            LEG = int(vals[0])
        elif flag == "--range":
            RLO, RHI = float(vals[0]), float(vals[1])
        elif flag == "--scale":
            SCALE = vals[0]
        else:
            TAG = "_" + vals[0]
for flag in ("--every", "--per-scan"):
    if flag in args:
        i = args.index(flag)
        if flag == "--every":
            EVERY = int(args[i + 1])
        else:
            PER_SCAN = int(args[i + 1])
        del args[i:i + 2]
EX, VW, RES = args
OUT = os.path.join(RES, "fusion")
off = load_offsets(os.path.join(OUT, "time_offsets.json"))
R_cl0 = np.array(json.load(open(os.path.join(OUT, "extrinsic_rotation.json")))["R_cl"])
cam = load_cameras(EX, VW, off)
lam_enc = 1.730 / cam["s"][-1]
if SCALE == "tag":
    lam_enc = json.load(open(os.path.join(OUT, "f4_tag_scale.json")))["mm_per_unit_tag"] / 1000
split_t = cam["t"][json.load(open(os.path.join(OUT, "vision_legs.json")))["split_camera"]]
print(f"{len(cam['t'])} cameras; fixed scale ({SCALE}) {lam_enc * 1000:.3f} mm per unit; leg {LEG or 'both'}, ranges {RLO}-{RHI} m")

# --- LiDAR wall points of the selected scans
tS, rS = stream_times(EX, off, "lidar")
fits = {int(r["idx"]): r for r in csv.DictReader(open(os.path.join(EX, "lidar_fit_short.csv"))) if r.get("diameter_mm")}
D_med = np.median([float(r["diameter_mm"]) for r in fits.values()])
UP = np.array([0.0, -1, 0])
rng = np.random.default_rng(0)
P_l, scan_of, T_of = [], [], []
scans = []
for k, (t, row) in enumerate(zip(tS, rS)):
    i = int(row["idx"])
    f = fits.get(i)
    if k % (EVERY if LEG is None else max(1, EVERY // 2)) or f is None or t < cam["t"][0] or t > cam["t"][-1]:
        continue
    if LEG == 1 and t > split_t - 10e9 or LEG == 2 and t < split_t + 3e9:
        continue
    if abs(float(f["diameter_mm"]) - D_med) > 12 or float(f["inlier_share"]) < 0.5:
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
    s = Q @ d
    r = np.linalg.norm(Q - np.outer(s, d), axis=1)
    m = (rho > RLO) & (rho < RHI) & (s > 0.05) & (np.abs(r - float(f["diameter_mm"]) / 2000) < 0.025)
    idx = np.flatnonzero(m)
    if len(idx) < 200:
        continue
    idx = rng.choice(idx, min(PER_SCAN, len(idx)), replace=False)
    Rw, Cw = pose_at(cam, t)
    P_l.append(P[idx])
    scan_of.append(np.full(len(idx), len(scans)))
    scans.append((t, i, Rw, Cw))
P_l = np.vstack(P_l)
scan_of = np.concatenate(scan_of)
RHO = np.linalg.norm(P_l, axis=1)
Rs = np.array([s[2] for s in scans])
Cs = np.array([s[3] for s in scans])
RsT_pt = np.transpose(Rs, (0, 2, 1))[scan_of]          # camera->world per point
C_pt = Cs[scan_of]
print(f"{len(scans)} scans, {len(P_l):,} LiDAR wall points")

# --- dense cloud and its normals
V = read_ply_vertices(os.path.join(VW, "mvs", "scene_dense.ply"))
X = np.column_stack([V["x"], V["y"], V["z"]]).astype(np.float32)
N = np.column_stack([V["nx"], V["ny"], V["nz"]]).astype(float)
N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
# orient every normal away from the camera path (outward from the pipe axis), so that a residual > 0 means
# the LiDAR point lies outside the vision wall
path = cam["Cs"][::4]
for a in range(0, len(X), 200000):
    b = min(len(X), a + 200000)
    Xa = X[a:b].astype(float)
    j = np.argmin(((Xa[:, None, :] - path[None, :, :]) ** 2).sum(-1), axis=1)
    flip = np.sum(N[a:b] * (Xa - path[j]), axis=1) < 0
    N[a:b][flip] *= -1
t0 = time.time()
index = cv2.flann_Index(X, dict(algorithm=1, trees=4))
print(f"dense cloud {len(X):,} points, FLANN index {time.time() - t0:.1f} s")


def transform(p, Rcl):
    """p = [tx, ty, tz, log lam, b, wx, wy, wz]"""
    lam = np.exp(p[3])
    Pl = P_l * (1 + p[4] / RHO)[:, None]
    R = _expmap(p[5:8]) @ Rcl
    xc = Pl @ R.T + p[:3]
    return C_pt + np.einsum("nij,nj->ni", RsT_pt, xc) / lam, lam


def residuals(p, Rcl, Y, NY):
    Xw, lam = transform(p, Rcl)
    return np.sum((Xw - Y) * NY, axis=1) * lam          # metres


def nn(Xw):
    ids, dist = index.knnSearch(Xw.astype(np.float32), 1, params={"checks": 64})
    ids = ids.ravel()
    return X[ids].astype(float), N[ids], np.sqrt(dist.ravel())


def solve(p0, free, Rcl, iters=25, huber=0.005):
    p = np.array(p0, float)
    for it in range(iters):
        Xw, lam = transform(p, Rcl)
        Y, NY, dist = nn(Xw)
        keep = dist * lam < 0.03                          # correspondences closer than 3 cm
        r = residuals(p, Rcl, Y, NY)[keep]
        w = np.where(np.abs(r) < huber, 1.0, huber / np.maximum(np.abs(r), 1e-12))
        J = np.zeros((keep.sum(), len(free)))
        for c, j in enumerate(free):
            h = 1e-6 if j != 3 else 1e-5
            dp = p.copy()
            dp[j] += h
            J[:, c] = (residuals(dp, Rcl, Y, NY)[keep] - r) / h
        Jw = J * w[:, None]
        step, *_ = np.linalg.lstsq(Jw.T @ J, -(Jw.T @ r), rcond=None)
        p[free] += step
        if np.abs(step).max() < 1e-7:
            break
    Xw, lam = transform(p, Rcl)
    Y, NY, dist = nn(Xw)
    keep = dist * lam < 0.03
    r = residuals(p, Rcl, Y, NY)
    return p, r, keep


# --- coarse start: LiDAR offset along the camera axes, encoder scale, no bias
best = None
for tz in np.arange(-0.30, 0.41, 0.05):
    for ty in np.arange(-0.10, 0.11, 0.05):
        p = np.array([0.0, ty, tz, np.log(lam_enc), 0, 0, 0, 0])
        Xw, lam = transform(p, R_cl0)
        _, _, dist = nn(Xw[::20])
        score = np.median(dist * lam)
        if best is None or score < best[0]:
            best = (score, p)
print(f"coarse start: t = {np.round(best[1][:3], 3)} m, median NN distance {best[0] * 1000:.1f} mm")
p_start = best[1]

variants = {
    "A  t only (scale = encoder, no range bias)": [0, 1, 2],
    "B  t + scale": [0, 1, 2, 3],
    "C  t + range bias (scale = encoder)": [0, 1, 2, 4],
    "D  t + scale + range bias": [0, 1, 2, 3, 4],
    "A+rot": [0, 1, 2, 5, 6, 7],
    "B+rot": [0, 1, 2, 3, 5, 6, 7],
    "C+rot": [0, 1, 2, 4, 5, 6, 7],
    "D+rot": [0, 1, 2, 3, 4, 5, 6, 7],
}
results = {}
for name, free in variants.items():
    p, r, keep = solve(p_start, free, R_cl0)
    rk = r[keep]
    # radial-residual trend with range: slope of r against R/range (a range bias b gives r ~ -b * R/range)
    x = 0.1 / RHO[keep]
    A = np.column_stack([x, np.ones_like(x)])
    (slope, icpt), *_ = np.linalg.lstsq(A, rk, rcond=None)
    results[name] = {"t_cl_m": p[:3].tolist(), "mm_per_unit": float(np.exp(p[3]) * 1000),
                     "scale_vs_encoder": float(np.exp(p[3]) / lam_enc), "range_bias_mm": float(p[4] * 1000),
                     "rot_refine_deg": float(np.degrees(np.linalg.norm(p[5:8]))),
                     "rms_mm": float(np.sqrt(np.mean(rk ** 2)) * 1000), "median_abs_mm": float(np.median(np.abs(rk)) * 1000),
                     "inliers": float(keep.mean()), "trend_slope_mm": float(slope * 1000), "trend_intercept_mm": float(icpt * 1000),
                     "p": p.tolist()}
    q = results[name]
    print(f"{name:45s} t = ({q['t_cl_m'][0] * 1000:6.1f}, {q['t_cl_m'][1] * 1000:6.1f}, {q['t_cl_m'][2] * 1000:6.1f}) mm, "
          f"scale/encoder {q['scale_vs_encoder']:.4f}, bias {q['range_bias_mm']:6.2f} mm, rot {q['rot_refine_deg']:.2f} deg | "
          f"rms {q['rms_mm']:.2f} mm, median |r| {q['median_abs_mm']:.2f} mm, inliers {q['inliers']:.0%}, "
          f"residual = {q['trend_intercept_mm']:.2f} + {q['trend_slope_mm']:.2f} * (0.1 m / range) mm")

json.dump({"fixed_scale": SCALE, "leg": LEG, "range_m": [RLO, RHI], "fixed_mm_per_unit": lam_enc * 1000, "scans": len(scans), "points": int(len(P_l)),
           "variants": results}, open(os.path.join(OUT, f"f3_register{TAG}.json"), "w"), indent=1)

# residual against range, per variant
fig, ax = plt.subplots(2, 4, figsize=(20, 8), sharey=True)
bins = np.linspace(0.12, 0.9, 27)
for a_, (name, q) in zip(ax.ravel(), results.items()):
    p = np.array(q["p"])
    _, r, keep = solve(p, [], R_cl0, iters=0)
    k = np.digitize(RHO[keep], bins)
    med = [np.median(r[keep][k == j]) * 1000 if (k == j).sum() > 50 else np.nan for j in range(1, len(bins))]
    a_.plot(RHO[keep][::50], r[keep][::50] * 1000, ".", ms=1, alpha=.3)
    a_.plot((bins[1:] + bins[:-1]) / 2, med, "r-o", ms=3)
    a_.axhline(0, color="k", lw=.5)
    a_.set_title(name[:28] + f"\nrms {q['rms_mm']:.1f} mm", fontsize=8)
    a_.set_xlabel("LiDAR range (m)")
    a_.set_ylim(-15, 15)
    a_.grid(alpha=.3)
ax[0, 0].set_ylabel("point-to-surface (mm), + = outside the vision wall")
plt.tight_layout()
plt.savefig(os.path.join(OUT, f"f3_residual_vs_range{TAG}.png"), dpi=80)
