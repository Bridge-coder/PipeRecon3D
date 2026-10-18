"""F5: fused trajectory, LiDAR wall coloured by the camera, and the comparison table.

Fused trajectory (metric, one pose per SfM frame and per LiDAR scan):
  - along the pipe: wheel encoders (lag from F1; checked against two AprilTags)
  - across the pipe: the camera's offset from the vision pipe centerline, in metres with the AprilTag scale
  - orientation: vision SfM (checked against both gyros in F2)
  The centerline is the two straight legs of the vision model (vision_legs.json), joined at the bend.
LiDAR wall coloured by the camera: every LiDAR point in the reliable range band is mapped into the colour
image taken closest in time (x_cam = R_cl x_lidar + t_cl, recorded K) - no vision model is involved - and
placed (a) in the fused metric world frame and (b) in pipe coordinates (s from the encoders, angle from the
slot) for unrolled colour / radius maps.

usage: python -I f5_fuse.py EXTRACT_DIR VISION_WORK RESULTS_DIR [--every N] [--range LO HI]
outputs in RESULTS_DIR/fusion (small files) and FUSED_BIG of paths.py (point clouds)
"""
import csv
import json
import os
import sys

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from timebase import load_offsets, stream_times  # noqa: E402
from vision_model import load_cameras, pose_at  # noqa: E402

args = sys.argv[1:]
EVERY, RLO, RHI = 2, 0.20, 0.60
if "--every" in args:
    i = args.index("--every")
    EVERY = int(args[i + 1])
    del args[i:i + 2]
if "--range" in args:
    i = args.index("--range")
    RLO, RHI = float(args[i + 1]), float(args[i + 2])
    del args[i:i + 3]
if "--lead" in args:
    i = args.index("--lead")
    del args[i:i + 2]
EX, VW, RES = args
OUT = os.path.join(RES, "fusion")
from paths import FUSED_BIG as BIG  # noqa: E402
os.makedirs(BIG, exist_ok=True)
off = load_offsets(os.path.join(OUT, "time_offsets.json"))
R_cl = np.array(json.load(open(os.path.join(OUT, "extrinsic_rotation.json")))["R_cl"])
# translation: final values (lateral from colour consistency, axial from the AprilTag board) if present,
# otherwise the geometric method (f3b_translation.py); ICP (f3_register.py) was ill-conditioned here
_tf = os.path.join(OUT, "extrinsic_translation_final.json")
t_cl = np.array(json.load(open(_tf if os.path.exists(_tf) else os.path.join(OUT, "extrinsic_translation.json")))["t_cl_m"])
if "--lead" in sys.argv:                       # override the axial part (sensitivity tests)
    t_cl[2] = float(sys.argv[sys.argv.index("--lead") + 1])
lam = json.load(open(os.path.join(OUT, "f4_tag_scale.json")))["mm_per_unit_tag"] / 1000
legs = json.load(open(os.path.join(OUT, "vision_legs.json")))
cam = load_cameras(EX, VW, off)
ci = json.load(open(os.path.join(EX, "bag_info.json")))["camera_info"]["color"]
K = np.array(ci["k"]).reshape(3, 3)
W, H = ci["width"], ci["height"]
print(f"t_cl = {np.round(t_cl * 1000, 1)} mm, tag scale {lam * 1000:.3f} mm/unit")

# --- centerline of the vision model, metric
a1, a2 = np.array(legs["legs"][0]["dir"]), np.array(legs["legs"][1]["dir"])
B = np.array(legs["bend_point"])


def centerline_coords(P):
    """arc position u (units, 0 at the bend, negative before it) and the closest centerline point."""
    u1 = (P - B) @ a1                       # <= 0 on leg 1
    u2 = (P - B) @ a2                       # >= 0 on leg 2
    on2 = u2 > 0
    u = np.where(on2, u2, np.minimum(u1, 0))
    Q = np.where(on2[:, None], B + np.outer(u2, a2), B + np.outer(np.minimum(u1, 0), a1))
    return u, Q


def centerline_point(u):
    u = np.atleast_1d(u)
    return np.where((u > 0)[:, None], B + np.outer(u, a2), B + np.outer(u, a1))


u_cam, Q_cam = centerline_coords(cam["C"])
lat_cam = (cam["C"] - Q_cam) * lam                          # metres, camera offset from the centerline
tE, rE = stream_times(EX, off, "encoders")
sE = np.array([(float(r["v6"]) - float(r["v7"])) / 2 for r in rE])
s_enc = lambda t: np.interp(t, tE, sE) - np.interp(cam["t"][0], tE, sE)  # noqa: E731
u0 = u_cam[0] * lam                                          # metres, first camera's arc position


def fused_pose(t):
    R, _ = pose_at(cam, t)
    u_m = u0 + s_enc(t)
    lat = np.array([np.interp(t, cam["t"], lat_cam[:, j]) for j in range(3)])
    C = centerline_point(u_m / lam)[0] * lam + lat           # metric world (vision axes, scaled)
    return R, C, u_m


traj = []
for k in range(len(cam["t"])):
    R, C, u_m = fused_pose(cam["t"][k])
    traj.append([cam["idx"][k], cam["t"][k], *C, u_m, *(cam["C"][k] * lam), np.linalg.norm(lat_cam[k]) * 1000])
with open(os.path.join(OUT, "fused_trajectory.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["color_idx", "t_ns", "x_m", "y_m", "z_m", "arc_m", "vision_x_m", "vision_y_m", "vision_z_m", "lateral_mm"])
    w.writerows(traj)
traj = np.array(traj, float)
d_axial = np.linalg.norm(traj[:, 2:5] - traj[:, 6:9], axis=1)
print(f"fused trajectory: {len(traj)} frames; fused vs vision(tag scale) centres differ by up to {d_axial.max() * 1000:.0f} mm "
      f"(end {d_axial[-1] * 1000:.0f} mm); camera lateral offset from the centerline median {np.median(traj[:, 9]):.1f} mm")

# --- LiDAR wall coloured by the camera
tS, rS = stream_times(EX, off, "lidar")
tC, rC = stream_times(EX, off, "color")
fits = {int(r["idx"]): r for r in csv.DictReader(open(os.path.join(EX, "lidar_fit_short.csv"))) if r.get("diameter_mm")}
D_med = np.median([float(r["diameter_mm"]) for r in fits.values()])
UP = np.array([0.0, -1, 0])
rng = np.random.default_rng(0)
acc = {k: [] for k in ("Xw", "rgb", "s", "th", "r", "scan")}
img_cache = {}
for k, (t, row) in enumerate(zip(tS, rS)):
    i = int(row["idx"])
    f = fits.get(i)
    if k % EVERY or f is None or t < cam["t"][0] or t > cam["t"][-1]:
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
    s_loc = Q @ d
    rad = np.linalg.norm(Q - np.outer(s_loc, d), axis=1)
    m = (rho > RLO) & (rho < RHI) & (s_loc > 0.05) & (np.abs(rad - float(f["diameter_mm"]) / 2000) < 0.025)
    if m.sum() < 100:
        continue
    P, s_loc, rad, Q = P[m], s_loc[m], rad[m], Q[m]
    xc = P @ R_cl.T + t_cl
    j = int(np.argmin(np.abs(tC - t)))
    if j not in img_cache:
        if len(img_cache) > 4:
            img_cache.clear()
        img_cache[j] = cv2.imread(os.path.join(EX, "images", "color", f"{int(rC[j]['idx']):06d}.png"))
    img = img_cache[j]
    uvw = xc @ K.T
    ok = uvw[:, 2] > 0.05
    u = np.full(len(xc), -1.0)
    v = np.full(len(xc), -1.0)
    u[ok], v[ok] = uvw[ok, 0] / uvw[ok, 2], uvw[ok, 1] / uvw[ok, 2]
    ok &= (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    if ok.sum() == 0:
        continue
    rgb = img[v[ok].astype(int), u[ok].astype(int)][:, ::-1]
    Rw, Cw = fused_pose(t)[:2]
    Xw = Cw + (xc[ok]) @ Rw                               # R^T x  (world = C + R^T x_cam)
    th = np.degrees(np.arctan2(Q[ok] @ side, Q[ok] @ up))
    th = (th - float(f["slot_centre_deg"]) + 180 + 180) % 360 - 180   # slot at +-180
    acc["Xw"].append(Xw)
    acc["rgb"].append(rgb)
    acc["s"].append(s_enc(t) + s_loc[ok])
    acc["th"].append(th)
    acc["r"].append(rad[ok])
    acc["scan"].append(np.full(ok.sum(), i))
for kk in acc:
    acc[kk] = np.concatenate(acc[kk])
n = len(acc["s"])
print(f"coloured LiDAR wall points: {n:,} (range {RLO}-{RHI} m, every {EVERY}th scan)")

# coloured point cloud (metric world frame), subsampled to <= 4 M points
sel = np.arange(n) if n <= 4_000_000 else rng.choice(n, 4_000_000, replace=False)
with open(os.path.join(BIG, "lidar_coloured_world.ply"), "wb") as f:
    f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(sel)}\nproperty float x\nproperty float y\n"
             f"property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
    rec = np.zeros(len(sel), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = acc["Xw"][sel].T
    rec["r"], rec["g"], rec["b"] = acc["rgb"][sel].T
    f.write(rec.tobytes())

# unrolled maps: colour and radius, 5 mm x 2 deg
s, th, r, rgb = acc["s"], acc["th"], acc["r"], acc["rgb"]
DS, DT = 0.005, 2.0
ns = int(np.ceil((s.max() - s.min()) / DS)) + 1
nt = int(360 / DT)
iS = ((s - s.min()) / DS).astype(int)
iT = np.clip(((th + 180) / DT).astype(int), 0, nt - 1)
flat = iT * ns + iS
cnt = np.bincount(flat, minlength=ns * nt)
col = np.zeros((ns * nt, 3))
for c in range(3):
    col[:, c] = np.bincount(flat, weights=rgb[:, c].astype(float), minlength=ns * nt)
col = (col / np.maximum(cnt, 1)[:, None]).reshape(nt, ns, 3)
col[cnt.reshape(nt, ns) == 0] = 255
rsum = np.bincount(flat, weights=r, minlength=ns * nt)
rmap = (rsum / np.maximum(cnt, 1)).reshape(nt, ns)
rmap[cnt.reshape(nt, ns) == 0] = np.nan
ext = [s.min(), s.max(), -180, 180]
fig, ax = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
ax[0].imshow(col.astype(np.uint8), origin="lower", aspect="auto", extent=ext, interpolation="nearest")
ax[0].set_ylabel("angle (deg, slot at +-180)"); ax[0].set_title("LiDAR wall points coloured from the camera (unrolled)")
im = ax[1].imshow((rmap - np.nanmedian(rmap)) * 1000, origin="lower", aspect="auto", extent=ext, cmap="RdBu_r",
                  vmin=-10, vmax=10, interpolation="nearest")
ax[1].set_ylabel("angle (deg)"); ax[1].set_xlabel("s along the pipe (m, encoders)")
ax[1].set_title(f"LiDAR radius minus median {np.nanmedian(rmap) * 1000:.1f} mm (ranges {RLO}-{RHI} m)")
plt.colorbar(im, ax=ax[1], label="mm", orientation="horizontal", fraction=0.05, pad=0.15)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "f5_unrolled_colour_radius.png"), dpi=90)
cv2.imwrite(os.path.join(OUT, "f5_unrolled_colour.png"), cv2.resize(col.astype(np.uint8)[::-1, :, ::-1], (ns * 2, nt * 4),
                                                                      interpolation=cv2.INTER_NEAREST))
np.savez_compressed(os.path.join(BIG, "lidar_coloured_pipe_coords.npz"), s=s, theta=th, r=r, rgb=rgb, scan=acc["scan"])
print(f"outputs: {OUT}/f5_unrolled_colour_radius.png, f5_unrolled_colour.png, fused_trajectory.csv; {BIG}/lidar_coloured_world.ply")
