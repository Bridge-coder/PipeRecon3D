"""F3 (second method): translation t_cl from two geometric observations, instead of point-to-surface ICP.

ICP of LiDAR points against the dense vision surface (f3_register.py) turned out to be ill-conditioned in
this nearly cylindrical pipe: axial offset and scale trade against each other (54-150 mm, scale 0.89-1.0
across variants). Here the two parts of t_cl come from quantities that each sensor measures directly:

  lateral (perpendicular to the pipe): the LiDAR origin's offset from the pipe axis (centre of its own
     circle fit - unaffected by a radial range bias) minus the camera centre's offset from the vision axis
     of the same leg (tag scale) = the cross-section part of R(t)^T t_cl. Linear least squares over many
     scans; done per leg as a consistency check.
  axial (along the pipe): the LiDAR's fitted axis turns from the leg-1 to the leg-2 direction while its
     fit window passes the bend; the camera passes the bend later. lead = encoder travel between the two
     moments - (distance of the window's point mass ahead of the LiDAR).

usage: python -I f3b_translation.py EXTRACT_DIR VISION_WORK RESULTS_DIR
"""
import csv
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from timebase import load_offsets, stream_times  # noqa: E402
from vision_model import load_cameras, pose_at  # noqa: E402

EX, VW, RES = sys.argv[1:4]
OUT = os.path.join(RES, "fusion")
off = load_offsets(os.path.join(OUT, "time_offsets.json"))
R_cl = np.array(json.load(open(os.path.join(OUT, "extrinsic_rotation.json")))["R_cl"])
lam = json.load(open(os.path.join(OUT, "f4_tag_scale.json")))["mm_per_unit_tag"] / 1000
legs = json.load(open(os.path.join(OUT, "vision_legs.json")))
cam = load_cameras(EX, VW, off)
a = [np.array(L["dir"]) for L in legs["legs"]]
pt = [np.array(L["point"]) for L in legs["legs"]]
B = np.array(legs["bend_point"])
tE, rE = stream_times(EX, off, "encoders")
sE = np.array([(float(r["v6"]) - float(r["v7"])) / 2 for r in rE])
tS, rS = stream_times(EX, off, "lidar")
fits = {int(r["idx"]): r for r in csv.DictReader(open(os.path.join(EX, "lidar_fit_short.csv"))) if r.get("diameter_mm")}
D_med = np.median([float(r["diameter_mm"]) for r in fits.values()])
UP = np.array([0.0, -1, 0])
t_split = cam["t"][legs["split_camera"]]
u_cam = np.where(np.arange(len(cam["t"])) < legs["split_camera"], (cam["C"] - B) @ a[0], (cam["C"] - B) @ a[1])
t_cam_bend = np.interp(0.0, u_cam, cam["t"].astype(float))     # camera passes the bend (u goes - to +)


def perp(v, d):
    return v - (v @ d) * d


# ---------- axial: when does the LiDAR's fitted axis swing from leg 1 to leg 2?
frac, tt, lat_rows = [], [], []
bend_ang = np.arccos(np.clip(a[0] @ a[1], -1, 1))
for t, row in zip(tS, rS):
    f = fits.get(int(row["idx"]))
    if f is None or t < cam["t"][0] or t > cam["t"][-1]:
        continue
    if abs(float(f["diameter_mm"]) - D_med) > 12 or float(f["inlier_share"]) < 0.5:
        continue
    R, C = pose_at(cam, t)
    dL = np.array([float(f["dir_x"]), float(f["dir_y"]), float(f["dir_z"])])
    aw = R.T @ R_cl @ dL
    # position between the two leg directions, projected on the plane they span
    n = np.cross(a[0], a[1])
    n /= np.linalg.norm(n)
    ap = perp(aw, n)
    ap /= np.linalg.norm(ap)
    ang1 = np.arctan2(np.cross(a[0], ap) @ n, a[0] @ ap)
    frac.append(ang1 / bend_ang)
    tt.append(t)
    # lateral observation rows (used below)
    leg = 0 if t < t_split - 12e9 else (1 if t > t_split + 3e9 else -1)
    if leg >= 0:
        side_v = np.cross(dL, perp(UP, dL) / np.linalg.norm(perp(UP, dL)))
        up_v = perp(UP, dL) / np.linalg.norm(perp(UP, dL))
        p0 = (float(f["off_side_mm"]) * side_v + float(f["off_up_mm"]) * up_v) / -1000.0   # axis point, LiDAR frame
        oL_w = R.T @ R_cl @ (-p0)                                       # LiDAR origin minus axis, world dirs, metres
        foot = pt[leg] + ((C - pt[leg]) @ a[leg]) * a[leg]
        oC_w = (C - foot) * lam                                         # camera centre minus axis, metres
        lat_rows.append((leg, t, R, oL_w, oC_w))
frac, tt = np.array(frac), np.array(tt)
k31 = np.ones(31) / 31
fs = np.convolve(np.pad(frac, 15, mode="edge"), k31, mode="valid")
j = int(np.argmax(fs > 0.5))
t_lidar_bend = tt[j - 1] + (0.5 - fs[j - 1]) / (fs[j] - fs[j - 1]) * (tt[j] - tt[j - 1])
# point mass of the fit window: median distance ahead of the LiDAR of the wall points it used
s_mid = []
for t, row in list(zip(tS, rS))[::50]:
    f = fits.get(int(row["idx"]))
    if f is None:
        continue
    A = np.load(os.path.join(EX, "lidar", f"{int(row['idx']):06d}.npy"))
    P = np.column_stack([A["x"], A["y"], A["z"]]).astype(float)
    P = P[np.isfinite(P).all(1) & (np.abs(P).sum(1) > 0)]
    dL = np.array([float(f["dir_x"]), float(f["dir_y"]), float(f["dir_z"])])
    s = P @ dL
    rr = np.linalg.norm(P - np.outer(s, dL), axis=1)
    m = (s > 0.15) & (s < 0.45) & (np.abs(rr - float(f["diameter_mm"]) / 2000) < 0.03)
    if m.sum() > 100:
        s_mid.append(np.median(s[m]))
c_w = float(np.median(s_mid))
ds = np.interp(t_cam_bend, tE, sE) - np.interp(t_lidar_bend, tE, sE)
lead = ds - c_w
print(f"LiDAR axis swings to the leg-2 direction at t = {(t_lidar_bend - tE[0]) / 1e9:.1f} s, camera passes the bend at "
      f"{(t_cam_bend - tE[0]) / 1e9:.1f} s; encoder travel between them {ds * 1000:.0f} mm; fit-window point mass "
      f"{c_w * 1000:.0f} mm ahead of the LiDAR -> LiDAR origin {lead * 1000:.0f} mm ahead of the camera centre")

# ---------- lateral: perp_a(R^T t) = oL - oC  (axis component of t not observable here)
sol = {}
for name, sel in (("leg 1", [r for r in lat_rows if r[0] == 0]), ("leg 2", [r for r in lat_rows if r[0] == 1]),
                  ("both", lat_rows)):
    M, y = [], []
    for leg, t, R, oL, oC in sel:
        Pa = np.eye(3) - np.outer(a[leg], a[leg])
        M.append(Pa @ R.T)
        y.append(oL - oC)
    M, y = np.vstack(M), np.concatenate(y)
    # unknown t (camera frame); fix its component along the camera z (~ pipe axis) to the axial estimate
    Mz, Mxy = M[:, 2], M[:, :2]
    txy, *_ = np.linalg.lstsq(Mxy, y - Mz * lead, rcond=None)
    tvec = np.array([txy[0], txy[1], lead])
    res = (y - M @ tvec).reshape(-1, 3)
    sol[name] = (tvec, np.sqrt(np.mean(np.sum(res ** 2, 1))), len(sel))
    print(f"lateral from {name:5s} ({len(sel):4d} scans): t_cl = ({tvec[0] * 1000:6.1f}, {tvec[1] * 1000:6.1f}, {tvec[2] * 1000:6.1f}) mm, "
          f"residual rms {sol[name][1] * 1000:.1f} mm")
t_final = sol["both"][0]
json.dump({"t_cl_m": t_final.tolist(), "lead_m": lead, "window_point_mass_m": c_w,
           "per_leg": {k: {"t_cl_m": v[0].tolist(), "rms_m": v[1], "scans": v[2]} for k, v in sol.items()},
           "t_lidar_bend_ns": int(t_lidar_bend), "t_camera_bend_ns": int(t_cam_bend)},
          open(os.path.join(OUT, "extrinsic_translation.json"), "w"), indent=1)
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot((tt - tE[0]) / 1e9, frac, ".", ms=2, label="LiDAR axis: 0 = leg-1 direction, 1 = leg-2 direction")
ax.plot((tt - tE[0]) / 1e9, fs, "r-", label="smoothed")
ax.axvline((t_lidar_bend - tE[0]) / 1e9, color="r", ls="--", lw=.8)
ax.axvline((t_cam_bend - tE[0]) / 1e9, color="k", ls="--", lw=.8, label="camera passes the bend")
ax.set_xlabel("time (s)"); ax.set_ylim(-0.5, 1.5); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "f3b_axial.png"), dpi=85)
