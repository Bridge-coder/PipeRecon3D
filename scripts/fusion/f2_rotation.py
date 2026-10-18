"""F2: rotation between the LiDAR frame and the colour-camera (optical) frame, R_cl  (x_cam = R_cl x_lidar + t_cl).

Measurements (all on the common clock of F1):
  - LiDAR IMU (frame "hesai_lidar"; assumed to share the axes of the point cloud - checked: both have +y down)
  - RealSense IMU (camera_imu_optical_frame) -> colour optical frame through tf_static
  - vision SfM camera rotations (only as an independent check)
Constraints:
  1. gravity: mean specific force in each static period (robot standing still for > 2 s)
  2. rotation increments: gyro integrated over 1 s windows while driving, bias removed (static means)
     R_cl * dtheta_lidar = dtheta_cam  (rigid body; small-angle increments add like vectors)
  3. pipe axis: LiDAR short-window cylinder axis (lidar_fit_short.csv) against the axis of the vision leg (vision_legs.json)
     seen from the camera at the same time - needed because 1 and 2 are nearly parallel
Solved jointly (weighted Kabsch) and with subsets; residual angles of every set are reported.
Check: SfM rotation increments between registered frames against the (rotated) gyro increments.

usage: python -I f2_rotation.py EXTRACT_DIR VISION_WORK RESULTS_DIR
writes RESULTS_DIR/fusion/extrinsic_rotation.json and f2_rotation.png
"""
import csv
import json
import os
import subprocess
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from timebase import load_offsets, stream_times  # noqa: E402

EX, VW, RES = sys.argv[1:4]
OUT = os.path.join(RES, "fusion")
off = load_offsets(os.path.join(OUT, "time_offsets.json"))


def quat_to_R(x, y, z, w):
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rotvec(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    a = np.arccos(c)
    if a < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(a))
    return a * v


def kabsch(A, B, w):
    """R minimising sum w |B - R A|^2 (rows of A, B are vectors)."""
    H = (A * w[:, None]).T @ B
    U, S, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def ang_deg(a, b):
    return np.degrees(np.arccos(np.clip(np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)), -1, 1)))


# --- RealSense IMU optical frame -> colour optical frame, from tf_static
info = json.load(open(os.path.join(EX, "bag_info.json")))
tfs = {(t["parent"], t["child"]): t for t in info["tf_static"]}


def chain(frames):
    R = np.eye(3)
    for p, c in zip(frames[:-1], frames[1:]):
        t = tfs[(p, c)]
        R = R @ quat_to_R(t["qx"], t["qy"], t["qz"], t["qw"])
    return R          # R_parent<-child of the whole chain


R_link_imu = chain(["camera_link", "camera_gyro_frame", "camera_imu_frame", "camera_imu_optical_frame"])
R_link_col = chain(["camera_link", "camera_color_frame", "camera_color_optical_frame"])
R_col_imu = R_link_col.T @ R_link_imu
print(f"RealSense IMU -> colour optical rotation from tf_static: {np.degrees(np.linalg.norm(rotvec(R_col_imu))):.2f} deg")

# --- IMU data on the common clock
tL, rL = stream_times(EX, off, "imu_lidar")
GL = np.array([[float(r[k]) for k in ("gx", "gy", "gz")] for r in rL])
AL = np.array([[float(r[k]) for k in ("ax", "ay", "az")] for r in rL])
tC, rC = stream_times(EX, off, "imu_camera")
GC = np.array([[float(r[k]) for k in ("gx", "gy", "gz")] for r in rC]) @ R_col_imu.T
AC = np.array([[float(r[k]) for k in ("ax", "ay", "az")] for r in rC]) @ R_col_imu.T

# --- static periods from the encoders (no wheel motion for >= 2 s)
tE, rE = stream_times(EX, off, "encoders")
sE = np.array([(float(r["v6"]) - float(r["v7"])) / 2 for r in rE])
grid = np.arange(max(tL[0], tC[0]), min(tL[-1], tC[-1]), 1e8).astype(np.int64)     # 10 Hz
s_g = np.interp(grid, tE, sE)
moving = np.abs(np.gradient(s_g, 0.1)) > 0.001          # > 1 mm/s
static_runs = []
k = 0
while k < len(grid):
    if not moving[k]:
        j = k
        while j < len(grid) and not moving[j]:
            j += 1
        if (grid[j - 1] - grid[k]) > 2e9:
            static_runs.append((grid[k] + 3e8, grid[j - 1] - 3e8))        # trim 0.3 s at both ends
        k = j
    else:
        k += 1
print(f"{len(static_runs)} static periods: " + ", ".join(f"{(a - grid[0]) / 1e9:.1f}-{(b - grid[0]) / 1e9:.1f} s" for a, b in static_runs))


def mean_in(t, X, a, b):
    m = (t >= a) & (t <= b)
    return X[m].mean(0), m.sum()


grav_L, grav_C = [], []
for a, b in static_runs:
    gl, nl = mean_in(tL, AL, a, b)
    gc, nc = mean_in(tC, AC, a, b)
    if nl > 50 and nc > 50:
        grav_L.append(gl)
        grav_C.append(gc)
grav_L, grav_C = np.array(grav_L), np.array(grav_C)
bias_L = np.concatenate([GL[(tL >= a) & (tL <= b)] for a, b in static_runs]).mean(0)
bias_C = np.concatenate([GC[(tC >= a) & (tC <= b)] for a, b in static_runs]).mean(0)


def integrate(t, G, bias, a, b):
    m = (t >= a) & (t < b)
    if m.sum() < 10:
        return None
    tt = t[m] / 1e9
    g = G[m] - bias
    return np.sum(0.5 * (g[1:] + g[:-1]) * np.diff(tt)[:, None], axis=0)


inc_L, inc_C, inc_t = [], [], []
for a in np.arange(grid[0], grid[-1] - 1e9, 5e8).astype(np.int64):
    b = a + int(1e9)
    if not moving[(grid >= a) & (grid < b)].all():
        continue
    dl = integrate(tL, GL, bias_L, a, b)
    dc = integrate(tC, GC, bias_C, a, b)
    if dl is None or dc is None:
        continue
    if np.linalg.norm(dc) > np.radians(0.2) and np.linalg.norm(dl) > np.radians(0.2):
        inc_L.append(dl)
        inc_C.append(dc)
        inc_t.append(a)
inc_L, inc_C, inc_t = np.array(inc_L), np.array(inc_C), np.array(inc_t)
print(f"gravity pairs {len(grav_L)}, rotation-increment pairs {len(inc_L)} (1 s windows with > 0.2 deg)")


# --- 3. pipe-axis pairs: LiDAR short-window axis (LiDAR frame) against the vision leg axis seen from the camera
# (gravity and the gyro increments are nearly parallel - the robot only turns about the vertical - so on
# their own they leave the rotation about the vertical undetermined; the pipe axis is horizontal and fixes it)
from vision_model import load_cameras  # noqa: E402

cam = load_cameras(EX, VW, off)
legs = json.load(open(os.path.join(OUT, "vision_legs.json")))
m_per_unit = 1.730 / legs["path_units"]          # encoder travel / vision path (scale only for leg selection)
s_bend = legs["bend_point_path_units"] * m_per_unit
LEAD_MAX, WIN_HI = 0.30, 0.45                    # LiDAR at most 0.3 m ahead of the camera; fit window ends 0.45 m ahead


def R_at(t):
    """World->camera rotation at common time t (slerp between registered frames; clamped at both ends)."""
    if t <= cam["t"][0]:
        return cam["R"][0]
    if t >= cam["t"][-1]:
        return cam["R"][-1]
    j = int(np.searchsorted(cam["t"], t)) - 1
    a = (t - cam["t"][j]) / (cam["t"][j + 1] - cam["t"][j])
    Ra, Rb = cam["R"][j].T, cam["R"][j + 1].T                    # camera->world
    v = rotvec(Ra.T @ Rb) * a
    th = np.linalg.norm(v)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) / (th if th > 0 else 1)
    E = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
    return (Ra @ E).T


tS, rS = stream_times(EX, off, "lidar")
fits = {int(r["idx"]): r for r in csv.DictReader(open(os.path.join(EX, "lidar_fit_short.csv"))) if r.get("diameter_mm")}
D_med = np.median([float(r["diameter_mm"]) for r in fits.values()])
ax_L, ax_C, ax_leg, ax_t = [], [], [], []
static_kept = 0
for t, row in zip(tS, rS):
    i = int(row["idx"])
    f = fits.get(i)
    if f is None or abs(float(f["diameter_mm"]) - D_med) > 10 or float(f["inlier_share"]) < 0.5:
        continue
    s_c = np.interp(t, cam["t"], cam["s"]) * m_per_unit
    if s_c < s_bend - LEAD_MAX - WIN_HI:
        leg = 0
        if t < cam["t"][0]:                       # standing at the start: keep one scan in 20
            static_kept += 1
            if static_kept % 20:
                continue
    elif s_c > s_bend + 0.05:
        leg = 1
    else:
        continue
    dL = np.array([float(f["dir_x"]), float(f["dir_y"]), float(f["dir_z"])])
    dC = R_at(t) @ np.array(legs["legs"][leg]["dir"])
    ax_L.append(dL)
    ax_C.append(dC)
    ax_leg.append(leg)
    ax_t.append(t)
ax_L, ax_C, ax_leg, ax_t = map(np.array, (ax_L, ax_C, ax_leg, ax_t))
print(f"pipe-axis pairs: {np.sum(ax_leg == 0)} in leg 1, {np.sum(ax_leg == 1)} in leg 2")

gL = grav_L / np.linalg.norm(grav_L, axis=1, keepdims=True)
gC = grav_C / np.linalg.norm(grav_C, axis=1, keepdims=True)
uL = inc_L / np.linalg.norm(inc_L, axis=1, keepdims=True)
uC = inc_C / np.linalg.norm(inc_C, axis=1, keepdims=True)
w_inc = np.linalg.norm(inc_C, axis=1)
w_inc = w_inc / w_inc.sum()


def solve(sets):
    A = np.vstack([s[0] for s in sets])
    B = np.vstack([s[1] for s in sets])
    w = np.concatenate([s[2] for s in sets])
    return kabsch(A, B, w)


def eq(n):
    return np.full(n, 1.0 / n)


sol = {
    "gravity + axis": solve([(gL, gC, eq(len(gL))), (ax_L, ax_C, eq(len(ax_L)))]),
    "gravity + axis + increments": solve([(gL, gC, eq(len(gL))), (ax_L, ax_C, eq(len(ax_L))), (uL, uC, 0.5 * w_inc)]),
    "axis + increments (no gravity)": solve([(ax_L, ax_C, eq(len(ax_L))), (uL, uC, w_inc)]),
    "leg 1 axis + gravity": solve([(gL, gC, eq(len(gL))), (ax_L[ax_leg == 0], ax_C[ax_leg == 0], eq(np.sum(ax_leg == 0)))]),
    "leg 2 axis + gravity": solve([(gL, gC, eq(len(gL))), (ax_L[ax_leg == 1], ax_C[ax_leg == 1], eq(np.sum(ax_leg == 1)))]),
}
R_final = sol["gravity + axis + increments"]
lines = []
for name, R in sol.items():
    e_g = ang_deg(gL @ R.T, gC)
    e_a = ang_deg(ax_L @ R.T, ax_C)
    e_i = ang_deg(inc_L @ R.T, inc_C)
    lines.append(f"{name:32s}: differs from final by {np.degrees(np.linalg.norm(rotvec(R @ R_final.T))):5.2f} deg | "
                 f"residuals gravity {np.median(e_g):.2f}, axis median {np.median(e_a):.2f} (leg1 {np.median(e_a[ax_leg == 0]):.2f}, "
                 f"leg2 {np.median(e_a[ax_leg == 1]):.2f}), increments size-weighted {np.sum(w_inc * e_i):.1f} deg")
print("\n".join(lines))

# --- check with the vision SfM rotations (independent of every constraint above)
sfm_inc, gyroC_inc, gyroL_inc = [], [], []
for j in range(len(cam["t"]) - 1):
    ta, tb = cam["t"][j], cam["t"][j + 1]
    v = rotvec(cam["R"][j] @ cam["R"][j + 1].T)                  # body rotation from a to b, in frame a
    if np.linalg.norm(v) < np.radians(1.0):
        continue
    sfm_inc.append(v)
    gyroC_inc.append(integrate(tC, GC, bias_C, ta, tb))
    gyroL_inc.append(integrate(tL, GL, bias_L, ta, tb) @ R_final.T)
sfm_inc, gyroC_inc, gyroL_inc = map(np.array, (sfm_inc, gyroC_inc, gyroL_inc))
chk = (f"SfM check, {len(sfm_inc)} frame pairs turning > 1 deg: direction vs RealSense gyro median "
       f"{np.median(ang_deg(sfm_inc, gyroC_inc)):.1f} deg, vs LiDAR gyro rotated by R_cl median "
       f"{np.median(ang_deg(sfm_inc, gyroL_inc)):.1f} deg; size SfM/RealSense "
       f"{np.median(np.linalg.norm(sfm_inc, axis=1) / np.linalg.norm(gyroC_inc, axis=1)):.3f}")
print(chk)
tot_sfm = rotvec(cam["R"][0] @ cam["R"][-1].T)
tot_L = integrate(tL, GL, bias_L, cam["t"][0], cam["t"][-1]) @ R_final.T
tot_C = integrate(tC, GC, bias_C, cam["t"][0], cam["t"][-1])
tot = (f"whole run, camera axes (deg): SfM {np.round(np.degrees(tot_sfm), 1)}, RealSense gyro {np.round(np.degrees(tot_C), 1)}, "
       f"LiDAR gyro rotated {np.round(np.degrees(tot_L), 1)}")
print(tot)
zc = R_final[:, 2]
yc = R_final[:, 1]
geo = (f"LiDAR z (forward) in camera axes {np.round(zc, 3)} -> {np.degrees(np.arctan2(zc[0], zc[2])):.1f} deg right of the "
       f"optical axis, {np.degrees(np.arctan2(-zc[1], zc[2])):.1f} deg up; LiDAR y (down) in camera axes {np.round(yc, 3)}")
print(geo)
json.dump({"R_cl": R_final.tolist(), "solutions": {k: v.tolist() for k, v in sol.items()},
           "gravity_pairs": int(len(gL)), "axis_pairs": int(len(ax_L)), "increment_pairs": int(len(inc_L)),
           "gyro_bias_lidar": bias_L.tolist(), "gyro_bias_cam": bias_C.tolist(), "R_colour_from_rs_imu": R_col_imu.tolist(),
           "report": lines + [chk, tot, geo]},
          open(os.path.join(OUT, "extrinsic_rotation.json"), "w"), indent=1)

fig, ax = plt.subplots(1, 3, figsize=(17, 5))
lab = ["x", "y", "z"]
for j in range(3):
    ax[0].plot(np.degrees(inc_C[:, j]), np.degrees((inc_L @ R_final.T)[:, j]), ".", ms=5, label=f"camera {lab[j]}")
lim = np.degrees(np.abs(inc_C).max()) * 1.1
ax[0].plot([-lim, lim], [-lim, lim], "k", lw=.5)
ax[0].set_xlabel("RealSense gyro, 1 s increment (deg)"); ax[0].set_ylabel("LiDAR gyro rotated by R_cl (deg)")
ax[0].legend(); ax[0].grid(alpha=.3); ax[0].set_title("rotation increments (camera axes)")
for j in range(3):
    ax[1].plot(np.degrees(sfm_inc[:, j]), np.degrees(gyroL_inc[:, j]), "x", ms=6, label=f"LiDAR rot. {lab[j]}")
    ax[1].plot(np.degrees(sfm_inc[:, j]), np.degrees(gyroC_inc[:, j]), ".", ms=6, label=f"RealSense {lab[j]}")
lim = np.degrees(np.abs(sfm_inc).max()) * 1.1
ax[1].plot([-lim, lim], [-lim, lim], "k", lw=.5)
ax[1].set_xlabel("SfM rotation between frames (deg)"); ax[1].set_ylabel("gyro (deg)"); ax[1].legend(fontsize=7)
ax[1].grid(alpha=.3); ax[1].set_title("independent check against vision SfM")
e_a = ang_deg(ax_L @ R_final.T, ax_C)
ax[2].plot((ax_t - ax_t[0]) / 1e9, e_a, ".", ms=3)
ax[2].set_xlabel("time (s)"); ax[2].set_ylabel("angle between LiDAR and vision pipe axis (deg)")
ax[2].grid(alpha=.3); ax[2].set_title("pipe-axis residuals (leg 1 early, leg 2 late)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "f2_rotation.png"), dpi=85)
