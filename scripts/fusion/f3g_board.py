"""Axial part of t_cl from the AprilTag board beyond the pipe exit, seen by both sensors (target-based calibration).

Camera: tag 91 corners -> PnP -> board plane (normal n_c, offset d_c) in the colour-camera frame, metric
        through the tag side (100 mm; also 102.8 mm, the value under which the vision model would be isotropic).
LiDAR:  points of the scan closest in time that lie on that plane (white plate + tag), plane re-fitted in the
        LiDAR frame.
Each LiDAR board point gives n_c . (R_cl x + t_cl) = d_c. The board faces the robot, so n_c is close to the
pipe axis and the constraint fixes the axial component t_z (lateral parts held at the colour-consistency
values). Per-frame estimates against distance separate a constant offset (the extrinsic) from errors that
grow with distance (tag size, LiDAR range scale); the board normal also checks R_cl.

Why this and not the wall: colour consistency is only weakly sensitive to t_z inside a straight pipe
(f3c), the ring seams have no colour contrast and the screws/scratches have no LiDAR return (f3e), and the
low-frequency brightness along the wall comes from the robot's own lamp (f3f) - see 02_融合结果.md.

usage: python -I f3g_board.py EXTRACT_DIR RESULTS_DIR
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
from timebase import load_offsets, stream_times  # noqa: E402

EX, RES = sys.argv[1:3]
OUT = os.path.join(RES, "fusion")
off = load_offsets(os.path.join(OUT, "time_offsets.json"))
R_cl = np.array(json.load(open(os.path.join(OUT, "extrinsic_rotation.json")))["R_cl"])
t_lat = np.array(json.load(open(os.path.join(OUT, "extrinsic_translation_photo.json")))["t_cl_m"][:2])
t_geo = np.array(json.load(open(os.path.join(OUT, "extrinsic_translation.json")))["t_cl_m"])
ci = json.load(open(os.path.join(EX, "bag_info.json")))["camera_info"]["color"]
K = np.array(ci["k"]).reshape(3, 3)
det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), cv2.aruco.DetectorParameters())
tC, rC = stream_times(EX, off, "color")
tS, rS = stream_times(EX, off, "lidar")
tag_rows = list(csv.DictReader(open(os.path.join(RES, "tag", "tag_track_color.csv"))))
idx_to_k = {int(r["idx"]): k for k, r in enumerate(rC)}
cands = [int(r["idx"]) for r in tag_rows if 0.2 < 0.1 * float(r["dist_unit"]) < 0.8][::3]


def board_frames(L):
    obj = L * np.array([[-0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, -0.5, 0], [-0.5, -0.5, 0]], np.float32)
    out = []
    for ci_ in cands:
        g = cv2.imread(os.path.join(EX, "images", "color", f"{ci_:06d}.png"), cv2.IMREAD_GRAYSCALE)
        c, ids, _ = det.detectMarkers(g)
        if ids is None or 91 not in ids.ravel():
            continue
        c = c[list(ids.ravel()).index(91)].reshape(-1, 1, 2).astype(np.float32)
        cv2.cornerSubPix(g, c, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
        ok, rv, tv = cv2.solvePnP(obj, c.reshape(4, 2), K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        Rm, _ = cv2.Rodrigues(rv)
        n = Rm[:, 2]
        ctr = tv.ravel()
        if n @ ctr > 0:          # normal towards the camera
            n = -n
        t_img = tC[idx_to_k[ci_]]
        j = int(np.argmin(np.abs(tS - t_img)))
        if abs(tS[j] - t_img) > 60e6:
            continue
        out.append((ci_, n, float(n @ ctr), ctr, int(rS[j]["idx"])))
    return out


def lidar_board_points(scan, n, d, ctr, t_guess):
    a = np.load(os.path.join(EX, "lidar", f"{scan:06d}.npy"))
    P = np.column_stack([a["x"], a["y"], a["z"]]).astype(float)
    P = P[np.isfinite(P).all(1) & (np.abs(P).sum(1) > 0)]
    X = P @ R_cl.T + t_guess
    dist = X @ n - d
    inpl = np.linalg.norm(X - np.outer(dist, n) - ctr, axis=1)
    sel = (np.abs(dist) < 0.05) & (inpl < 0.12)
    if sel.sum() < 200:
        return None
    Q = P[sel]
    for _ in range(3):                           # plane in the LiDAR frame, 1 cm band
        mu = Q.mean(0)
        _, _, Vt = np.linalg.svd(Q - mu, full_matrices=False)
        nl = Vt[2]
        r = (Q - mu) @ nl
        Q = Q[np.abs(r) < max(0.01, 2.5 * np.std(r))]
    return Q, nl


results = {}
for L in (0.100, 0.1028):
    frames = board_frames(L)
    t_guess = np.array([t_lat[0], t_lat[1], t_geo[2]])
    per = []
    allA, allb = [], []
    for ci_, n, d, ctr, scan in frames:
        bp = lidar_board_points(scan, n, d, ctr, t_guess)
        if bp is None:
            continue
        Q, nl = bp
        # n.(R x + [tx, ty, tz]) = d  ->  n_z tz = d - n.(R x) - n_x tx - n_y ty
        rhs = d - (Q @ R_cl.T) @ n - n[0] * t_lat[0] - n[1] * t_lat[1]
        tz_f = np.median(rhs / n[2])
        nlc = R_cl @ nl
        nlc = nlc if nlc @ n > 0 else -nlc
        per.append((ci_, np.linalg.norm(ctr), tz_f, np.degrees(np.arccos(np.clip(nlc @ n, -1, 1))), len(Q)))
        k = np.random.default_rng(ci_).choice(len(Q), min(400, len(Q)), replace=False)
        allA.append(np.full(len(k), n[2]))
        allb.append(rhs[k])
    per = np.array(per)
    A, b = np.concatenate(allA), np.concatenate(allb)
    tz_all = float(np.sum(A * b) / np.sum(A * A))
    # constant + proportional-to-distance model for the per-frame values
    M = np.column_stack([np.ones(len(per)), per[:, 1]])
    (c0, c1), *_ = np.linalg.lstsq(M, per[:, 2], rcond=None)
    results[L] = {"frames": int(len(per)), "t_z_mm_pooled": tz_all * 1000, "t_z_mm_median": float(np.median(per[:, 2]) * 1000),
                  "t_z_mm_spread_p10_p90": [float(np.percentile(per[:, 2], 10) * 1000), float(np.percentile(per[:, 2], 90) * 1000)],
                  "t_z_mm_at_zero_distance": float(c0 * 1000), "slope_mm_per_m": float(c1 * 1000),
                  "normal_angle_deg_median": float(np.median(per[:, 3])), "per_frame": per.tolist()}
    print(f"tag side {L * 1000:.1f} mm: {len(per)} frames (board 0.2-0.8 m): t_z pooled {tz_all * 1000:.1f} mm, per-frame median "
          f"{np.median(per[:, 2]) * 1000:.1f} mm (p10-p90 {results[L]['t_z_mm_spread_p10_p90'][0]:.1f} .. "
          f"{results[L]['t_z_mm_spread_p10_p90'][1]:.1f}); trend with distance {c1 * 1000:+.1f} mm per m, at 0 m {c0 * 1000:.1f} mm; "
          f"board normal LiDAR (rotated by R_cl) vs camera: median {np.median(per[:, 3]):.2f} deg")
json.dump({"lateral_m": t_lat.tolist(), "by_tag_side_m": {str(k): v for k, v in results.items()}},
          open(os.path.join(OUT, "f3g_board.json"), "w"), indent=1)
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
for L, r in results.items():
    per = np.array(r["per_frame"])
    ax[0].plot(per[:, 1], per[:, 2] * 1000, ".", ms=4, label=f"tag {L * 1000:.1f} mm")
    ax[1].plot(per[:, 1], per[:, 3], ".", ms=4, label=f"tag {L * 1000:.1f} mm")
ax[0].axhline(t_geo[2] * 1000, color="k", ls="--", lw=.8, label="earlier geometric estimate")
ax[0].set_xlabel("board distance from the camera (m)"); ax[0].set_ylabel("t_z from this frame (mm)"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].set_xlabel("board distance (m)"); ax[1].set_ylabel("angle between board normals (deg)"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "f3g_board.png"), dpi=85)
