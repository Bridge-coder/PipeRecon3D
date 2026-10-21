"""Referee for the InPipe pipe radius (docs/10 3.7, proposed in docs/history/inpipe_172011/03 and never run):
the RealSense stereo depth has its own scale (factory baseline 49.8 mm), independent of LiDAR, AprilTag and encoder.
Every N-th depth frame is back-projected (depth intrinsics), moved into the colour camera with the factory
extrinsic (tf_static), into the world with the same fused + R2b-corrected pose that R4, lampmap and the Gaussian
models use, and put into pipe coordinates (u, theta, r) about the R2 centerline. Its wall radius is compared point by
point with the R3 LiDAR height field r(u, theta) at the same place.

usage: python -I scripts/analysis/depth_referee.py OUT_DIR [--every 20] [--zmin 0.26] [--zmax 0.60]
writes OUT_DIR/depth_referee.json and depth_referee.png (radius vs u: depth, LiDAR height field), and
       OUT_DIR/radius_correction.npz: a smooth correction of the height field that the depth asks for,
       dr(u, theta) = a(u) + b(u) cos(theta) + c(u) sin(theta) (radius offset + centre shift), fitted to the medians
       of depth - LiDAR in 5 cm x 15 deg cells, smoothed over 3 cells along u (pipesplat/export_surface.py --radius-correction)
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "recon"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "fusion"))
from recon_common import BIG, EX, Centerline, FusedPoses, off, stream_times  # noqa: E402
from vision_model import _expmap  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)


def quat_R(qx, qy, qz, qw):
    x, y, z, w = qx, qy, qz, qw
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"), ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--zmin", type=float, default=0.26), ap.add_argument("--zmax", type=float, default=0.60)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    info = json.load(open(os.path.join(EX, "bag_info.json"), encoding="utf-8"))
    kd = np.array(info["camera_info"]["depth"]["k"]).reshape(3, 3)
    tf = {(t["parent"], t["child"]): t for t in info["tf_static"]}

    def T(parent, child):
        t = tf[(parent, child)]
        return quat_R(t["qx"], t["qy"], t["qz"], t["qw"]), np.array([t["x"], t["y"], t["z"]])

    # depth optical -> camera_link -> colour frame -> colour optical
    R_do, _ = T("camera_depth_frame", "camera_depth_optical_frame")         # optical axes in the depth frame
    R_cf, t_cf = T("camera_link", "camera_color_frame")
    R_co, _ = T("camera_color_frame", "camera_color_optical_frame")
    R_dc = R_co.T @ R_cf.T @ R_do                     # depth optical -> colour optical
    t_dc = R_co.T @ (R_cf.T @ (np.zeros(3) - t_cf))
    print(f"depth -> colour: rotation {np.degrees(np.arccos((np.trace(R_dc) - 1) / 2)):.2f} deg, "
          f"shift {np.round(t_dc * 1000, 2).tolist()} mm")

    # the pose of R4 / lampmap / the Gaussian models (pipesplat/export_g0.py export_inpipe)
    fp = FusedPoses()
    sc = np.load(os.path.join(BIG, "scan_corrections.npz"))
    tsc = sc["t"].astype(float)

    def pose_at(t):
        R, C, _ = fp.pose(t)
        g = lambda A: np.array([np.interp(float(t), tsc, A[:, j]) for j in range(3)])  # noqa: E731
        d, wv, O = g(sc["d"]), g(sc["w"]), g(sc["O"])
        return R @ _expmap(wv).T, C + d + np.cross(wv, C - O)

    color_t, color_rows = stream_times(EX, off, "color")
    t_of = {int(r["idx"]): t for r, t in zip(color_rows, color_t)}
    stamp_c = {int(r["idx"]): int(r["stamp_ns"]) for r in color_rows}
    import csv
    drows = list(csv.DictReader(open(os.path.join(EX, "images", "depth_index.csv"), encoding="utf-8")))
    by_stamp = {s: i for i, s in stamp_c.items()}
    cl = Centerline(os.path.join(BIG, "centerline.npz"))
    hf = np.load(os.path.join(BIG, "heightfield.npz"))
    hu, hth, hr = hf["u"], hf["th"], hf["r_mesh"]
    t_lo, t_hi = float(min(fp.cam["t"])), float(max(fp.cam["t"]))
    H, W = 720, 1280
    vv, uu = np.mgrid[0:H, 0:W]
    rays = np.stack([(uu - kd[0, 2]) / kd[0, 0], (vv - kd[1, 2]) / kd[1, 1], np.ones((H, W))], -1)
    U, TH, RD, RL, FR = [], [], [], [], []
    used = 0
    for r in drows[::a.every]:
        ci = by_stamp.get(int(r["stamp_ns"]))
        if ci is None or ci not in t_of or not (t_lo <= t_of[ci] <= t_hi):
            continue
        dimg = cv2.imread(os.path.join(EX, "images", "depth", f"{int(r['idx']):06d}.png"), cv2.IMREAD_UNCHANGED)
        z = dimg.astype(np.float64) / 1000.0
        ok = (z > a.zmin) & (z < a.zmax)
        if ok.sum() < 500:
            continue
        Xd = rays[ok] * z[ok][:, None]
        Xc = Xd @ R_dc.T + t_dc
        R, C = pose_at(t_of[ci])
        Xw = Xc @ R + C                                 # R: world -> camera
        u, th, rad = cl.coords(Xw)
        iu = np.round((u - hu[0]) / (hu[1] - hu[0])).astype(int)
        it = np.round((th - hth[0]) / (hth[1] - hth[0])).astype(int) % len(hth)
        inside = (iu >= 0) & (iu < len(hu))
        rl = np.full(len(u), np.nan)
        rl[inside] = hr[iu[inside], it[inside]]
        U.append(u), TH.append(th), RD.append(rad), RL.append(rl), FR.append(np.full(len(u), ci))
        used += 1
    U, TH, RD, RL, FR = (np.concatenate(v) for v in (U, TH, RD, RL, FR))
    keep = np.isfinite(RL) & (RD > 0.05) & (RD < 0.16) & (np.abs(RD - RL) < 0.04)
    print(f"{used} depth frames, {len(U):,} points, {keep.sum():,} on the wall next to a height-field cell")
    d = (RD - RL)[keep] * 1000
    legs = {"leg 1 (u < 0)": U[keep] < 0, "leg 2 (u > 0.05)": U[keep] > 0.05}
    res = dict(depth_frames=used, points=int(keep.sum()), zrange_m=[a.zmin, a.zmax],
               depth_minus_lidar_mm=dict(median=float(np.median(d)), p25=float(np.percentile(d, 25)),
                                         p75=float(np.percentile(d, 75))), legs={})
    for name, m in legs.items():
        if m.sum() < 100:
            continue
        res["legs"][name] = dict(points=int(m.sum()), depth_radius_mm=float(np.median(RD[keep][m] * 1000)),
                                 lidar_radius_mm=float(np.median(RL[keep][m] * 1000)),
                                 depth_minus_lidar_mm=float(np.median(d[m])),
                                 by_angle_deg={f"{lo}..{lo + 60}": float(np.median(d[m & (TH[keep] >= lo) & (TH[keep] < lo + 60)]))
                                               for lo in range(-180, 180, 60) if (m & (TH[keep] >= lo) & (TH[keep] < lo + 60)).sum() > 100})
        print(f"{name}: {m.sum():,} points, radius depth {res['legs'][name]['depth_radius_mm']:.1f} mm, "
              f"LiDAR height field {res['legs'][name]['lidar_radius_mm']:.1f} mm, depth - LiDAR median "
              f"{res['legs'][name]['depth_minus_lidar_mm']:+.1f} mm; by angle {res['legs'][name]['by_angle_deg']}")
    # smooth correction field: per 5 cm of u, least squares of a + b cos + c sin over the 15-degree cell medians
    ub = np.arange(hu[0], hu[-1] + 0.05, 0.05)
    tb = np.arange(-180, 181, 15)
    ku, kt = np.digitize(U[keep], ub) - 1, np.digitize(TH[keep], tb) - 1
    dd = (RD - RL)[keep]
    coef = np.full((len(ub) - 1, 3), np.nan)
    for i in range(len(ub) - 1):
        rows_, ys = [], []
        for j in range(len(tb) - 1):
            m = (ku == i) & (kt == j)
            if m.sum() >= 30:
                t = np.radians((tb[j] + tb[j + 1]) / 2)
                rows_.append([1.0, np.cos(t), np.sin(t)])
                ys.append(np.median(dd[m]))
        if len(rows_) >= 6:
            coef[i] = np.linalg.lstsq(np.array(rows_), np.array(ys), rcond=None)[0]
    good = np.isfinite(coef[:, 0])
    uc = (ub[:-1] + ub[1:]) / 2
    filled = np.column_stack([np.interp(uc, uc[good], coef[good, j]) for j in range(3)])
    sm = np.column_stack([np.convolve(np.pad(filled[:, j], 1, mode="edge"), np.ones(3) / 3, mode="valid") for j in range(3)])
    np.savez(os.path.join(a.out, "radius_correction.npz"), u=uc, a=sm[:, 0], b=sm[:, 1], c=sm[:, 2], measured=good)
    res["correction"] = dict(cells_u=int(len(uc)), measured=int(good.sum()),
                             offset_mm_median=float(np.median(sm[:, 0]) * 1000),
                             centre_shift_mm_median=float(np.median(np.hypot(sm[:, 1], sm[:, 2])) * 1000))
    print(f"correction field: {good.sum()} of {len(uc)} 5 cm cells measured; radius offset median "
          f"{res['correction']['offset_mm_median']:+.1f} mm, centre shift median {res['correction']['centre_shift_mm_median']:.1f} mm")
    json.dump(res, open(os.path.join(a.out, "depth_referee.json"), "w"), indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(10, 7))
    bins = np.arange(hu[0], hu[-1], 0.02)
    k = np.digitize(U[keep], bins)
    xs = [(bins[i - 1] + bins[i]) / 2 for i in range(1, len(bins)) if (k == i).sum() > 50]
    ax[0].plot(xs, [np.median(RD[keep][k == i]) * 1000 for i in range(1, len(bins)) if (k == i).sum() > 50], ".-", label="RealSense depth")
    ax[0].plot(xs, [np.median(RL[keep][k == i]) * 1000 for i in range(1, len(bins)) if (k == i).sum() > 50], ".-", label="LiDAR height field (R3)")
    ax[0].set_ylabel("wall radius mm"), ax[0].legend(), ax[0].grid(alpha=0.3)
    ax[0].set_title("InPipe 172011: wall radius at the same (u, theta)")
    ax[1].hist(d, bins=np.arange(-30, 30, 0.5))
    ax[1].set_xlabel("depth - LiDAR radius, mm"), ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "depth_referee.png"), dpi=110)
    print(f"saved {a.out}/depth_referee.json / .png")


if __name__ == "__main__":
    main()
