"""Vision side of the fusion: SfM cameras on the common clock, the bend of the camera path, and the two
straight pipe legs fitted to the OpenMVS dense cloud.

load_cameras(EX, VW, offsets) -> dict with t (common ns), idx (colour frame), R (world->camera), C (centres),
                                 s (arc length of the 5-camera smoothed path, model units)
python -I vision_model.py EXTRACT_DIR VISION_WORK RESULTS_DIR   -> RESULTS_DIR/fusion/vision_legs.json
"""
import csv
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from timebase import load_offsets, stream_times  # noqa: E402

from paths import CONVERTER as CONV  # noqa: E402


def load_cameras(EX, VW, off):
    js = os.path.join(VW, "sfm", "recon", "sfm_vm.json")
    subprocess.run([CONV, "-i", os.path.join(VW, "sfm", "recon", "sfm_data.bin"), "-o", js, "-V", "-I", "-E"],
                   check=True, stdout=subprocess.DEVNULL)
    d = json.load(open(js))
    os.remove(js)
    frames = {r["file"]: r for r in csv.DictReader(open(os.path.join(VW, "frames.csv")))}
    tcol, rcol = stream_times(EX, off, "color")
    tby = {int(r["idx"]): t for r, t in zip(rcol, tcol)}
    view = {v["value"]["ptr_wrapper"]["data"]["id_view"]: v["value"]["ptr_wrapper"]["data"] for v in d["views"]}
    pose = {e["key"]: (np.array(e["value"]["rotation"]), np.array(e["value"]["center"])) for e in d["extrinsics"]}
    cams = sorted((tby[int(frames[v["filename"]]["color_idx"])], int(frames[v["filename"]]["color_idx"]),
                   pose[v["id_pose"]]) for v in view.values() if v["id_pose"] in pose)
    t = np.array([c[0] for c in cams], np.int64)
    R = np.array([c[2][0] for c in cams])
    C = np.array([c[2][1] for c in cams])
    k5 = np.ones(5) / 5
    Cs = np.column_stack([np.convolve(np.pad(C[:, j], 2, mode="edge"), k5, mode="valid") for j in range(3)])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(Cs, axis=0), axis=1))])
    return {"t": t, "idx": np.array([c[1] for c in cams]), "R": R, "C": C, "Cs": Cs, "s": s}


def _rotvec(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    a = np.arccos(c)
    if a < 1e-9:
        return np.zeros(3)
    return a * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(a))


def _expmap(v):
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) / th
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def pose_at(cam, t):
    """World->camera rotation R and centre C at common time t: slerp / linear between registered frames,
    clamped to the first/last frame outside the registered span (the robot stands still there)."""
    if t <= cam["t"][0]:
        return cam["R"][0], cam["C"][0]
    if t >= cam["t"][-1]:
        return cam["R"][-1], cam["C"][-1]
    j = int(np.searchsorted(cam["t"], t)) - 1
    a = (t - cam["t"][j]) / (cam["t"][j + 1] - cam["t"][j])
    Ra, Rb = cam["R"][j].T, cam["R"][j + 1].T                    # camera->world
    Rw = Ra @ _expmap(_rotvec(Ra.T @ Rb) * a)
    return Rw.T, (1 - a) * cam["C"][j] + a * cam["C"][j + 1]


def line_fit(P):
    mu = P.mean(0)
    _, sv, Vt = np.linalg.svd(P - mu)
    return mu, Vt[0], np.sqrt(np.sum(sv[1:] ** 2) / len(P))


def bend_split(Cs):
    n = len(Cs)
    best = None
    for k in range(int(0.2 * n), int(0.8 * n)):
        _, _, r1 = line_fit(Cs[:k])
        _, _, r2 = line_fit(Cs[k:])
        cost = r1 ** 2 * k + r2 ** 2 * (n - k)
        if best is None or cost < best[0]:
            best = (cost, k)
    return best[1]


if __name__ == "__main__":
    from pipe_geometry import fit_cylinder  # noqa: E402
    from ply_io import read_ply_vertices  # noqa: E402

    EX, VW, RES = sys.argv[1:4]
    off = load_offsets(os.path.join(RES, "fusion", "time_offsets.json"))
    cam = load_cameras(EX, VW, off)
    Cs, s = cam["Cs"], cam["s"]
    k = bend_split(Cs)
    m1, d1, r1 = line_fit(Cs[:k])
    m2, d2, r2 = line_fit(Cs[k:])
    d1 = d1 if d1 @ (Cs[k - 1] - Cs[0]) > 0 else -d1
    d2 = d2 if d2 @ (Cs[-1] - Cs[k]) > 0 else -d2
    bend_deg = np.degrees(np.arccos(np.clip(d1 @ d2, -1, 1)))
    print(f"camera path splits at camera {k} (colour frame {cam['idx'][k]}, path {s[k]:.2f} of {s[-1]:.2f} units); "
          f"legs straight to rms {r1:.3f} / {r2:.3f} units; angle between the legs {bend_deg:.1f} deg")
    V = read_ply_vertices(os.path.join(VW, "mvs", "scene_dense.ply"))
    X = np.column_stack([V["x"], V["y"], V["z"]]).astype(float)
    rng = np.random.default_rng(0)
    legs = []
    for name, mu, d, lo, hi in (("leg1", m1, d1, Cs[0], Cs[k]), ("leg2", m2, d2, Cs[k], Cs[int(0.85 * len(Cs))])):
        a, b = (lo - mu) @ d, (hi - mu) @ d
        q = X - mu
        sx = q @ d
        rx = np.linalg.norm(q - np.outer(sx, d), axis=1)
        margin = 3.0                            # model units (~1.3 pipe radii) away from the bend
        sel = (sx > a + (0 if name == "leg1" else margin)) & (sx < b - (margin if name == "leg1" else 0))
        sel &= rx < 2.5 * np.median(rx[sel])
        idx = np.flatnonzero(sel)
        idx = idx if len(idx) < 60000 else rng.choice(idx, 60000, replace=False)
        f = fit_cylinder(X[idx], d0=d, max_points=60000)
        ax = f["dir"] if f["dir"] @ d > 0 else -f["dir"]
        legs.append({"name": name, "dir": ax.tolist(), "point": f["point"].tolist(), "radius_units": f["radius"],
                     "rms_units": f["rms"], "points": int(sel.sum()), "path_dir": d.tolist()})
        print(f"{name}: {sel.sum():,} dense points, radius {f['radius']:.3f} units, rms {f['rms'] / f['radius']:.1%} of R, "
              f"axis vs camera-path direction {np.degrees(np.arccos(np.clip(ax @ d, -1, 1))):.2f} deg")
    a1, a2 = np.array(legs[0]["dir"]), np.array(legs[1]["dir"])
    print(f"bend between the fitted pipe axes: {np.degrees(np.arccos(np.clip(a1 @ a2, -1, 1))):.1f} deg")
    # where the two axes come closest: the bend point
    p1, p2 = np.array(legs[0]["point"]), np.array(legs[1]["point"])
    w0 = p1 - p2
    A = np.array([[a1 @ a1, -a1 @ a2], [a1 @ a2, -a2 @ a2]])
    u, v = np.linalg.solve(A, [-(w0 @ a1), -(w0 @ a2)])
    q1, q2 = p1 + u * a1, p2 + v * a2
    bend_pt = (q1 + q2) / 2
    s_bend = float(np.interp(0, [0, 1], [0, 0]))
    j = int(np.argmin(np.linalg.norm(Cs - bend_pt, axis=1)))
    print(f"axes pass each other at {np.linalg.norm(q1 - q2):.3f} units; bend point nearest camera {j} (path {s[j]:.2f} units)")
    json.dump({"split_camera": int(k), "split_colour_idx": int(cam["idx"][k]), "path_units": float(s[-1]),
               "bend_point": bend_pt.tolist(), "bend_point_path_units": float(s[j]), "axes_gap_units": float(np.linalg.norm(q1 - q2)),
               "legs": legs}, open(os.path.join(RES, "fusion", "vision_legs.json"), "w"), indent=1)
