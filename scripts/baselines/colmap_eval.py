"""Evaluate a COLMAP reconstruction with exactly the metrics of vision/evaluate.py (sparse and dense).

usage: python -I colmap_eval.py SPARSE_TXT_DIR FRAMES_CSV COUNTER_CSV OUT_PREFIX [--dense FUSED_PLY]
       SPARSE_TXT_DIR  one model written by run_colmap.py (cameras.txt, images.txt, points3D.txt)

evaluate.py reads OpenMVG models through load_sfm(); here load_sfm is replaced by a reader of COLMAP's text model
that returns the same tuple, so the pipe fit, the scale from the 300 mm bore, the travel against the on-screen
counter and the dense wall statistics are computed by the same code for both pipelines.
Conventions: COLMAP stores world->camera rotation and translation (centre = -R^T t), like OpenMVG's rotation and
centre; COLMAP's principal point counts from the corner of the first pixel, OpenMVG's from its centre (-0.5).
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vision"))
import evaluate as ev  # noqa: E402


def qvec_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def colmap_loader(model_dir, frames_csv):
    def load_sfm(_converter, _sfm_bin):
        cams = {}
        for ln in open(os.path.join(model_dir, "cameras.txt"), encoding="utf-8"):
            if ln.strip() and not ln.startswith("#"):
                p = ln.split()
                cams[int(p[0])] = (p[1], [float(x) for x in p[4:]])
        model, prm = next(iter(cams.values()))
        if model not in ("SIMPLE_RADIAL", "SIMPLE_PINHOLE", "RADIAL"):    # f cx cy [k1 [k2]]
            sys.exit(f"camera model {model} not supported (run_colmap.py uses SIMPLE_RADIAL)")
        intr = {"focal_length": prm[0], "principal_point": [prm[1] - 0.5, prm[2] - 0.5],
                "disto_k1": [prm[3] if len(prm) > 3 else 0.0]}
        files = [r["file"] for r in csv.DictReader(open(frames_csv, encoding="utf-8"))]
        vid_of = {f: k for k, f in enumerate(files)}
        views = {k: (f, -1) for k, f in enumerate(files)}         # unregistered: pose id not in poses
        poses, img_vid = {}, {}
        lines = [ln for ln in open(os.path.join(model_dir, "images.txt"), encoding="utf-8")
                 if not ln.startswith("#")]
        for a in range(0, len(lines) - 1, 2):
            p = lines[a].split()
            if not p:
                continue
            iid, q, t, name = int(p[0]), [float(x) for x in p[1:5]], np.array([float(x) for x in p[5:8]]), p[9]
            R = qvec_to_R(q)
            v = vid_of[name]
            views[v] = (name, v)
            poses[v] = -R.T @ t
            img_vid[iid] = v
        pts, obs = [], []
        for ln in open(os.path.join(model_dir, "points3D.txt"), encoding="utf-8"):
            if ln.strip() and not ln.startswith("#"):
                p = ln.split()
                pts.append([float(x) for x in p[1:4]])
                track = p[8:]
                obs.append(np.array(sorted({img_vid[int(track[k])] for k in range(0, len(track), 2)}), int))
        pts = np.array(pts, float).reshape(-1, 3)
        nobs = np.array([len(o) for o in obs], int)
        return views, f"COLMAP {model}", intr, poses, pts, nobs, obs
    return load_sfm


if __name__ == "__main__":
    args = sys.argv[1:]
    dense = None
    if "--dense" in args:
        i = args.index("--dense")
        dense = args[i + 1]
        del args[i:i + 2]
    model_dir, frames_csv, counter_csv, out_prefix = args
    ev.load_sfm = colmap_loader(model_dir, frames_csv)
    ev.cmd_sparse(None, None, frames_csv, counter_csv, out_prefix + "_sparse")
    if dense:
        ev.cmd_dense(None, None, frames_csv, counter_csv, dense, out_prefix + "_dense")
