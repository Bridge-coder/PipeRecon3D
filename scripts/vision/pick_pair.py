"""Rank candidate initial image pairs for OpenMVG's incremental SfM. No ground truth is used.

Usage: pick_pair.py MATCHES_DIR OUT_TXT [MIN_GAP] [--views FIRST LAST] [--loose]
       --views: only pairs with both images inside this range of view ids (to start a reconstruction in a
                stretch of the video that an earlier reconstruction did not reach)
       --loose: accept pairs with 40 matches (instead of 60) and list up to 8 candidates at least 10 views
                apart (instead of 6 at least 25 apart)

OpenMVG starts from two images and needs matches whose two viewing rays differ by more than 2 degrees.
A camera driving straight down a pipe has few of those, and OpenMVG's own choice (largest median angle)
sometimes lands on the flat end cap and fails. This script takes every pair with 60 or more verified
matches that is at least MIN_GAP frames apart (default 6), computes the two-view geometry with OpenCV
using the intrinsics in MATCHES_DIR/sfm_data.json, drops pairs whose recovered motion is not mostly along
the viewing direction, and ranks the rest by the number of matches above 2 degrees.
OUT_TXT gets one pair per line, best first, from different parts of the video:
  fileA fileB matches_above_2deg median_angle_deg inliers
"""
import json
import os
import struct
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
args = sys.argv[1:]
view_range = None
if "--views" in args:                     # only pairs with both views inside this range of view ids
    k = args.index("--views")
    view_range = (int(args[k + 1]), int(args[k + 2]))
    del args[k:k + 3]
min_matches, spread = 60, 25              # --loose: for a stretch where the usual thresholds leave too few candidates
if "--loose" in args:
    args.remove("--loose")
    min_matches, spread = 40, 10
matches, out_txt = args[0], args[1]
min_gap = int(args[2]) if len(args) > 2 else 6

sfm = json.load(open(os.path.join(matches, "sfm_data.json"), encoding="utf-8"))
names = {v["value"]["ptr_wrapper"]["data"]["id_view"]: v["value"]["ptr_wrapper"]["data"]["filename"] for v in sfm["views"]}
intr = sfm["intrinsics"][0]["value"]["ptr_wrapper"]["data"]
f, (cx, cy) = intr["focal_length"], intr["principal_point"]
K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
dist = np.array(list(intr.get("disto_k1", [0.0])) + [0.0, 0.0, 0.0])      # OpenCV order: k1 k2 p1 p2
print(f"intrinsics: focal {f:g}, principal point ({cx:g}, {cy:g}), k1 {dist[0]:g}")

data = open(os.path.join(matches, "matches.f.bin"), "rb").read()
(n,) = struct.unpack_from("<Q", data, 1)
off, pairs = 9, {}
for _ in range(n):
    i, j, m = struct.unpack_from("<IIQ", data, off)
    off += 16
    pairs[(i, j)] = np.frombuffer(data, dtype="<u4", count=2 * m, offset=off).reshape(m, 2)
    off += 8 * m

feats = {}


def rays(i, idx):
    """Unit-plane coordinates (distortion removed) of the given features of view i."""
    if i not in feats:
        feats[i] = np.loadtxt(os.path.join(matches, os.path.splitext(names[i])[0] + ".feat"), usecols=(0, 1), ndmin=2)
    return cv2.undistortPoints(feats[i][idx].reshape(-1, 1, 2), K, dist).reshape(-1, 2)


cand = [k for k, v in pairs.items() if k[1] - k[0] >= min_gap and len(v) >= min_matches
        and (view_range is None or (view_range[0] <= k[0] and k[1] <= view_range[1]))]
print(f"{len(pairs)} matched pairs, {len(cand)} of them at least {min_gap} frames apart with {min_matches} or more matches"
      + (f" inside views {view_range[0]}..{view_range[1]}" if view_range else ""))
ranked = []
for i, j in cand:
    a, b = rays(i, pairs[(i, j)][:, 0]), rays(j, pairs[(i, j)][:, 1])
    e, mask = cv2.findEssentialMat(a, b, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=1.5 / f)
    if e is None or e.shape != (3, 3):
        continue
    inl = mask.ravel().astype(bool)
    _, r, t, mask2 = cv2.recoverPose(e, a[inl], b[inl], np.eye(3))
    good = mask2.ravel() > 0
    a, b = a[inl][good], b[inl][good]
    if len(a) < 30:
        continue
    r1 = np.c_[a, np.ones(len(a))].T
    r2 = r.T @ np.c_[b, np.ones(len(b))].T
    ang = np.degrees(np.arccos(np.clip((r1 * r2).sum(0) / (np.linalg.norm(r1, axis=0) * np.linalg.norm(r2, axis=0)), -1, 1)))
    # a pipe robot drives along its viewing direction; a recovered motion that is mostly sideways comes
    # from a degenerate two-view solution (e.g. only the flat end cap matched) and is not a usable start
    if abs(float(t[2, 0])) < 0.8:
        continue
    ranked.append((int((ang > 2).sum()), float(np.median(ang)), len(a), i, j))
ranked.sort(reverse=True)
print(f"{len(ranked)} pairs with a forward two-view solution")

chosen = []
for row in ranked:                       # keep candidates from different parts of the video
    if all(abs(row[3] - c[3]) > spread for c in chosen):
        chosen.append(row)
    if len(chosen) == (6 if spread == 25 else 8):
        break
with open(out_txt, "w", encoding="utf-8") as out:
    for n2, med, n_in, i, j in chosen:
        out.write(f"{names[i]} {names[j]} {n2} {med:.2f} {n_in}\n")
        print(f"   {names[i]} + {names[j]}: {n2} matches above 2 deg, median angle {med:.2f} deg, {n_in} inliers")
if not chosen:
    sys.exit("no usable initial pair found")
print(f"{len(chosen)} candidate pair(s) written to {out_txt}")
