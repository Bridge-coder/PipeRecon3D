"""F4, independent scale of the vision model from the front AprilTag (side 100 mm).

The four corners of tag 91 are detected in every SfM-registered colour frame that sees it and triangulated
(linear multi-view DLT, then reprojection-error refinement) with the SfM poses and the recorded K. The side
length in model units gives metres-per-unit from an object that lies ACROSS the pipe direction - the same
direction as the pipe radius - without the encoders or the LiDAR. Compared with the encoder scale (travel
ALONG the pipe), it shows whether the vision model is stretched differently along and across the pipe.

usage: python -I f4_tag_scale.py EXTRACT_DIR VISION_WORK RESULTS_DIR
"""
import csv
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from timebase import load_offsets  # noqa: E402
from vision_model import load_cameras  # noqa: E402

EX, VW, RES = sys.argv[1:4]
OUT = os.path.join(RES, "fusion")
off = load_offsets(os.path.join(OUT, "time_offsets.json"))
cam = load_cameras(EX, VW, off)
ci = json.load(open(os.path.join(EX, "bag_info.json")))["camera_info"]["color"]
f = (ci["k"][0] + ci["k"][4]) / 2          # the SfM used this single focal length
K = np.array([[f, 0, ci["k"][2]], [0, f, ci["k"][5]], [0, 0, 1.0]])
det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
                              cv2.aruco.DetectorParameters())
obs = []          # (camera index, 4x2 corners)
for k, idx in enumerate(cam["idx"]):
    img = cv2.imread(os.path.join(EX, "images", "color", f"{idx:06d}.png"), cv2.IMREAD_GRAYSCALE)
    corners, ids, _ = det.detectMarkers(img)
    if ids is None or 91 not in ids.ravel():
        continue
    c = corners[list(ids.ravel()).index(91)].reshape(4, 2).astype(float)
    cv2.cornerSubPix(img, c.astype(np.float32).reshape(-1, 1, 2), (5, 5), (-1, -1),
                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
    obs.append((k, c))
print(f"tag 91 seen in {len(obs)} of {len(cam['idx'])} registered frames")


def P_of(k):
    R, C = cam["R"][k], cam["C"][k]
    return K @ np.hstack([R, (-R @ C)[:, None]])


def triangulate(views):
    A = []
    for k, uv in views:
        P = P_of(k)
        A.append(uv[0] * P[2] - P[0])
        A.append(uv[1] * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1]
    X = X[:3] / X[3]
    for _ in range(10):                      # Gauss-Newton on the reprojection error
        J, r = [], []
        for k, uv in views:
            P = P_of(k)
            h = P @ np.append(X, 1)
            r.extend(uv - h[:2] / h[2])
            for j in range(3):
                d = np.zeros(3)
                d[j] = 1e-6
                h2 = P @ np.append(X + d, 1)
                J.append(None)
            Jk = np.zeros((2, 3))
            for j in range(3):
                d = np.zeros(3)
                d[j] = 1e-6
                h2 = P @ np.append(X + d, 1)
                Jk[:, j] = ((h2[:2] / h2[2]) - (h[:2] / h[2])) / 1e-6
            J = [j_ for j_ in J if j_ is not None]
            J.append(Jk)
        Jm = np.vstack(J)
        step, *_ = np.linalg.lstsq(Jm, np.array(r), rcond=None)
        X = X + step
    res = []
    for k, uv in views:
        h = P_of(k) @ np.append(X, 1)
        res.append(np.linalg.norm(uv - h[:2] / h[2]))
    return X, np.array(res)


Xc, rep = [], []
for j in range(4):
    X, r = triangulate([(k, c[j]) for k, c in obs])
    Xc.append(X)
    rep.append(r)
Xc = np.array(Xc)
sides = np.linalg.norm(np.roll(Xc, -1, axis=0) - Xc, axis=1)
diag = np.linalg.norm(Xc[0] - Xc[2]), np.linalg.norm(Xc[1] - Xc[3])
side = sides.mean()
lam_tag = 0.100 / side
lam_enc = 1.730 / cam["s"][-1]
print(f"corner reprojection rms {np.sqrt(np.mean(np.concatenate(rep) ** 2)):.2f} px; sides {np.round(sides, 4)} units "
      f"(spread {100 * (sides.max() - sides.min()) / side:.1f}%), diagonals/side {np.mean(diag) / side:.3f} (square: 1.414)")
print(f"scale from the tag: {lam_tag * 1000:.3f} mm per unit; from the encoders (path length): {lam_enc * 1000:.3f} mm per unit; "
      f"tag / encoder = {lam_tag / lam_enc:.4f}")
vleg = json.load(open(os.path.join(OUT, "vision_legs.json")))
legs = vleg["legs"]
# scale consistency across the bend: the same tag triangulated from leg-1 frames only and leg-2 frames only
split = vleg["split_camera"]
per_leg = {}
for name, subset in (("leg-1 frames", [o for o in obs if o[0] < split]), ("leg-2 frames", [o for o in obs if o[0] >= split])):
    if len(subset) < 5:
        continue
    Xs = np.array([triangulate([(k, c[j]) for k, c in subset])[0] for j in range(4)])
    sd = np.linalg.norm(np.roll(Xs, -1, axis=0) - Xs, axis=1)
    per_leg[name] = {"frames": len(subset), "side_units": float(sd.mean()), "mm_per_unit": 100.0 / sd.mean(),
                     "shift_of_corners_units": float(np.linalg.norm(Xs - Xc, axis=1).mean())}
    print(f"  tag from {name} only ({len(subset)} frames, colour {cam['idx'][subset[0][0]]}-{cam['idx'][subset[-1][0]]}): "
          f"side {sd.mean():.4f} units -> {100.0 / sd.mean():.3f} mm per unit; corners {np.linalg.norm(Xs - Xc, axis=1).mean():.3f} units from the all-frame result")
for L in legs:
    print(f"  {L['name']}: radius {L['radius_units']:.3f} units -> D {2 * L['radius_units'] * lam_tag * 1000:.1f} mm with the tag scale, "
          f"{2 * L['radius_units'] * lam_enc * 1000:.1f} mm with the encoder scale")
# distance of the tag from the camera path end, and tag position along the leg-2 axis
json.dump({"frames": len(obs), "sides_units": sides.tolist(), "mm_per_unit_tag": lam_tag * 1000,
           "mm_per_unit_encoder": lam_enc * 1000, "ratio_tag_over_encoder": lam_tag / lam_enc,
           "reproj_rms_px": float(np.sqrt(np.mean(np.concatenate(rep) ** 2))), "corners_units": Xc.tolist(),
           "per_leg": per_leg},
          open(os.path.join(OUT, "f4_tag_scale.json"), "w"), indent=1)
