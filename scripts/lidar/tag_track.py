"""Axial reference from the rear camera: distance to the fixed AprilTag behind the robot.

In sequence 172011 the rear-facing camera (/camera/usbcam, 1280x720, recorded K and plumb-bob distortion)
sees an AprilTag 36h11, id 80, on a board outside the pipe entrance for the whole run. The tag does not
move, so the camera-to-tag distance grows exactly by the distance the robot travels.

The printed size of the tag is not known, so the pose is solved for a tag of side 1 and every distance is
then known only up to one factor L (the true side in metres). compare_odometry() fits L and an offset so
that the tag distance matches the wheel-encoder distance; what is left (the residual along the run) shows
wheel slip or tag-detection noise, independent of L.

The front colour camera likewise sees AprilTag id 91 on a board beyond the far pipe end from about colour
frame 1600 to 3200; there the distance shrinks as the robot advances (fitted L comes out negative).

usage: python -I tag_track.py EXTRACT_DIR OUT_DIR [--camera usbcam|color] [--tag ID]
writes OUT_DIR/tag_track_<camera>.csv (image idx, log time, stamp, distance and position for a tag of
side 1, side length in pixels) and OUT_DIR/tag_vs_encoder_<camera>.png
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

args = sys.argv[1:]
CAM, TAG_ID = "usbcam", 80
if "--camera" in args:
    i = args.index("--camera")
    CAM = args[i + 1]
    del args[i:i + 2]
    TAG_ID = 91 if CAM == "color" else 80
if "--tag" in args:
    i = args.index("--tag")
    TAG_ID = int(args[i + 1])
    del args[i:i + 2]
EX, OUT = args[:2]
os.makedirs(OUT, exist_ok=True)

with open(os.path.join(EX, "bag_info.json")) as f:
    ci = json.load(f)["camera_info"][CAM]
K = np.array(ci["k"], float).reshape(3, 3)
dist = np.array(ci["d"], float)
det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
                              cv2.aruco.DetectorParameters())
obj = np.array([[-0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, -0.5, 0], [-0.5, -0.5, 0]], np.float32)  # side 1

with open(os.path.join(EX, "images", f"{CAM}_index.csv")) as f:
    index = list(csv.DictReader(f))
rows = []
for r in index:
    i = int(r["idx"])
    img = cv2.imread(os.path.join(EX, "images", CAM, f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
    corners, ids, _ = det.detectMarkers(img)
    if ids is None or TAG_ID not in ids.ravel():
        continue
    c = corners[list(ids.ravel()).index(TAG_ID)].reshape(4, 2).astype(np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj, c, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        continue
    side_px = np.mean(np.linalg.norm(np.roll(c, -1, axis=0) - c, axis=1))
    t = tvec.ravel()
    rows.append({"idx": i, "log_time_ns": int(r["log_time_ns"]), "stamp_ns": int(r["stamp_ns"]),
                 "dist_unit": float(np.linalg.norm(t)), "x_unit": t[0], "y_unit": t[1], "z_unit": t[2],
                 "side_px": side_px})
with open(os.path.join(OUT, f"tag_track_{CAM}.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"tag {TAG_ID} found in {len(rows)} of {len(index)} {CAM} frames")

# compare with the wheel encoders: s_enc(t) = L * d_unit(t) + c
enc = list(csv.DictReader(open(os.path.join(EX, "encoders.csv"))))
te = np.array([int(r["log_time_ns"]) for r in enc], np.int64)
se = np.array([(float(r["v6"]) - float(r["v7"])) / 2 for r in enc])
tt = np.array([r["log_time_ns"] for r in rows], np.int64)
du = np.array([r["dist_unit"] for r in rows])
s_at = np.interp(tt - te[0], te - te[0], se - se[0])
A = np.column_stack([du, np.ones_like(du)])
(L, c), *_ = np.linalg.lstsq(A, s_at, rcond=None)
res = s_at - (L * du + c)
moving = s_at > 0.005
print(f"fitted tag side |L| = {abs(L) * 1000:.1f} mm; encoder minus tag distance: rms {1000 * np.sqrt(np.mean(res ** 2)):.1f} mm, "
      f"range {1000 * res.min():.1f}..{1000 * res.max():.1f} mm over {s_at.max():.3f} m of encoder travel")
tsec = (tt - te[0]) / 1e9
fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
ax[0].plot((te - te[0]) / 1e9, se - se[0], label="wheel encoders (v6 - v7)/2")
ax[0].plot(tsec, L * du + c, ".", ms=2, label=f"{CAM} tag {TAG_ID} distance x L (|L| = {abs(L) * 1000:.0f} mm fitted)")
ax[0].set_ylabel("travel (m)"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(tsec, 1000 * res, ".", ms=2)
ax[1].set_ylabel("encoder - tag (mm)"); ax[1].set_xlabel("time since start of bag (s)"); ax[1].grid(alpha=.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, f"tag_vs_encoder_{CAM}.png"), dpi=90)
