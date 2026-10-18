"""Camera calibration from a checkerboard video, in the camera model the reconstruction uses.

usage: python -I calibrate_checkerboard.py VIDEO OUT_JSON [--pattern COLS ROWS] [--crop X0 Y0 X1 Y1] [--resize W H]
                                                [--every N] [--max M]
       --pattern  inner corners per row and column (default 9 6)
       --crop     picture area to keep, in pixels of the video (x1, y1 exclusive), e.g. to cut black side bars and
                  burnt-in text; use the same crop for the inspection video (run_pipeline.py --crop)
       --resize   scale the (cropped) picture to W x H first: when the calibration video was digitised at another size
                  than the inspection video (e.g. the same PAL picture scaled x1.25 into 1280x720)
       --every    look at every Nth frame (default 5); --max: use at most M boards, spread over the video (default 60)

OpenMVG's camera "pinhole radial 1" has one focal length and x_d = x_u (1 + k1 r^2) in normalised coordinates, which is
OpenCV's model with fx = fy, k2 = k3 = p1 = p2 = 0. Analogue cameras digitised to a different aspect ratio have
non-square pixels (fx != fy), so the calibration is done in two passes:
  1. free fx, fy, cx, cy, k1 (k2, k3, tangential fixed at 0): measures the pixel aspect  s = fy / fx
  2. if s differs from 1 by more than 0.5 %: the frames are to be stretched horizontally by s (run_pipeline.py
     --stretch-x s); in stretched pixels fx' = fx * s = fy, cx' = cx * s, and k1 is unchanged (normalised coordinates
     do not change). Then fx = fy holds and the model above applies.
OUT_JSON: focal, cx, cy, k1 in the pixels of the cropped (and stretched) frames, the crop and stretch to use, RMS
reprojection error, number of boards. Print-out ends with the run_pipeline.py options.
"""
import json
import sys

import cv2
import numpy as np

args = sys.argv[1:]


def opt(flag, n, default):
    if flag in args:
        i = args.index(flag)
        v = args[i + 1:i + 1 + n]
        del args[i:i + 1 + n]
        return v
    return default


pattern = tuple(int(x) for x in opt("--pattern", 2, ["9", "6"]))
crop = [int(x) for x in opt("--crop", 4, [])] or None
resize = tuple(int(x) for x in opt("--resize", 2, [])) or None
every = int(opt("--every", 1, ["5"])[0])
max_boards = int(opt("--max", 1, ["60"])[0])
video, out_json = args

cap = cv2.VideoCapture(video)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
obj = np.zeros((pattern[0] * pattern[1], 3), np.float32)
obj[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
found, size, idx = [], None, 0
crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if idx % every == 0:
        if crop:
            frame = frame[crop[1]:crop[3], crop[0]:crop[2]]
        if resize:
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = g.shape[::-1]
        ok, c = cv2.findChessboardCorners(g, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            c = cv2.cornerSubPix(g, c, (5, 5), (-1, -1), crit)
            found.append((idx, c))
    idx += 1
print(f"{idx} frames, board found in {len(found)} of {len(range(0, idx, every))} checked; picture {size[0]}x{size[1]}")
if len(found) < 8:
    sys.exit("too few boards for a calibration")
pick = np.linspace(0, len(found) - 1, min(max_boards, len(found))).round().astype(int)
img_pts = [found[k][1] for k in pick]
obj_pts = [obj] * len(img_pts)
flags = cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
rms, K, D, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None, flags=flags)
fx, fy, cx, cy, k1 = K[0, 0], K[1, 1], K[0, 2], K[1, 2], D.ravel()[0]
s = fy / fx
print(f"pass 1 (free aspect): fx {fx:.1f}  fy {fy:.1f}  (pixel aspect fy/fx = {s:.4f})  cx {cx:.1f}  cy {cy:.1f}  k1 {k1:+.4f}  "
      f"rms {rms:.3f} px  ({len(img_pts)} boards)")
stretch = 1.0
if abs(s - 1) > 0.005:
    stretch = s
    # pass 2: re-fit in stretched pixels with one focal length, to confirm the stretched model fits as well
    img2 = [p * np.array([[[s, 1.0]]], np.float32) for p in img_pts]
    size2 = (int(round(size[0] * s)), size[1])
    rms2, K2, D2, _, _ = cv2.calibrateCamera(obj_pts, img2, size2, None, None, flags=flags | cv2.CALIB_FIX_ASPECT_RATIO)
    f, cx, cy, k1 = K2[0, 0], K2[0, 2], K2[1, 2], D2.ravel()[0]
    print(f"pass 2 (frames stretched x{s:.4f} to {size2[0]}x{size2[1]}, square pixels): f {f:.1f}  cx {cx:.1f}  cy {cy:.1f}  "
          f"k1 {k1:+.4f}  rms {rms2:.3f} px")
    rms, size = rms2, size2
else:
    f = (fx + fy) / 2
# OpenCV puts pixel centres at integers, like OpenMVG: no half-pixel shift needed
out = {"focal": float(f), "cx": float(cx), "cy": float(cy), "k1": float(k1), "stretch_x": float(stretch),
       "calibration_crop": crop, "calibration_resize": resize, "frame_size": list(size), "rms_px": float(rms), "boards": len(img_pts), "pattern": list(pattern),
       "video": video}
json.dump(out, open(out_json, "w"), indent=1)
opts = f"--plain --focal {f:.1f} --cx {cx:.1f} --cy {cy:.1f} --k1 {k1:.4f}"
if stretch != 1.0:
    opts += f" --stretch-x {stretch:.4f}"
print("run_pipeline.py options:", opts)
print("saved", out_json)
