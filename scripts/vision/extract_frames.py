"""Step 2: turn the inspection video into SfM-ready frames.

Usage: extract_frames.py VIDEO OUT_DIR [--every N] [--clahe] [--first A] [--last B] [--plain]
       --plain  any other video: frames are kept at their own size (no PAL stretch, no crop, no counter strip);
                steps 1, 2 and the OSD strip below apply only to the 720x576 Sheffield Minicam videos
       --crop X0 Y0 X1 Y1   (with --plain) keep only this part of the frame, e.g. to cut burnt-in text
       --stretch-x S        (with --plain) then stretch the width by S (non-square pixels, calibrate_checkerboard.py)

For every Nth frame (default 12, about 2 frames per second at 25 fps):
  1. stretch 720x576 to 787x576  (analogue PAL pixels are not square: aspect 59:54)
  2. crop left 16, right 3, bottom 28 px -> 768x548  (black border and on-screen text)
  3. with --clahe: contrast-limited adaptive histogram equalisation of the brightness channel
  4. save as PNG in OUT_DIR
The untouched bottom strip (date / distance counter / time) of each kept frame goes to OUT_DIR/../osd;
it is only used to read the ground-truth travel distance, never for reconstruction.
OUT_DIR/../frames.csv lists file name, source frame index and time.
"""
import csv
import os
import sys

import cv2

STRETCH_W = 787       # 720 * 59/54, rounded
CROP_LEFT, CROP_RIGHT, CROP_BOTTOM = 16, 3, 28
OSD_ROWS = 28         # bottom rows of the original frame that hold the on-screen text

args = sys.argv[1:]
use_clahe = "--clahe" in args
plain = "--plain" in args
crop = [int(x) for x in args[args.index("--crop") + 1:args.index("--crop") + 5]] if "--crop" in args else None
stretch_x = float(args[args.index("--stretch-x") + 1]) if "--stretch-x" in args else 1.0
every = int(args[args.index("--every") + 1]) if "--every" in args else 12  # keep one frame out of this many
first = int(args[args.index("--first") + 1]) if "--first" in args else 0   # only source frames first..last
last = int(args[args.index("--last") + 1]) if "--last" in args else 10 ** 9
video, out_dir = args[0], args[1]
work = os.path.dirname(os.path.abspath(out_dir))
osd_dir = os.path.join(work, "osd")
os.makedirs(out_dir, exist_ok=True)
os.makedirs(osd_dir, exist_ok=True)

cap = cv2.VideoCapture(video)
if not cap.isOpened():
    sys.exit(f"cannot open {video}")
fps = cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if not plain and (w, h) != (720, 576):
    sys.exit(f"unexpected frame size {w}x{h}; the crop values assume 720x576")
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if use_clahe else None

rows, idx = [], 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if idx % every == 0 and first <= idx <= last:
        name = f"frame_{idx:06d}.png"
        if plain:
            out = frame[crop[1]:crop[3], crop[0]:crop[2]] if crop else frame.copy()
            if stretch_x != 1.0:
                out = cv2.resize(out, (int(round(out.shape[1] * stretch_x)), out.shape[0]), interpolation=cv2.INTER_CUBIC)
        else:
            square = cv2.resize(frame, (STRETCH_W, h), interpolation=cv2.INTER_CUBIC)
            out = square[: h - CROP_BOTTOM, CROP_LEFT: STRETCH_W - CROP_RIGHT]
        if clahe is not None:
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            lab[..., 0] = clahe.apply(lab[..., 0])
            out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        cv2.imwrite(os.path.join(out_dir, name), out)
        if not plain:
            cv2.imwrite(os.path.join(osd_dir, name), frame[h - OSD_ROWS:])
        rows.append((name, idx, f"{idx / fps:.2f}"))
    idx += 1
cap.release()

with open(os.path.join(work, "frames.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f)
    wr.writerow(["file", "source_frame", "time_s"])
    wr.writerows(rows)

print(f"read {idx} frames at {fps:.2f} fps; kept {len(rows)} (every {every}th){'; CLAHE on' if use_clahe else ''}")
print(f"output size {out.shape[1]}x{out.shape[0]} -> {out_dir}")
print(f"counter strips -> {osd_dir}")
