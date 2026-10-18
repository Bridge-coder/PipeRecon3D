"""Step 2 of run_vision.py: pick SfM frames from the colour images written by extract_bag.py.

Usage: prepare_frames.py EXTRACT_DIR OUT_DIR [--every N] [--first IDX] [--last IDX] [--clahe]

Keeps colour frames first..last, one out of N, as OUT_DIR/frame_<idx>.png (idx = colour frame index of
the bag, so names stay comparable between runs). --clahe: contrast-limited adaptive histogram
equalisation of the brightness channel, as for the Sheffield videos.
OUT_DIR/../frames.csv lists file name, colour frame index, log time and header stamp (ns).
"""
import csv
import os
import sys

import cv2

args = sys.argv[1:]


def take(flag, default):
    if flag in args:
        i = args.index(flag)
        v = int(args[i + 1])
        del args[i:i + 2]
        return v
    return default


every, first, last = take("--every", 5), take("--first", 0), take("--last", 10 ** 9)
use_clahe = "--clahe" in args
if use_clahe:
    args.remove("--clahe")
extract, out_dir = args
work = os.path.dirname(os.path.abspath(out_dir))
os.makedirs(out_dir, exist_ok=True)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if use_clahe else None

with open(os.path.join(extract, "images", "color_index.csv")) as f:
    index = list(csv.DictReader(f))
rows = []
for r in index:
    i = int(r["idx"])
    if not (first <= i <= last) or (i - first) % every:
        continue
    img = cv2.imread(os.path.join(extract, "images", "color", f"{i:06d}.png"))
    if clahe is not None:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[..., 0] = clahe.apply(lab[..., 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    name = f"frame_{i:06d}.png"
    cv2.imwrite(os.path.join(out_dir, name), img)
    rows.append((name, i, r["log_time_ns"], r["stamp_ns"]))

with open(os.path.join(work, "frames.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f)
    wr.writerow(["file", "color_idx", "log_time_ns", "stamp_ns"])
    wr.writerows(rows)
print(f"kept {len(rows)} of {len(index)} colour frames (every {every}, {first}..{min(last, len(index) - 1)})"
      f"{'; CLAHE on' if use_clahe else ''} -> {out_dir}")
