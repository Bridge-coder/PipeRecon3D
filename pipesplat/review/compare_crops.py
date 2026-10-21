"""docs/10 figures: the same held-out frames from several runs side by side (video | run 1 | run 2 ...), whole frame
on top and a 2x enlarged crop below (default: the left wall near the image edge, where the wall is closest).
Reads each run's renders/<tag>_<frame>.jpg triptychs (video | B | model).

usage: python -I pipesplat/review/compare_crops.py OUT.jpg DATA_DIR "label=run_dir_or_B" ... [--frames held_001012 ...]
           [--crop x0 y0 x1 y1]
label=B takes the B panel of the first run; any other value is a run folder name inside DATA_DIR.
"""
import os
import sys

import cv2
import numpy as np

args = sys.argv[1:]
out, root = args[0], args[1]
frames, crop, runs = None, None, []
i = 2
while i < len(args):
    if args[i] == "--frames":
        frames = []
        i += 1
        while i < len(args) and not args[i].startswith("--"):
            frames.append(args[i])
            i += 1
    elif args[i] == "--crop":
        crop = [int(v) for v in args[i + 1:i + 5]]
        i += 5
    else:
        runs.append(args[i].split("=", 1))
        i += 1
first = next(r for _, r in runs if r != "B")
names = sorted(os.listdir(os.path.join(root, first, "renders")))
frames = frames or [n[:-4] for n in names if n.startswith("held")][::3][:4]


def panels(run, frame):
    img = cv2.imdecode(np.fromfile(os.path.join(root, run, "renders", frame + ".jpg"), np.uint8), cv2.IMREAD_COLOR)
    w = img.shape[1] // 3
    return img[:, :w], img[:, w:2 * w], img[:, 2 * w:]


def label(img, text):
    img = img.copy()
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


rows = []
for fr in frames:
    video, B, _ = panels(first, fr)
    h, w = video.shape[:2]
    x0, y0, x1, y1 = crop or (0, int(h * 0.25), int(w * 0.33), int(h * 0.75))
    cols = [("video", video)]
    for lab, run in runs:
        cols.append((lab, B if run == "B" else panels(run, fr)[2]))
    top = np.hstack([label(c, f"{lab}  {fr}" if lab == "video" else lab) for lab, c in cols])
    zoom = np.hstack([cv2.resize(c[y0:y1, x0:x1], ((x1 - x0) * 2, (y1 - y0) * 2), interpolation=cv2.INTER_NEAREST)
                      for _, c in cols])
    if zoom.shape[1] != top.shape[1]:
        zoom = cv2.resize(zoom, (top.shape[1], int(zoom.shape[0] * top.shape[1] / zoom.shape[1])), interpolation=cv2.INTER_NEAREST)
    rows += [top, zoom]
cv2.imencode(".jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(out)
print(f"wrote {out}: {len(frames)} frames x {len(runs) + 1} columns")
