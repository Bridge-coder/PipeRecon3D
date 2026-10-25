"""docs/11: comparison video of models on frames none of them was trained on - the 44 in-sequence held-out frames and
the two 10 cm block stretches - side by side: video | v1.0 model (B) | other methods | ours, in travel order, every
panel labelled with its PSNR on that frame (ruler v2, the same numbers as the tables). Frames from dump_frames.py.
Also writes two stills: the block frame with the MEDIAN advantage of ours over the best other model (OUT_poster.jpg,
the typical case) and the one with the largest (OUT_extreme.jpg, labelled as such).

usage: python -I pipesplat/review/compare_video.py DATA OUT.mp4 "TITLE" "LABEL=RUN" ... [--fps 2] [--width 480]
       LABEL=B takes the v1.0 model from the first run's dump; the video frame comes first automatically
"""
import csv
import glob
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

V = "D:/pipe3d_gs/review/video"
G = "D:/pipe3d_gs/g0"
FONT = "C:/Windows/Fonts/msyh.ttc"
args = sys.argv[1:]
fps = float(args[args.index("--fps") + 1]) if "--fps" in args else 2.0
pw = int(args[args.index("--width") + 1]) if "--width" in args else 480
args = [x for i, x in enumerate(args) if x not in ("--fps", "--width") and (i == 0 or args[i - 1] not in ("--fps", "--width"))]
ds, out, title = args[0], args[1], args[2]
cols = [c.split("=", 1) for c in args[3:]]
first = next(r for l, r in cols if r != "B")


def psnr_table(run):
    p = os.path.join(G, ds, run, "per_frame_dump.csv")
    t = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        t[(r["tag"], int(r["frame"]))] = r
    return t


tabs = {r: psnr_table(r) for _, r in cols if r != "B"}
keys = sorted({(os.path.basename(f).split("_")[0], int(os.path.basename(f).split("_")[1]))
               for f in glob.glob(os.path.join(V, ds, first, "*_G.jpg"))}, key=lambda k: k[1])
f_big, f_small = ImageFont.truetype(FONT, 26), ImageFont.truetype(FONT, 20)


def panel(img, text, sub):
    h, w = img.shape[:2]
    ph = int(round(h * pw / w))
    im = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA)
    bar = np.full((64, pw, 3), 255, np.uint8)
    pil = Image.fromarray(np.vstack([bar, im[:, :, ::-1]]))
    d = ImageDraw.Draw(pil)
    d.text((10, 4), text, font=f_small, fill=(20, 30, 40))
    d.text((10, 32), sub, font=f_small, fill=(12, 118, 106) if sub.startswith("PSNR") else (90, 100, 110))
    return np.asarray(pil)[:, :, ::-1]


frames, score_gap = [], []
for tag, fr in keys:
    tiles, ps = [], []
    real = cv2.imread(os.path.join(V, ds, first, f"{tag}_{fr:06d}_real.jpg"))
    tiles.append(panel(real, "原视频", "（比较对象）"))
    for label, run in cols:
        if run == "B":
            img = cv2.imread(os.path.join(V, ds, first, f"{tag}_{fr:06d}_B.jpg"))
            v = float(tabs[first][(tag, fr)]["B_psnr"])
        else:
            img = cv2.imread(os.path.join(V, ds, run, f"{tag}_{fr:06d}_G.jpg"))
            v = float(tabs[run][(tag, fr)]["G_psnr"])
        ps.append(v)
        tiles.append(panel(img, label, f"PSNR {v:.1f} dB"))
    row = np.hstack(tiles)
    top = Image.fromarray(np.full((56, row.shape[1], 3), 255, np.uint8))
    d = ImageDraw.Draw(top)
    kind = "整段留出（训练时拿掉的 10 cm 管段）" if tag == "block" else "序列内留出帧（没参加训练）"
    d.text((12, 12), f"{title}   ·   帧 {fr}   ·   {kind}", font=f_big, fill=(22, 33, 43))
    frame = np.vstack([np.asarray(top)[:, :, ::-1], row])
    frames.append(frame)
    score_gap.append((ps[-1] - max(ps[:-1]) if len(ps) > 1 else 0) if tag == "block" else -1e9)
H, W = frames[0].shape[:2]
os.makedirs(os.path.dirname(out), exist_ok=True)
vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))   # MPEG-4 (no H.264 encoder here)
for f in frames:
    vw.write(f)
vw.release()
blk = [i for i, k in enumerate(keys) if k[0] == "block"]
order = sorted(blk, key=lambda i: score_gap[i])
med, ext = order[len(order) // 2], order[-1]
cv2.imencode(".jpg", frames[med], [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(out[:-4] + "_poster.jpg")
cv2.imencode(".jpg", frames[ext], [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(out[:-4] + "_extreme.jpg")
print(f"{out}: {len(frames)} frames ({len(blk)} block), {W}x{H}, {fps} fps; typical block frame {keys[med][1]} "
      f"(ours {score_gap[med]:+.1f} dB over the best other), extreme {keys[ext][1]} ({score_gap[ext]:+.1f} dB)")
