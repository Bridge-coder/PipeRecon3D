"""The original camera data of an extracted InPipe-SLAM run as a playable video (the bag only stores images).

Left: RealSense colour camera (front, the one used for the visual reconstruction). Right: rear USB camera,
plus time and the encoder travel s (same definition as the fusion: (v6 - v7)/2, zero at the first SfM frame).
Frames are taken at a fixed rate on the common clock (nearest recorded image), so the video plays in real time.

usage: python -I make_video.py EXTRACT_DIR VISION_WORK RESULTS_DIR OUT_MP4 [--fps 20] [--width 960]
"""
import csv
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "fusion"))
from timebase import load_offsets, stream_times  # noqa: E402

args = sys.argv[1:]
FPS, WIDTH = 20.0, 960
for name in ("--fps", "--width"):
    if name in args:
        i = args.index(name)
        if name == "--fps":
            FPS = float(args[i + 1])
        else:
            WIDTH = int(args[i + 1])
        del args[i:i + 2]
EX, VW, RES, OUT = args
FONT = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 22)
FONT_S = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 18)

off = load_offsets(os.path.join(RES, "fusion", "time_offsets.json"))
tC, rC = stream_times(EX, off, "color")
tU, rU = stream_times(EX, off, "usbcam")
tE, rE = stream_times(EX, off, "encoders")
sE = np.array([(float(r["v6"]) - float(r["v7"])) / 2 for r in rE])
first_sfm = int(next(csv.DictReader(open(os.path.join(VW, "frames.csv"))))["color_idx"])
t_first = tC[[int(r["idx"]) for r in rC].index(first_sfm)]
s0 = np.interp(t_first, tE, sE)
sfm_idx = {int(r["color_idx"]) for r in csv.DictReader(open(os.path.join(VW, "frames.csv")))}

H = WIDTH * 9 // 16
w2, h2 = WIDTH // 2, H // 2
t0, t1 = max(tC[0], tU[0]), min(tC[-1], tU[-1])
times = np.arange(t0, t1, 1e9 / FPS)
fourcc = cv2.VideoWriter_fourcc(*"avc1")
vw = cv2.VideoWriter(OUT, fourcc, FPS, (WIDTH + w2, H))
if not vw.isOpened():
    vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH + w2, H))
cache = {}


def frame(stream, rows, tt, t, size):
    j = int(np.clip(np.searchsorted(tt, t), 1, len(tt) - 1))
    j = j - 1 if abs(tt[j - 1] - t) < abs(tt[j] - t) else j
    key = (stream, j)
    if key not in cache:
        if len(cache) > 8:
            cache.clear()
        im = cv2.imread(os.path.join(EX, "images", stream, f"{int(rows[j]['idx']):06d}.png"))
        cache[key] = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
    return cache[key], int(rows[j]["idx"])


for k, t in enumerate(times):
    front, ci = frame("color", rC, tC, t, (WIDTH, H))
    rear, _ = frame("usbcam", rU, tU, t, (w2, h2))
    canvas = np.full((H, WIDTH + w2, 3), 30, np.uint8)
    canvas[:, :WIDTH] = front
    canvas[:h2, WIDTH:] = rear
    img = Image.fromarray(canvas[:, :, ::-1])
    d = ImageDraw.Draw(img)
    s = np.interp(t, tE, sE) - s0
    d.text((12, 8), "前视彩色相机（RealSense，视觉重建用的就是它）", font=FONT_S, fill=(255, 255, 0))
    d.text((WIDTH + 10, 6), "后视相机（USB）", font=FONT_S, fill=(255, 255, 0))
    y = h2 + 20
    for line in (f"时间  {(t - t0) / 1e9:6.1f} s", f"编码器行程  s = {s:+.3f} m", f"彩色帧号  {ci}",
                 "★ 参与视觉重建的帧" if ci in sfm_idx else ""):
        d.text((WIDTH + 16, y), line, font=FONT, fill=(255, 255, 255))
        y += 36
    d.text((WIDTH + 16, H - 70), "s = 0：第一张参与重建的帧\ns 为负：机器人尚未出发", font=FONT_S, fill=(170, 170, 170))
    vw.write(np.asarray(img)[:, :, ::-1])
    if k % 200 == 0:
        print(f"{k}/{len(times)}", flush=True)
vw.release()
print(f"wrote {OUT}: {len(times)} frames at {FPS} fps ({len(times) / FPS:.0f} s), {os.path.getsize(OUT) / 1e6:.1f} MB")
