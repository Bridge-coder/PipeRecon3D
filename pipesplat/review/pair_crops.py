"""docs/10 X8b: before / after crops of pictures with the same camera (e.g. outside_views_obj.py's LABEL_<view>.png of
two OBJs, or obj_check sheets), side by side per crop box, each crop enlarged by --zoom, with labels.

usage: python -I pipesplat/review/pair_crops.py OUT.jpg --img LABEL=PATH [LABEL=PATH ..] --box x0 y0 x1 y1 [--box ..]
           [--zoom 2]
"""
import argparse

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "C:/Windows/Fonts/msyh.ttc"


def text(img, s):
    im = Image.fromarray(img[:, :, ::-1].copy())
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, 28)
    for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
        d.text((10 + dx, 6 + dy), s, font=f, fill=(0, 0, 0))
    d.text((10, 6), s, font=f, fill=(255, 255, 255))
    return np.asarray(im)[:, :, ::-1]


ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("out")
ap.add_argument("--img", nargs="+", required=True)
ap.add_argument("--box", nargs=4, type=int, action="append", required=True)
ap.add_argument("--zoom", type=float, default=2.0)
a = ap.parse_args()
imgs = [(s.split("=", 1)[0], cv2.imdecode(np.fromfile(s.split("=", 1)[1], np.uint8), cv2.IMREAD_COLOR)) for s in a.img]
rows = []
for x0, y0, x1, y1 in a.box:
    row = []
    for lab, im in imgs:
        c = im[y0:y1, x0:x1]
        c = cv2.resize(c, (int(c.shape[1] * a.zoom), int(c.shape[0] * a.zoom)), interpolation=cv2.INTER_NEAREST)
        row.append(np.pad(text(c, lab), ((0, 6), (0, 6), (0, 0)), constant_values=255))
    rows.append(np.hstack(row))
W = max(r.shape[1] for r in rows)
sheet = np.vstack([np.pad(r, ((0, 0), (0, W - r.shape[1]), (0, 0)), constant_values=255) for r in rows])
cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(a.out)
print(f"wrote {a.out} ({sheet.shape[1]}x{sheet.shape[0]})")
