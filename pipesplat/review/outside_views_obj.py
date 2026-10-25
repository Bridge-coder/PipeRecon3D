"""docs/10 X8 / 5.5: a textured OBJ in the export_tex_obj.py layout (e.g. the texture baked from G0-A) drawn from outside
with the same cameras as outside_views.py (VIEW_DIR/cams.json) and the same shading, then one sheet per view kind:
old model (OpenMVS, CLAHE texture) / Gaussian model / this OBJ, cropped to the same box; the brightness of the Gaussian
and OBJ pictures scaled so that their median over the model equals the old picture's (as outside_compose.py).

usage: python -I pipesplat/review/outside_views_obj.py VIEW_DIR MODEL.obj LABEL OUT.jpg
writes VIEW_DIR/<LABEL>_<view>.png and the sheet OUT.jpg
"""
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (os.path.join(ROOT, "scripts", "vision"), os.path.join(ROOT, "scripts"), os.path.join(ROOT, "lampmap"), HERE):
    sys.path.insert(0, p)
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
import render_views as rv  # noqa: E402
from obj_check import read_tex_obj  # noqa: E402
from outside_views import vertex_normals  # noqa: E402

vdir, obj, lab, out = sys.argv[1:5]
cams = json.load(open(os.path.join(vdir, "cams.json"), encoding="utf-8"))
v, tri, uv, atlas = read_tex_obj(obj)
v = (v - np.array(cams["o"])) @ np.array(cams["Rp"]).T     # world -> the views' pipe frame (identity for vision data)
model = dict(xyz=v, tri=tri, uv=uv, tex=[atlas], tid=np.zeros(len(tri), int), blank=np.zeros(len(tri), bool),
             centroid=v[tri].mean(1), vn=vertex_normals(v, tri))
print(f"{obj}: {len(v):,} vertices, {len(tri):,} faces, atlas {atlas.shape[1]}x{atlas.shape[0]}")
for name, c in cams["views"].items():
    cam = dict(eye=np.array(c["eye"]), R=np.array(c["R"]), size=tuple(c["size"]))
    cam["scale" if c["ortho"] else "focal"] = c["scale"] if c["ortho"] else c["focal"]
    cut = name == "cut" or name.endswith("_cut")
    faces = ~rv.toward_viewer(model, cam) if cut else None
    img = rv.draw(model, cam, faces, light=0.25 if cut else rv.LIGHT, gain=rv.GAIN)
    rv.save(os.path.join(vdir, f"{lab}_{name}.png"), img)

FONT = "C:/Windows/Fonts/msyh.ttc"


def load(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


def match(img, ref):
    m, r = (img < 245).any(2), (ref < 245).any(2)
    g = np.median(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)[r]) / max(np.median(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[m]), 1)
    o = img.astype(np.float32)
    o[m] *= g
    return np.clip(o, 0, 255).astype(np.uint8)


def text(img, s):
    im = Image.fromarray(img[:, :, ::-1].copy())
    ImageDraw.Draw(im).text((12, 8), s, font=ImageFont.truetype(FONT, 34), fill=(0, 0, 0))
    return np.asarray(im)[:, :, ::-1]


panels = []
for name in cams["views"]:
    ims = [load(os.path.join(vdir, f"{p}_{name}.png")) for p in ("old", "new", lab)]
    ims[1], ims[2] = match(ims[1], ims[0]), match(ims[2], ims[0])
    m = np.zeros(ims[0].shape[:2], bool)
    for im in ims:
        m |= (im < 245).any(2)
    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = max(ys.min() - 20, 0), ys.max() + 20, max(xs.min() - 20, 0), xs.max() + 20
    for im, s in zip(ims, ("旧模型 OpenMVS 网格（CLAHE 贴图）", "高斯模型 G0-A", "烘焙网格（G0-A → 贴图）")):
        panels.append(text(im[y0:y1, x0:x1], f"{name}: {s}"))
W = max(p.shape[1] for p in panels)
sheet = np.vstack([np.pad(p, ((0, 6), (0, W - p.shape[1]), (0, 0)), constant_values=255) for p in panels])
if sheet.shape[1] > 2400:
    sheet = cv2.resize(sheet, (2400, int(sheet.shape[0] * 2400 / sheet.shape[1])), interpolation=cv2.INTER_AREA)
cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tofile(out)
print(f"wrote {out}")
