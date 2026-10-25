"""docs/10: old vs new model views side by side (outputs of outside_views.py + outside_views_gs.py), one sheet per data
set: outside, cut-away, side elevation (outside / cut), each as old (v1.0 model) above new (Gaussian model), cropped to
the same box. The new picture's brightness is scaled so that its median over the model equals the old one's (the old
pictures carry render_views' texture gain and shading, the Gaussians the raw video colours).

usage: python -I pipesplat/review/outside_compose.py VIEW_DIR OUT.jpg "TITLE" "OLD LABEL" "NEW LABEL" ["OPAQUE LABEL"]
       (with the last label, the new_opaque_<view>.png pictures of outside_views_gs.py --opaque are a third row)
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

vdir, out, title, old_lab, new_lab = sys.argv[1:6]
opq_lab = sys.argv[6] if len(sys.argv) > 6 else None
FONT = "C:/Windows/Fonts/msyh.ttc"


def load(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


def box(*imgs, pad=20):
    m = np.zeros(imgs[0].shape[:2], bool)
    for im in imgs:
        m |= (im < 245).any(2)
    ys, xs = np.nonzero(m)
    return max(ys.min() - pad, 0), ys.max() + pad, max(xs.min() - pad, 0), xs.max() + pad


def match(new, old):
    mo, mn = (old < 245).any(2), (new < 245).any(2)
    g = np.median(old[mo].mean(1)) / max(np.median(new[mn].mean(1)), 1)
    adj = new.astype(np.float32)
    adj[mn] = np.clip(adj[mn] * g, 0, 255)
    return adj.astype(np.uint8), g


def label(img, text, size=34):
    pil = Image.fromarray(img[:, :, ::-1])
    d = ImageDraw.Draw(pil)
    d.text((12, 8), text, font=ImageFont.truetype(FONT, size), fill=(20, 20, 20))
    return np.array(pil)[:, :, ::-1]


blocks = []
views = [("whole", "管外（透视）"), ("cut", "剖开：去掉靠近观察者的一半（透视）")]
views += sorted({(n[4:-4].rsplit("_", 1)[0] + "_out", "侧视：外观（正投影，按比例）") for n in os.listdir(vdir) if n.startswith("old_side")})
views += sorted({(n[4:-4].rsplit("_", 1)[0] + "_cut", "侧视：剖开，看到远侧内壁（正投影）") for n in os.listdir(vdir) if n.startswith("old_side")})
W = 2000
for name, desc in views:
    old, new = load(os.path.join(vdir, f"old_{name}.png")), load(os.path.join(vdir, f"new_{name}.png"))
    new, g = match(new, old)
    rows = [(old, old_lab), (new, new_lab)]
    if opq_lab:
        opq, _ = match(load(os.path.join(vdir, f"new_opaque_{name}.png")), old)
        rows.append((opq, opq_lab))
    y0, y1, x0, x1 = box(*[r[0] for r in rows])
    for img, lab in rows:
        c = img[y0:y1, x0:x1]
        c = cv2.resize(c, (W, max(1, int(round(c.shape[0] * W / c.shape[1])))), interpolation=cv2.INTER_AREA)
        top = np.full((52, W, 3), 255, np.uint8)
        blocks.append(label(np.vstack([top, c]), f"{desc}  -  {lab}"))
    blocks.append(np.full((24, W, 3), 255, np.uint8))
head = label(np.full((70, W, 3), 255, np.uint8), title, size=40)
sheet = np.vstack([head] + blocks)
cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(out)
print(f"wrote {out} ({sheet.shape[1]}x{sheet.shape[0]})")
