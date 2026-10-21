"""Check of the ruler itself (docs/10): what do the held-out metrics give for images that are the video frame itself,
only resampled / blurred / denoised / noisier?  No model involved - this is the ceiling and the bias of each metric.

  roundtrip      raw frame -> train_gs.py's undistorted training canvas -> back to raw (both cv2.remap bilinear):
                 the best any model trained and scored the way train_gs.py does it can reach
  roundtrip_int  the same with the canvas centre put on whole pixels (candidate fix)
  blur s         Gaussian blur sigma s px            denoise   cv2.fastNlMeansDenoisingColored h=5
  noise s        + white Gaussian noise sigma s (8-bit units)
  N              the next video frame (the reference used so far)

usage: /opt/gsenv/bin/python pipesplat/diag_ruler.py SCENE_DIR OUT_JSON
"""
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "lampmap"))
import imgmetrics as im  # noqa: E402

scene, out_json = sys.argv[1], sys.argv[2]
d = np.load(os.path.join(scene, "scene.npz"))
K = d["K"].astype(np.float64)
k1 = float(d["k1"])
w, h = (int(v) for v in d["size"])
fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]


def undistort_grid(xd, yd, k1):
    out = (xd ** 2 + yd ** 2 >= (0.95 * 2 / (3 * np.sqrt(-3 * k1))) ** 2) if k1 < 0 else np.zeros(xd.shape, bool)
    xd, yd = np.where(out, 0.0, xd), np.where(out, 0.0, yd)
    xu, yu = xd.copy(), yd.copy()
    for _ in range(20):
        g = 1 + k1 * (xu ** 2 + yu ** 2)
        xu, yu = xd / g, yd / g
    xu[out], yu[out] = np.nan, np.nan
    return xu, yu


def maps(aligned):
    """train_gs.py's canvas maps; aligned=True puts the canvas centre at cx/cy plus a whole number of pixels"""
    XU, YU = undistort_grid(*np.meshgrid((np.arange(w) - cx) / fx, (np.arange(h) - cy) / fy), k1)
    valid_raw = np.isfinite(XU)
    CW, CH = 2 * int(np.ceil(np.nanmax(np.abs(XU)) * fx)) + 4, 2 * int(np.ceil(np.nanmax(np.abs(YU)) * fy)) + 4
    ccx, ccy = CW / 2 - 0.5, CH / 2 - 0.5
    if aligned:
        ccx, ccy = cx + round(ccx - cx), cy + round(ccy - cy)
    to_raw = (np.nan_to_num(ccx + XU * fx, nan=-10).astype(np.float32), np.nan_to_num(ccy + YU * fy, nan=-10).astype(np.float32))
    xu, yu = np.meshgrid((np.arange(CW) - ccx) / fx, (np.arange(CH) - ccy) / fy)
    g = 1 + k1 * (xu ** 2 + yu ** 2)
    to_canvas = ((cx + fx * xu * g).astype(np.float32), (cy + fy * yu * g).astype(np.float32))
    frac = float(np.abs(((ccx - cx) % 1.0)))
    return valid_raw, to_canvas, to_raw, frac


def roundtrip(img, mp):
    _, tc, tr, _ = mp
    can = cv2.remap(img, tc[0], tc[1], cv2.INTER_LINEAR, borderValue=0)
    return cv2.remap(can, tr[0], tr[1], cv2.INTER_LINEAR, borderValue=0)


rng = np.random.default_rng(0)
variants = {}
if k1 != 0:
    mp0, mp1 = maps(False), maps(True)
    print(f"canvas centre fraction: current {mp0[3]:.2f} px, aligned {mp1[3]:.2f} px")
    variants["roundtrip"] = lambda x: roundtrip(x, mp0)
    variants["roundtrip_int"] = lambda x: roundtrip(x, mp1)
    valid_raw = mp0[0]
else:
    valid_raw = np.ones((h, w), bool)
for s in (0.5, 0.8, 1.2):
    variants[f"blur{s}"] = (lambda s: lambda x: cv2.GaussianBlur(x, (0, 0), s))(s)
variants["denoise"] = lambda x: cv2.fastNlMeansDenoisingColored(x.astype(np.uint8), None, 5, 5, 7, 21).astype(np.float32)
for s in (3.0, 6.0):
    variants[f"noise{s:.0f}"] = (lambda s: lambda x: np.clip(x + rng.normal(0, s, x.shape), 0, 255).astype(np.float32))(s)

rows = {k: [] for k in list(variants) + ["N"]}
imgs, nxt, masks = d["held_images"], d["held_next"], d["held_mask"]
for k in range(len(imgs)):
    real = imgs[k].astype(np.float32)
    m = masks[k] & valid_raw
    cand = {name: f(real) for name, f in variants.items()}
    cand["N"] = nxt[k].astype(np.float32)
    for name, img in cand.items():
        mk = m & (img.sum(-1) > 0) if name.startswith("roundtrip") else m
        x = im.linear_match(img, real, mk)
        rows[name].append(im.all_metrics(x, real, mk))
    print(f"{k + 1}/{len(imgs)}", end="\r", flush=True)
summary = {name: {key: float(np.nanmedian([r[key] for r in rs])) for key in rs[0]} for name, rs in rows.items()}
print()
for name, s in summary.items():
    print(f"{name:14s} " + "  ".join(f"{key} {v:.3f}" for key, v in s.items()))
json.dump(dict(scene=scene, n=len(imgs), summary=summary), open(out_json, "w"), indent=1)
