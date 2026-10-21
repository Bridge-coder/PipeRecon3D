"""Where is a trained G0 model not opaque? Renders the opacity of some held-out frames in raw-frame geometry and
reports the share of pixels with alpha <= 0.99 by distance from the image centre (in tenths of the half-diagonal).

usage: /opt/gsenv/bin/python pipesplat/diag_alpha.py SCENE_DIR RUN_DIR [--frames 0 22 43]
writes RUN_DIR/alpha_<frame>.png (white = opaque, black = transparent; red = alpha <= 0.99)
"""
import argparse
import os

import cv2
import numpy as np
import torch
import gsplat

ap = argparse.ArgumentParser()
ap.add_argument("scene"), ap.add_argument("run"), ap.add_argument("--frames", type=int, nargs="+", default=[0, 22, 43])
a = ap.parse_args()
dev = "cuda"
d = np.load(os.path.join(a.scene, "scene.npz"))
m = torch.load(os.path.join(a.run, "model.pt"))
P = {k: m[k].to(dev) for k in ("means", "scales", "quats", "opacities", "sh0", "shN")}
Kc, (CW, CH) = m["K"].to(dev), m["size"]
K, k1 = d["K"].astype(np.float64), float(d["k1"])
w, h = (int(v) for v in d["size"])
fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
ccx, ccy = Kc[0, 2].item() - 0.5, Kc[1, 2].item() - 0.5
if k1 != 0:
    xd, yd = np.meshgrid((np.arange(w) - cx) / fx, (np.arange(h) - cy) / fy)
    xu, yu = xd.copy(), yd.copy()
    for _ in range(20):
        g = 1 + k1 * (xu ** 2 + yu ** 2)
        xu, yu = xd / g, yd / g
    mx, my = (ccx + xu * fx).astype(np.float32), (ccy + yu * fy).astype(np.float32)
rr = np.hypot(*np.meshgrid(np.arange(w) - (w - 1) / 2, np.arange(h) - (h - 1) / 2)) / np.hypot(w / 2, h / 2)
op = torch.sigmoid(P["opacities"])
print(f"{len(op):,} Gaussians; opacity median {op.median().item():.3f}, share < 0.05: {(op < 0.05).float().mean().item():.1%}")
for k in a.frames:
    v = torch.from_numpy(d["held_viewmats"][k]).to(dev)
    with torch.no_grad():
        _, al, _ = gsplat.rasterization(P["means"], P["quats"], torch.exp(P["scales"]), op,
                                        torch.cat([P["sh0"], P["shN"]], 1), v[None], Kc[None], CW, CH, sh_degree=3,
                                        packed=False, near_plane=0.01)
    al = al[0, ..., 0].cpu().numpy()
    if k1 != 0:
        al = cv2.remap(al, mx, my, cv2.INTER_LINEAR, borderValue=0)
    low = al <= 0.99
    bins = [f"{low[(rr >= b / 10) & (rr < (b + 1) / 10)].mean():.0%}" for b in range(10)]
    print(f"frame {d['held_ids'][k]}: alpha<=0.99 on {low.mean():.1%}; alpha median {np.median(al):.3f}, "
          f"p5 {np.percentile(al, 5):.3f}; by radius (centre -> corner): {' '.join(bins)}")
    vis = np.dstack([al, al, al]) * 255
    vis[low] = (vis[low] * 0.4 + np.array([0, 0, 255]) * 0.6)
    cv2.imwrite(os.path.join(a.run, f"alpha_{int(d['held_ids'][k]):06d}.png"), vis.astype(np.uint8))
