"""How big are a trained model's Gaussians on screen? (docs/10: is the render blur a capacity limit?)
For some held-out frames, every Gaussian gets as colour its projected size in pixels (focal x its largest scale /
its depth; 1 sigma, an upper bound along the foreshortened direction), the frame is rendered with these colours, and
the opacity-weighted blob size per pixel is reported by distance from the image centre (tenths of the half diagonal).
A pixel-sharp render needs blobs of about 1 px or less.

usage: /opt/gsenv/bin/python pipesplat/diag_footprint.py SCENE_DIR RUN_DIR [--frames 0 11 22 33 43]
writes RUN_DIR/footprint.json and RUN_DIR/footprint_<frame>.png (blob size, 0..8 px as black..white)
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch
import gsplat

ap = argparse.ArgumentParser()
ap.add_argument("scene"), ap.add_argument("run"), ap.add_argument("--frames", type=int, nargs="+", default=[0, 11, 22, 33, 43])
a = ap.parse_args()
dev = "cuda"
d = np.load(os.path.join(a.scene, "scene.npz"))
m = torch.load(os.path.join(a.run, "model.pt"))
P = {k: m[k].to(dev) for k in ("means", "scales", "quats", "opacities")}
Kc, (CW, CH) = m["K"].to(dev), m["size"]
fx = Kc[0, 0].item()
op = torch.sigmoid(P["opacities"])
smax = torch.exp(P["scales"]).max(1).values
rr = np.hypot(*np.meshgrid(np.arange(CW) - (CW - 1) / 2, np.arange(CH) - (CH - 1) / 2)) / np.hypot(CW / 2, CH / 2)
res = {"gaussians": int(len(op)), "largest_scale_mm_median": float(smax.median().item() * 1000), "frames": {}}
for k in a.frames:
    v = torch.from_numpy(d["held_viewmats"][k]).to(dev)
    z = (P["means"] @ v[:3, :3].T + v[:3, 3])[:, 2].clamp(min=1e-4)
    px = (fx * smax / z)[:, None].repeat(1, 3)
    with torch.no_grad():
        out, al, _ = gsplat.rasterization(P["means"], P["quats"], torch.exp(P["scales"]), op, px, v[None], Kc[None],
                                          CW, CH, packed=False, near_plane=0.01)
    al = al[0, ..., 0].cpu().numpy()
    size = out[0, ..., 0].cpu().numpy() / np.maximum(al, 1e-6)
    ok = al > 0.5
    bins = [float(np.median(size[ok & (rr >= b / 10) & (rr < (b + 1) / 10)])) if (ok & (rr >= b / 10) & (rr < (b + 1) / 10)).any()
            else float("nan") for b in range(10)]
    res["frames"][int(d["held_ids"][k])] = dict(median_px=float(np.median(size[ok])), share_over_2px=float((size[ok] > 2).mean()),
                                                by_radius_px=[round(x, 2) for x in bins])
    print(f"frame {d['held_ids'][k]}: blob size median {np.median(size[ok]):.2f} px, > 2 px on {(size[ok] > 2).mean():.0%}; "
          f"by radius (centre -> corner): {' '.join(f'{x:.1f}' for x in bins)}")
    cv2.imwrite(os.path.join(a.run, f"footprint_{int(d['held_ids'][k]):06d}.png"),
                np.clip(size / 8 * 255, 0, 255).astype(np.uint8))
json.dump(res, open(os.path.join(a.run, "footprint.json"), "w"), indent=1)
