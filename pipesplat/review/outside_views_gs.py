"""docs/10: the new Gaussian model drawn with exactly the cameras of outside_views.py (outside / cut-away / side), for a
side-by-side comparison with the old model. Runs in WSL (gsplat).

Colours: the Gaussians' view-independent base colour (degree-0 spherical harmonics) - the view-dependent part was
learnt from inside the pipe only and means nothing from outside; for --lamp-ambient models the ambient colour is added.
Gaussians larger than --max-scale-mm (a few tenths of a per cent, docs/10 3.9) are left out: lying flat in the wall,
seen from outside they stick out of the pipe as sheets. White background, drawn at the cameras' oversized resolution and
averaged down like render_views.py.

usage: /opt/gsenv/bin/python pipesplat/review/outside_views_gs.py VIEW_DIR RUN_DIR [--max-scale-mm 8]
writes VIEW_DIR/new_<view>.png
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch
import gsplat

ap = argparse.ArgumentParser()
ap.add_argument("views"), ap.add_argument("run"), ap.add_argument("--max-scale-mm", type=float, default=8.0)
ap.add_argument("--opaque", action="store_true",
                help="draw every (anchored, wall-lying) Gaussian opaque: they were trained only from grazing views inside "
                     "the pipe, where many semi-transparent ones overlap; seen face-on from outside they let the "
                     "background through. Display only; files new_opaque_<view>.png")
a = ap.parse_args()
dev = "cuda"
cams = json.load(open(os.path.join(a.views, "cams.json")))
m = torch.load(os.path.join(a.run, "model.pt"), map_location="cpu")
means = m["means"].to(dev)
quats = torch.nn.functional.normalize(m["quats"].to(dev), dim=1)
scales = torch.exp(m["scales"].to(dev))
opac = torch.sigmoid(m["opacities"].to(dev))
if a.opaque:
    opac = torch.ones_like(opac)
rgb = (0.5 + 0.28209479177387814 * m["sh0"].to(dev)[:, 0, :])
if "amb" in m:
    rgb = rgb + torch.sigmoid(m["amb"].to(dev))
rgb = rgb.clamp(0, 1)
small = scales.max(1).values <= a.max_scale_mm / 1000
o = torch.tensor(cams["o"], device=dev, dtype=torch.float32)
Rp = torch.tensor(cams["Rp"], device=dev, dtype=torch.float32)
pipe = (means - o) @ Rp.T
SS = cams["SS"]
print(f"{len(means):,} Gaussians, {int((~small).sum()):,} larger than {a.max_scale_mm} mm left out")
for name, v in cams["views"].items():
    R_p = torch.tensor(v["R"], device=dev, dtype=torch.float32)
    eye_p = torch.tensor(v["eye"], device=dev, dtype=torch.float32)
    R_w = R_p @ Rp                                   # rows: right, up, forward in the world
    eye_w = o + Rp.T @ eye_p
    R_cv = torch.stack([R_w[0], -R_w[1], R_w[2]])    # OpenCV: x right, y down, z forward
    view = torch.eye(4, device=dev)
    view[:3, :3], view[:3, 3] = R_cv, -R_cv @ eye_w
    W, H = (int(s) for s in v["size"])
    f = v["scale"] if v["ortho"] else v["focal"]
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], device=dev, dtype=torch.float32)
    keep = small.clone()
    if v["cut_side"] is not None:
        keep &= pipe @ torch.tensor(v["cut_side"], device=dev, dtype=torch.float32) <= 0
    with torch.no_grad():
        img, alpha, _ = gsplat.rasterization(means[keep], quats[keep], scales[keep], opac[keep], rgb[keep], view[None],
                                             K[None], W, H, packed=False, near_plane=0.01, far_plane=1e4,
                                             backgrounds=torch.ones(1, 3, device=dev),
                                             camera_model="ortho" if v["ortho"] else "pinhole")
    out = (img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.float32)
    out = cv2.resize(out, (W // SS, H // SS), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(a.views, f"new_{'opaque_' if a.opaque else ''}{name}.png"), out.round().astype(np.uint8)[:, :, ::-1])
    print(f"{name}: {W // SS}x{H // SS}, {int(keep.sum()):,} Gaussians")
