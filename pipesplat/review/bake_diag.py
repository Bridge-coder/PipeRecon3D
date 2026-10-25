"""docs/10 X8 diagnosis: does a raw pixel land on the same surface point (u, theta) through the two camera paths?
  raw path     train_tex.py: measured-surface depth rendered by gsplat 3DGUT in the raw frame (radial k1), back-projected
               along the undistorted ray of the pixel (undistort_grid)
  canvas path  train_gs.py / bake_gs_tex.py: depth rendered on the undistorted pinhole canvas, back-projected there,
               then the canvas maps taken back to the raw frame (train_gs.py to_raw, bilinear)
Prints the difference of u (mm) and theta (mm at the wall) by distance from the image centre, for a few poses.

usage: /opt/gsenv/bin/python pipesplat/review/bake_diag.py SCENE_DIR [MODEL_RUN BAKE_DIR [SURFACE.npz [MESH.npz]]]
part 2 (with MODEL_RUN BAKE_DIR): the baked texture rendered through the raw path on SURFACE (default: the
scene's surface.npz) against the Gaussian model from the same training / held-out poses (optical flow by radius)
"""
import os
import sys

import cv2
import numpy as np
import torch
import gsplat

dev = "cuda"
scene = sys.argv[1]
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


XU, YU = undistort_grid(*np.meshgrid((np.arange(w) - cx) / fx, (np.arange(h) - cy) / fy), k1)
CW, CH = 2 * int(np.ceil(np.nanmax(np.abs(XU)) * fx)) + 4, 2 * int(np.ceil(np.nanmax(np.abs(YU)) * fy)) + 4
ccx, ccy = CW / 2 - 0.5, CH / 2 - 0.5
to_raw = (np.nan_to_num(ccx + XU * fx, nan=-10).astype(np.float32), np.nan_to_num(ccy + YU * fy, nan=-10).astype(np.float32))
xu, yu = np.meshgrid((np.arange(CW) - ccx) / fx, (np.arange(CH) - ccy) / fy)
Kc_can = torch.tensor([[fx, 0, ccx + 0.5], [0, fy, ccy + 0.5], [0, 0, 1]], device=dev, dtype=torch.float32)
Kc_raw = torch.tensor([[fx, 0, cx + 0.5], [0, fy, cy + 0.5], [0, 0, 1]], device=dev, dtype=torch.float32)
UT = dict(with_ut=True, with_eval3d=True, radial_coeffs=torch.tensor([[k1, 0, 0, 0, 0, 0]], device=dev))


def quat_z_to(n):
    z = torch.tensor([0.0, 0.0, 1.0], device=n.device)
    axis = torch.cross(z.expand_as(n), n, dim=1)
    s = axis.norm(dim=1, keepdim=True)
    c = n[:, 2:3].clamp(-1, 1)
    ang = torch.atan2(s, c)
    axis = torch.where(s > 1e-8, axis / s.clamp(min=1e-12), torch.tensor([1.0, 0, 0], device=n.device).expand_as(n))
    return torch.cat([torch.cos(ang / 2), axis * torch.sin(ang / 2)], dim=1)


sf = np.load(sys.argv[4] if len(sys.argv) > 4 else os.path.join(scene, "surface.npz"))
S_c = torch.from_numpy(sf["centres"]).to(dev)
S_q = quat_z_to(torch.from_numpy(sf["normals"]).to(dev))
S_s = torch.stack([0.6 * torch.from_numpy(sf["size"]).to(dev)] * 2 + [torch.full((len(S_c),), 1e-5, device=dev)], 1)
S_o = torch.full((len(S_c),), 0.999, device=dev)
S_col = torch.zeros(len(S_c), 3, device=dev)


def depth(view, Kc, W, H, ut):
    out, al, _ = gsplat.rasterization(S_c, S_q, S_s, S_o, S_col, view[None], Kc[None], W, H, packed=False,
                                      render_mode="ED", near_plane=0.01, **(UT if ut else {}))
    dep = out[0, ..., 0].cpu().numpy()
    return np.where(al[0, ..., 0].cpu().numpy() > 0.5, dep, 0)


def to_world(view, x, y, D):
    R, t = view[:3, :3].cpu().numpy().astype(np.float64), view[:3, 3].cpu().numpy().astype(np.float64)
    Xc = np.stack([x * D, y * D, D], -1)
    return (Xc - t) @ R


rr = np.hypot(*np.meshgrid(np.arange(w) - cx, np.arange(h) - cy))
views = d["train_viewmats"]
for n in (100, 200, 300):
    view = torch.from_numpy(views[n]).to(dev)
    Dr = depth(view, Kc_raw, w, h, True)
    Xr = to_world(view, np.nan_to_num(XU), np.nan_to_num(YU), Dr)
    Dc = depth(view, Kc_can, CW, CH, False)
    Xc_ = to_world(view, xu, yu, Dc)
    Xc_raw = np.stack([cv2.remap(Xc_[..., k].astype(np.float32), *to_raw, cv2.INTER_LINEAR) for k in range(3)], -1)
    Dc_raw = cv2.remap(Dc.astype(np.float32), *to_raw, cv2.INTER_NEAREST)
    ok = (Dr > 0) & (Dc_raw > 0) & np.isfinite(XU)
    du_ = (Xr[..., 0] - Xc_raw[..., 0]) * 1000
    th = lambda X: np.arctan2(-X[..., 1], X[..., 2])  # noqa: E731
    dth = (np.remainder(th(Xr) - th(Xc_raw) + np.pi, 2 * np.pi) - np.pi) * 150
    rel = (Dr - Dc_raw) / np.maximum(Dc_raw, 1e-6)
    print(f"view {n} (x {d['train_eyes'][n, 0]:.2f} m): raw - canvas path")
    for lo, hi in ((0, 120), (120, 220), (220, 320), (320, 500)):
        m = ok & (rr >= lo) & (rr < hi)
        print(f"  r {lo}-{hi} px: du median {np.median(du_[m]):+.2f} mm (p10 {np.percentile(du_[m], 10):+.2f}, p90 "
              f"{np.percentile(du_[m], 90):+.2f}), dtheta {np.median(dth[m]):+.2f} mm, depth {np.median(rel[m]):+.3%}")
    # where a canvas pixel centre lands in the raw frame: x_d = x_u (1 + k1 r^2), check against to_raw inverse
    j, i = 300, 700
    g = 1 + k1 * (xu[j, i] ** 2 + yu[j, i] ** 2)
    ud, vd = cx + fx * xu[j, i] * g, cy + fy * yu[j, i] * g
    print(f"  canvas ({i}, {j}) -> raw ({ud:.3f}, {vd:.3f}); to_raw there ({to_raw[0][int(round(vd)), int(round(ud))]:.3f}, "
          f"{to_raw[1][int(round(vd)), int(round(ud))]:.3f})")

# ---- part 2: the baked texture (raw path) against the Gaussian model (canvas -> raw) from the same pose ----------
if len(sys.argv) > 3:
    import json
    sys.path.insert(0, "/mnt/c/Users/PC/Desktop/PipeRecon3D/lampmap")
    import imgmetrics as im
    run, bake = sys.argv[2], sys.argv[3]
    MR = None
    if len(sys.argv) > 5:                       # draw the exact triangles of this mesh instead of the discs
        sys.path.insert(0, "/mnt/c/Users/PC/Desktop/PipeRecon3D/pipesplat")
        import meshraster
        mz = np.load(sys.argv[5])
        MR = meshraster.MeshRaster(mz["xyz"], mz["tri"], K, k1, w, h, np.nan_to_num(XU), np.nan_to_num(YU), dev)
    tj = json.load(open(os.path.join(bake, "texture.json")))
    tex = cv2.imread(os.path.join(bake, "texture.png"))[:, :, ::-1].astype(np.float32)
    sv = torch.load(os.path.join(run, "model.pt"))
    GS = {k: sv[k].to(dev) for k in ("means", "quats", "scales", "opacities", "sh0", "shN")}
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    ur, vr = (gx - cx) / np.maximum(rr, 1), (gy - cy) / np.maximum(rr, 1)
    poses = [("train", n, views[n]) for n in (100, 200, 300)] + [("held", int(d["held_ids"][k]), d["held_viewmats"][k]) for k in (12, 28)]
    for tag, n, vm in poses:
        view = torch.from_numpy(vm).to(dev)
        with torch.no_grad():
            out, al, _ = gsplat.rasterization(GS["means"], GS["quats"], torch.exp(GS["scales"]), torch.sigmoid(GS["opacities"]),
                                              torch.cat([GS["sh0"], GS["shN"]], 1), view[None], Kc_can[None], CW, CH,
                                              sh_degree=3, packed=False, render_mode="RGB", near_plane=0.01)
        G = cv2.remap(out[0].clamp(0, 1).mul(255).cpu().numpy(), *to_raw, cv2.INTER_LINEAR)
        Dr = depth(view, Kc_raw, w, h, True) if MR is None else MR(view)[0].cpu().numpy()
        Xr = to_world(view, np.nan_to_num(XU), np.nan_to_num(YU), Dr)
        cu = ((Xr[..., 0] - tj["U0"]) / tj["du"] - 0.5).astype(np.float32)
        cv_ = ((np.arctan2(-Xr[..., 1], Xr[..., 2]) + np.pi) / tj["dth"] - 0.5).astype(np.float32)
        T = cv2.remap(tex, cu, cv_, cv2.INTER_LINEAR)
        m = (Dr > 0) & np.isfinite(XU) & (Xr[..., 0] < tj["cap_x"])
        T = im.linear_match(T, G, m)
        A = im._CLAHE.apply(np.clip(im.grey(G), 0, 255).astype(np.uint8))
        B = im._CLAHE.apply(np.clip(im.grey(T), 0, 255).astype(np.uint8))
        fl = im._DIS.calc(A, B, None)
        gs = cv2.GaussianBlur(A.astype(np.float32), (0, 0), 1.5)
        gm = np.hypot(cv2.Sobel(gs, cv2.CV_32F, 1, 0), cv2.Sobel(gs, cv2.CV_32F, 0, 1))
        sel = m & (gm > np.percentile(gm[m], 75))
        rad = fl[..., 0] * ur + fl[..., 1] * vr
        s = []
        for lo, hi in ((0, 120), (120, 220), (220, 320), (320, 500)):
            mm = sel & (rr >= lo) & (rr < hi)
            s.append(f"r{lo}-{hi} |f| {np.median(np.hypot(fl[..., 0], fl[..., 1])[mm]):.2f} rad {np.median(rad[mm]):+.2f}")
        r2 = im.ruler2(T, G, G, m)
        s.append(f"band corr {r2['corr_fine']:.3f}/{r2['corr_mid']:.3f}/{r2['corr_coarse']:.3f}")
        print(f"texture vs model, {tag} {n}: " + " | ".join(s))
        if tag == "train" and n == 200:
            cv2.imwrite(os.path.join(bake, "diag_train200.jpg"), np.hstack([G, T]).clip(0, 255).astype(np.uint8)[:, :, ::-1])

# ---- part 3 (with MESH.npz): ruler v2 against the video for the model and the texture drawn without mip-mapping ----
if len(sys.argv) > 5:
    held, hmask, hnext, hview, hid = (d[k_] for k_ in ("held_images", "held_mask", "held_next", "held_viewmats", "held_ids"))
    for k in (12, 28, 34):
        vm = hview[k]
        view = torch.from_numpy(vm).to(dev)
        with torch.no_grad():
            out, al, _ = gsplat.rasterization(GS["means"], GS["quats"], torch.exp(GS["scales"]), torch.sigmoid(GS["opacities"]),
                                              torch.cat([GS["sh0"], GS["shN"]], 1), view[None], Kc_can[None], CW, CH,
                                              sh_degree=3, packed=False, render_mode="RGB", near_plane=0.01)
        G = cv2.remap(out[0].clamp(0, 1).mul(255).cpu().numpy(), *to_raw, cv2.INTER_LINEAR)
        Dr = MR(view)[0].cpu().numpy()
        Xr = to_world(view, np.nan_to_num(XU), np.nan_to_num(YU), Dr)
        cu = ((Xr[..., 0] - tj["U0"]) / tj["du"] - 0.5).astype(np.float32)
        cv_ = ((np.arctan2(-Xr[..., 1], Xr[..., 2]) + np.pi) / tj["dth"] - 0.5).astype(np.float32)
        T0 = cv2.remap(tex, cu, cv_, cv2.INTER_LINEAR)
        real = held[k].astype(np.float32)
        m = hmask[k] & np.isfinite(XU) & (Dr > 0)
        nal = im.linear_match(im.align_to(hnext[k].astype(np.float32), real), real, m)
        line = f"held {int(hid[k])}:"
        for name, x in (("G0-A", G), ("bake level 0, no lamp map", T0)):
            xm = im.linear_match(x, real, m)
            r2 = im.ruler2(xm, real, nal, m)
            line += f" | {name} PSNR {im.psnr(xm, real, m):.2f} fid {r2['fid_fine']:.3f}/{r2['fid_mid']:.3f}/{r2['fid_coarse']:.3f}"
        gm_ = im.linear_match(T0, G, m)
        r2 = im.ruler2(gm_, G, G, m)
        line += f" | bake vs G0-A corr {r2['corr_fine']:.3f}/{r2['corr_mid']:.3f}/{r2['corr_coarse']:.3f}"
        print(line)
