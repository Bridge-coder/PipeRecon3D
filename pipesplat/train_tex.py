"""Texture model TX (docs/10 X4): the measured surface is kept fixed and only a high-resolution texture on it is
learned from the RAW video frames by differentiable rendering - the deliverable is a textured mesh, unlike the
Gaussian models. Runs in the pipe3d WSL distro (gsplat only rasterises the measured surface for depth and normals).

Per training frame and iteration:
  1. depth D and lamp normal N of the measured surface (SCENE_DIR/surface.npz, its triangles as opaque thin discs,
     as train_gs.py --anchor) rendered directly in the raw frame (3DGUT with the radial k1, no resampling)
  2. every pixel's surface point X -> pipe coordinates: vision = axial x and angle about the x axis (the frame of
     scene.npz is the pipe frame of render_views.load_model); InPipe = the R2 centerline (u, theta), as R4
  3. colour = texture sampled there, with mip-mapping by the pixel's footprint in texels (trilinear), times the lamp
     factor of train_gs.py --lamp (camera-fixed map M x cos(incidence) / (r / r_ref)^2, learned lamp offset),
     plus (--ambient) a coarse ambient texture that the lamp does not scale, then a gain and offset per frame/channel
  4. loss 0.8 L1 + 0.2 D-SSIM against the raw frame where the surface covers it
  The end cap of a closed pipe (vision: surface beyond --cap-x with the lamp normal along the axis) gets its own
  planar texture in (y, z). --pose-opt adds a small rigid correction per training frame (rotation + camera centre;
  ray-plane intersection with the rendered surface point and normal gives the gradient), held-out frames take the
  correction interpolated between their two neighbouring training frames.
Scoring: the same frames and ruler v2 as train_gs.py --ruler2 (B's mask, references Nal and D) plus the old
metrics, summary_r2.json in the same layout, so summarize_r2.py lists TX next to the Gaussian runs.

usage: TORCH_HOME=/mnt/d/pipe3d_gs/torch_cache /opt/gsenv/bin/python pipesplat/train_tex.py SCENE_DIR OUT_DIR
           [--texel-mm 0.5] [--iters 20000] [--no-lamp] [--ambient] [--pose-opt] [--save 4] [--export]
writes OUT_DIR/train.log, tex.pt (texture + lamp + gains + pose corrections), summary_r2.json, per_frame_r2.csv,
       renders/*.jpg (video | B | TX), and with --export texture.png (+ cap.png) for the textured mesh (export_tex_obj.py)
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import gsplat
from fused_ssim import fused_ssim

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "lampmap"))
import imgmetrics as im  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("scene"), ap.add_argument("out")
ap.add_argument("--texel-mm", type=float, default=0.5)
ap.add_argument("--iters", type=int, default=20000)
ap.add_argument("--no-lamp", action="store_true", help="no lamp factor (texture x per-frame gain only)")
ap.add_argument("--ambient", action="store_true", help="ambient texture (4 mm) not scaled by the lamp (InPipe)")
ap.add_argument("--pose-opt", action="store_true", help="learn a small rigid correction of every training pose")
ap.add_argument("--disp-mm", type=float, default=0.0,
                help="also learn a radial displacement of the measured surface on a grid of this size (mm; 0 = off): "
                     "the wall moves in or out by up to --disp-max-mm where the images say it lies elsewhere")
ap.add_argument("--disp-max-mm", type=float, default=3.0)
ap.add_argument("--lamp-smooth-mm", type=float, default=20.0)
ap.add_argument("--surface", default="surface.npz", help="measured-surface file in SCENE_DIR (export_surface.py)")
ap.add_argument("--cap-x", type=float, default=None, help="vision: surface beyond this x with axial normal = end cap "
                "(default: last training camera + 0.15 m)")
ap.add_argument("--tex-clamp", action="store_true",
                help="the first version (run_tx, run_tx_disp2): texture values learnt directly and clamped to [0, 1.5] - "
                     "4-8 %% of the texels sat at the clamp and 1 %% at 0 (docs/10 X4c). Default now: log values, no clamp")
ap.add_argument("--dssim", type=float, default=0.2)
ap.add_argument("--lr", type=float, default=0.01)
ap.add_argument("--save", type=int, default=4), ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--eval-only", action="store_true")
ap.add_argument("--export", action="store_true", help="write the texture images for export_tex_obj.py")
a = ap.parse_args()
torch.manual_seed(a.seed)
np.random.seed(a.seed)
dev = "cuda"
os.makedirs(os.path.join(a.out, "renders"), exist_ok=True)
T0 = time.time()
_logf = open(os.path.join(a.out, "train.log"), "a", encoding="utf-8")


def log(msg):
    line = f"{msg} [{time.time() - T0:.0f} s]"
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


log(f"args {vars(a)} | torch {torch.__version__} gsplat {gsplat.__version__}")
d = np.load(os.path.join(a.scene, "scene.npz"))
K = d["K"].astype(np.float64)
k1 = float(d["k1"])
w, h = (int(v) for v in d["size"])
fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
inpipe = "depth_scan" in d.files


# ------------------------------------------------------------------------------ raw camera --------------------
def undistort_grid(xd, yd, k1):
    """inverse of x_d = x_u (1 + k1 r_u^2) (train_gs.py); NaN beyond 95 % of the radius the model reaches"""
    out = (xd ** 2 + yd ** 2 >= (0.95 * 2 / (3 * np.sqrt(-3 * k1))) ** 2) if k1 < 0 else np.zeros(xd.shape, bool)
    xd, yd = np.where(out, 0.0, xd), np.where(out, 0.0, yd)
    xu, yu = xd.copy(), yd.copy()
    for _ in range(20):
        g = 1 + k1 * (xu ** 2 + yu ** 2)
        xu, yu = xd / g, yd / g
    xu[out], yu[out] = np.nan, np.nan
    return xu, yu


XU, YU = undistort_grid(*np.meshgrid((np.arange(w) - cx) / fx, (np.arange(h) - cy) / fy), k1)
valid_raw = np.isfinite(XU)
RX = torch.from_numpy(np.nan_to_num(XU).astype(np.float32)).to(dev)
RY = torch.from_numpy(np.nan_to_num(YU).astype(np.float32)).to(dev)
VALID = torch.from_numpy(valid_raw).to(dev)
Kc = torch.tensor([[fx, 0, cx + 0.5], [0, fy, cy + 0.5], [0, 0, 1]], device=dev, dtype=torch.float32)
UT = dict(with_ut=True, with_eval3d=True, radial_coeffs=torch.tensor([[k1, 0, 0, 0, 0, 0]], device=dev)) if k1 else {}
log(f"raw {w}x{h}, k1 {k1}: rendering in the raw frame{' through 3DGUT' if UT else ''}, valid {valid_raw.mean():.1%}")

# ------------------------------------------------------------------------------ frames ------------------------
views_all = torch.from_numpy(d["train_viewmats"]).to(dev)
block_idx = d["block_idx"]
train = np.setdiff1d(np.arange(len(views_all)), block_idx)
raw_train = d["train_images"]                   # read once (each d[...] access re-reads the array from the npz)
IM = {int(i): torch.from_numpy(np.ascontiguousarray(raw_train[i])).pin_memory() for i in train}
log(f"{len(train)} training frames ({len(block_idx)} block frames left out)")

# ------------------------------------------------------------------------------ measured surface --------------
sf = np.load(os.path.join(a.scene, a.surface))
S_c = torch.from_numpy(sf["centres"]).to(dev)
S_n = torch.from_numpy(sf["normals"]).to(dev)


def quat_z_to(n):
    z = torch.tensor([0.0, 0.0, 1.0], device=n.device)
    axis = torch.cross(z.expand_as(n), n, dim=1)
    s = axis.norm(dim=1, keepdim=True)
    c = n[:, 2:3].clamp(-1, 1)
    ang = torch.atan2(s, c)
    axis = torch.where(s > 1e-8, axis / s.clamp(min=1e-12), torch.tensor([1.0, 0, 0], device=n.device).expand_as(n))
    return torch.cat([torch.cos(ang / 2), axis * torch.sin(ang / 2)], dim=1)


def smooth_normals(c, n, radius):
    """structure-tensor average of the triangle normals within about radius (train_gs.py)"""
    key = torch.floor(c / (radius / 2)).long()
    key = key - key.min(0).values + 1
    dims = key.max(0).values + 2
    lin = (key[:, 0] * dims[1] + key[:, 1]) * dims[2] + key[:, 2]
    uniq, inv = torch.unique(lin, return_inverse=True)
    T = (n[:, :, None] * n[:, None, :]).reshape(-1, 9)
    V = torch.zeros(len(uniq), 9, device=c.device).index_add_(0, inv, T)
    acc = torch.zeros_like(T)
    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            for oz in (-1, 0, 1):
                q = lin + (ox * dims[1] + oy) * dims[2] + oz
                pos = torch.searchsorted(uniq, q).clamp(max=len(uniq) - 1)
                hit = uniq[pos] == q
                acc[hit] += V[pos[hit]]
    A = acc.reshape(-1, 3, 3)
    m = n.clone()
    for _ in range(15):
        m = torch.bmm(A, m[:, :, None])[:, :, 0]
        m = m / m.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return torch.where((m * n).sum(1, keepdim=True) < 0, -m, m)


S_q = quat_z_to(S_n)
S_s = torch.stack([0.6 * torch.from_numpy(sf["size"]).to(dev)] * 2 + [torch.full((len(S_c),), 1e-5, device=dev)], 1)
S_o = torch.full((len(S_c),), 0.999, device=dev)
S_ns = smooth_normals(S_c, S_n, a.lamp_smooth_mm / 1000)
log(f"measured surface: {len(S_c):,} triangles; lamp normals averaged over ~{a.lamp_smooth_mm:.0f} mm")


@torch.no_grad()
def surface_dn(view):
    """depth, smoothed normal (world) and coverage of the measured surface in the raw frame"""
    out, al, _ = gsplat.rasterization(S_c, S_q, S_s, S_o, S_ns, view[None], Kc[None], w, h, packed=False,
                                      render_mode="RGB+ED", near_plane=0.01, **UT)
    a0 = al[0, ..., 0]
    ok = (a0 > 0.5) & VALID
    D = torch.where(ok, out[0, ..., 3], torch.zeros_like(a0))
    N = out[0, ..., :3] / a0[..., None].clamp(min=1e-6)
    return D, N / N.norm(dim=-1, keepdim=True).clamp(min=1e-12), ok


# ------------------------------------------------------------------------------ pipe coordinates --------------
if inpipe:
    cl = np.load(os.path.join(a.scene, "centerline.npz"))
    CL = {k: torch.from_numpy(cl[k].astype(np.float32)).to(dev) for k in ("u", "C", "T", "N", "Bn")}
    CL_th0, CL_ds = float(cl["th0"]), float(cl["u"][1] - cl["u"][0])
    CL_sub = CL["C"][::10]

    def to_pipe(X):
        """(u m, theta rad) of world points (recon_common.Centerline.coords)"""
        i = torch.cdist(X, CL_sub).argmin(1) * 10
        u = CL["u"][i]
        for _ in range(4):
            i = ((u - CL["u"][0]) / CL_ds).round().long().clamp(0, len(CL["u"]) - 1)
            u = CL["u"][i] + ((X - CL["C"][i]) * CL["T"][i]).sum(1)
        i = ((u - CL["u"][0]) / CL_ds).round().long().clamp(0, len(CL["u"]) - 1)
        q = X - CL["C"][i]
        th = torch.atan2((q * CL["Bn"][i]).sum(1), (q * CL["N"][i]).sum(1)) - math.radians(CL_th0)
        return u, torch.remainder(th + math.pi, 2 * math.pi) - math.pi
    R_mean = 0.095
else:
    def to_pipe(X):
        return X[:, 0], torch.atan2(-X[:, 1], X[:, 2])
    R_mean = float(np.median(np.hypot(sf["centres"][:, 1], sf["centres"][:, 2])))

def radial(X):
    """outward unit vector from the pipe axis at world points X"""
    if inpipe:
        i = torch.cdist(X, CL_sub).argmin(1) * 10
        u = CL["u"][i]
        for _ in range(4):
            i = ((u - CL["u"][0]) / CL_ds).round().long().clamp(0, len(CL["u"]) - 1)
            u = CL["u"][i] + ((X - CL["C"][i]) * CL["T"][i]).sum(1)
        q = X - CL["C"][i]
        q = q - (q * CL["T"][i]).sum(1, keepdim=True) * CL["T"][i]
    else:
        q = torch.stack([torch.zeros_like(X[:, 0]), X[:, 1], X[:, 2]], 1)
    return q / q.norm(dim=1, keepdim=True).clamp(min=1e-9)


u_all = to_pipe(S_c)[0]
U0, U1 = float(u_all.min()) - 0.01, float(u_all.max()) + 0.01
du = a.texel_mm / 1000
dth = du / R_mean
NU, NTH = int(math.ceil((U1 - U0) / du)), int(math.ceil(2 * math.pi / dth))
dth = 2 * math.pi / NTH
cap_x = None
if not inpipe:
    cap_x = a.cap_x if a.cap_x is not None else float(d["train_eyes"][:, 0].max()) + 0.15
CAP_N = int(math.ceil(2 * 0.2 / du)) if cap_x is not None else 0
log(f"texture {NTH} x {NU} texels ({a.texel_mm} mm along the pipe, {math.degrees(dth):.3f} deg = "
    f"{dth * R_mean * 1000:.2f} mm around at r {R_mean * 1000:.0f} mm), u {U0:.3f}..{U1:.3f} m"
    + (f"; end cap beyond x {cap_x:.2f} m: planar {CAP_N} x {CAP_N}" if cap_x is not None else ""))

# ------------------------------------------------------------------------------ parameters --------------------
TEX0 = 0.3 if a.tex_clamp else math.log(0.3)
tex = torch.nn.Parameter(torch.full((1, 3, NTH, NU), TEX0, device=dev))
params = {"tex": tex}
lrs = {"tex": a.lr}
if CAP_N:
    params["cap"] = torch.nn.Parameter(torch.full((1, 3, CAP_N, CAP_N), TEX0, device=dev))
    lrs["cap"] = a.lr
if a.ambient:
    na, nt = max(2, int((U1 - U0) / 0.004)), max(8, int(2 * math.pi * R_mean / 0.004))
    params["amb"] = torch.nn.Parameter(torch.full((1, 3, nt, na), -4.6, device=dev))   # sigmoid -> 0.01
    lrs["amb"] = 5e-3
GY, GX = 36, 48
lamp_on = not a.no_lamp
params["gain"] = torch.nn.Parameter(torch.zeros(len(views_all), 3, device=dev))
params["off"] = torch.nn.Parameter(torch.zeros(len(views_all), 3, device=dev))
lrs.update(gain=1e-3, off=1e-4)
if lamp_on:
    params["logM"] = torch.nn.Parameter(torch.zeros(1, 1, GY, GX, device=dev))
    params["lamp_t"] = torch.nn.Parameter(torch.zeros(3, device=dev))
    lrs.update(logM=3e-3, lamp_t=1e-4)
if a.disp_mm > 0:
    NUd = max(2, int(math.ceil((U1 - U0) / (a.disp_mm / 1000))))
    NTHd = max(8, int(math.ceil(2 * math.pi * R_mean / (a.disp_mm / 1000))))
    params["disp"] = torch.nn.Parameter(torch.zeros(1, 1, NTHd, NUd, device=dev))   # tanh -> +-disp_max_mm
    lrs["disp"] = 2e-3
if a.pose_opt:
    params["pose_w"] = torch.nn.Parameter(torch.zeros(len(views_all), 3, device=dev))   # rotation vector, rad
    params["pose_c"] = torch.nn.Parameter(torch.zeros(len(views_all), 3, device=dev))   # camera centre, m (world)
    params["pose_wg"] = torch.nn.Parameter(torch.zeros(3, device=dev))   # shared by all frames: rotation (camera)
    params["pose_cg"] = torch.nn.Parameter(torch.zeros(3, device=dev))   # and shift in camera axes (= extrinsic error)
    lrs.update(pose_w=2e-5, pose_c=2e-5, pose_wg=1e-4, pose_cg=1e-4)


def so3(wv):
    """rotation matrices of rotation vectors [N, 3]"""
    th = wv.norm(dim=1, keepdim=True).clamp(min=1e-12)
    k = wv / th
    Kx = torch.zeros(len(wv), 3, 3, device=wv.device)
    Kx[:, 0, 1], Kx[:, 0, 2], Kx[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    Kx = Kx - Kx.transpose(1, 2)
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    return torch.eye(3, device=wv.device)[None] + s * Kx + (1 - c) * (Kx @ Kx)


def corrected_view(view, wv, dc, dc_cam):
    """world->camera matrix with the correction: rotation exp(wv) applied in the camera, centre moved by dc (world)
    plus dc_cam (camera axes)"""
    R, t = view[:3, :3], view[:3, 3]
    C = -R.T @ t
    R2 = so3(wv[None])[0] @ R
    C2 = C + dc + R.T @ dc_cam
    out = torch.eye(4, device=view.device)
    out[:3, :3], out[:3, 3] = R2, -R2 @ C2
    return out


# ------------------------------------------------------------------------------ rendering ---------------------
def lin(t):
    """texture parameter -> linear colour (log parametrisation unless --tex-clamp)"""
    return t if a.tex_clamp else torch.exp(t)


def pyramid(t, levels):
    out = [lin(t)]
    for _ in range(levels - 1):
        x = out[-1]
        if x.shape[-1] < 2 or x.shape[-2] < 2:
            break
        out.append(F.avg_pool2d(x[..., : x.shape[-2] // 2 * 2, : x.shape[-1] // 2 * 2], 2))
    return out


def sample_mip(pyr, gu, gv, level, wrap_v):
    """trilinear sample of a texture pyramid at texel coordinates (gu along width, gv along height of level 0)"""
    if len(pyr) == 1:
        t = pyr[0]
        Hh, Ww = t.shape[-2:]
        u, v = gu, gv
        if wrap_v:
            t = torch.cat([t[..., -1:, :], t, t[..., :1, :]], dim=-2)
            v = torch.remainder(v + 0.5, Hh) - 0.5 + 1
            Hh += 2
        grid = torch.stack([(u + 0.5) / Ww * 2 - 1, (v + 0.5) / Hh * 2 - 1], -1)[None, None]
        return F.grid_sample(t, grid, mode="bilinear", padding_mode="border", align_corners=False)[0, :, 0].T
    lv = level.clamp(0, len(pyr) - 1 - 1e-4)
    l0 = lv.floor().long()
    fr = (lv - l0.float())[:, None]
    res = torch.zeros(len(gu), 3, device=gu.device)
    for L in range(len(pyr)):
        sel0, sel1 = l0 == L, l0 == L - 1
        sel = sel0 | sel1
        if not sel.any():
            continue
        t = pyr[L]
        Hh, Ww = t.shape[-2:]
        s = 2.0 ** L
        u, v = (gu[sel] + 0.5) / s - 0.5, (gv[sel] + 0.5) / s - 0.5
        if wrap_v:                              # circular in theta: pad one texel each side
            t = torch.cat([t[..., -1:, :], t, t[..., :1, :]], dim=-2)
            v = torch.remainder(v + 0.5, Hh) - 0.5 + 1
            Hh += 2
        grid = torch.stack([(u + 0.5) / Ww * 2 - 1, (v + 0.5) / Hh * 2 - 1], -1)[None, None]
        c = F.grid_sample(t, grid, mode="bilinear", padding_mode="border", align_corners=False)[0, :, 0].T
        wgt = torch.where(sel0[sel], 1 - fr[sel][:, 0], fr[sel][:, 0])[:, None]
        res[sel] += wgt * c
    return res


r_ref = None


def render(n, view_base, pose_idx=None, held_w=None):
    """raw-frame render of training frame n (or a held-out pose): colour [h, w, 3] before gain/offset, coverage"""
    global r_ref
    if a.pose_opt and (pose_idx is not None):
        if held_w is None:
            wv, dc = params["pose_w"][pose_idx], params["pose_c"][pose_idx]
        else:
            (i0, i1), t_ = pose_idx, held_w
            wv = (1 - t_) * params["pose_w"][i0] + t_ * params["pose_w"][i1]
            dc = (1 - t_) * params["pose_c"][i0] + t_ * params["pose_c"][i1]
        view = corrected_view(view_base, wv + params["pose_wg"], dc, params["pose_cg"])
    else:
        view = view_base
    D, Nw, ok = surface_dn(view.detach())
    R, t = view[:3, :3], view[:3, 3]
    C = -R.T @ t
    ray = torch.stack([RX[ok], RY[ok], torch.ones_like(RX[ok])], -1) @ R          # world directions (R^T d)
    Rd, td = R.detach(), t.detach()
    X0 = (torch.stack([RX[ok] * D[ok], RY[ok] * D[ok], D[ok]], -1) - td) @ Rd     # surface point (no grad)
    if a.pose_opt and view.requires_grad:      # ray-plane intersection: gradient towards the pose correction
        nrm = Nw[ok]                            # world normal (the normal map is rendered from world normals)
        s = ((X0 - C) * nrm).sum(1) / (ray * nrm).sum(1).clamp(min=1e-4)
        X = C + s[:, None] * ray
    else:
        X = X0
    if a.disp_mm > 0:                           # move the wall along the radius by h(u, theta), along the pixel's ray
        u0_, th0_ = to_pipe(X.detach())
        hd = sample_mip([params["disp"]], (u0_ - U0) / (U1 - U0) * params["disp"].shape[-1] - 0.5,
                        (th0_ + math.pi) / (2 * math.pi) * params["disp"].shape[-2] - 0.5, None, wrap_v=True)[:, 0]
        hm = a.disp_max_mm / 1000 * torch.tanh(hd)
        if CAP_N:                               # not on the end cap
            hm = torch.where((X[:, 0] > cap_x) & (Nw[ok][:, 0].abs() > 0.5), torch.zeros_like(hm), hm)
        er = radial(X.detach())
        dr = (ray * er).sum(1)
        dr = torch.where(dr.abs() < 0.05 * ray.norm(dim=1), 0.05 * ray.norm(dim=1) * torch.sign(dr + 1e-12), dr)
        X = X + ray * (hm / dr)[:, None]
    u, th = to_pipe(X)
    gu = (u - U0) / du - 0.5
    gv = (th + math.pi) / dth - 0.5
    # footprint in texels from the neighbouring pixels (no grad)
    with torch.no_grad():
        GU = torch.zeros(h, w, device=dev)
        GV = torch.zeros(h, w, device=dev)
        GU[ok], GV[ok] = gu.detach(), gv.detach()
        dvx = torch.remainder(GV[:, 1:] - GV[:, :-1] + NTH / 2, NTH) - NTH / 2
        dvy = torch.remainder(GV[1:] - GV[:-1] + NTH / 2, NTH) - NTH / 2
        okx, oky = ok[:, 1:] & ok[:, :-1], ok[1:] & ok[:-1]          # only between two covered pixels
        fpx = F.pad(torch.where(okx, torch.hypot(GU[:, 1:] - GU[:, :-1], dvx), torch.zeros_like(dvx)), (0, 1))
        fpy = F.pad(torch.where(oky, torch.hypot(GU[1:] - GU[:-1], dvy), torch.zeros_like(dvy)), (0, 0, 0, 1))
        fp = torch.maximum(fpx, fpy)[ok].clamp(min=1e-3, max=4096)
        level = torch.log2(fp).clamp(min=0)
    pyr = pyramid(params["tex"], 14)
    col = sample_mip(pyr, gu, gv, level, wrap_v=True)
    if CAP_N:
        capm = (X[:, 0] > cap_x) & (Nw[ok][:, 0].abs() > 0.5)
        if capm.any():
            cu = (X[capm, 1] + 0.2) / du - 0.5
            cv = (X[capm, 2] + 0.2) / du - 0.5
            with torch.no_grad():               # footprint on the cap plane in cap texels
                lc = torch.log2((D[ok][capm] / fx / Nw[ok][capm, 0].abs().clamp(min=0.2) / du).clamp(min=1))
            col_c = sample_mip(pyramid(params["cap"], 12), cu, cv, lc, wrap_v=False)
            full = torch.zeros_like(col)
            full[capm] = col_c
            col = torch.where(capm[:, None], full, col)
    if lamp_on:
        Xc = torch.stack([RX[ok] * D[ok], RY[ok] * D[ok], D[ok]], -1)              # camera coordinates
        Lv = params["lamp_t"] - Xc
        r = Lv.norm(dim=-1).clamp(min=1e-3)
        Nc = Nw[ok] @ Rd.T
        cos = ((Nc * Lv).sum(-1) / r).abs().clamp(min=0.05)
        if r_ref is None:
            r_ref = float(D[ok].median())
        geo = cos / (r / r_ref) ** 2
        M = torch.exp(F.interpolate(params["logM"], size=(h, w), mode="bilinear", align_corners=True))[0, 0][ok]
        col = col * (M * geo)[:, None]
    if a.ambient:
        ga = (u - U0) / (U1 - U0) * params["amb"].shape[-1] - 0.5
        gb = (th + math.pi) / (2 * math.pi) * params["amb"].shape[-2] - 0.5
        col = col + sample_mip([torch.sigmoid(params["amb"])], ga, gb, None, wrap_v=True)
    img = torch.zeros(h, w, 3, device=dev)
    img[ok] = col
    return img, ok


# lamp map start: M x cos / r^2 = 1 on average at every pixel (train_gs.py --lamp)
if lamp_on:
    with torch.no_grad():
        acc, cnt = torch.zeros(h, w, device=dev), torch.zeros(h, w, device=dev)
        for n in list(train)[::8]:
            D, Nw, ok = surface_dn(views_all[int(n)])
            R = views_all[int(n)][:3, :3]
            Xc = torch.stack([RX * D, RY * D, D], -1)
            r = (-Xc).norm(dim=-1).clamp(min=1e-3)
            cos = (((Nw @ R.T) * -Xc).sum(-1) / r).abs().clamp(min=0.05)
            if r_ref is None:
                r_ref = float(D[ok].median())
            acc[ok] += torch.log(cos / (r / r_ref) ** 2)[ok]
            cnt[ok] += 1
        lg = torch.where(cnt > 0, acc / cnt.clamp(min=1), torch.zeros_like(acc))
        params["logM"][:] = -F.adaptive_avg_pool2d(lg[None, None], (GY, GX))
    log(f"lamp model: r_ref {r_ref * 1000:.0f} mm, M on a {GX}x{GY} grid started from the mean geometry factor")

# ------------------------------------------------------------------------------ training ----------------------
if a.eval_only:
    saved = torch.load(os.path.join(a.out, "tex.pt"))
    with torch.no_grad():
        for k in params:
            params[k][:] = saved[k].to(dev)
    r_ref = saved["r_ref"]
optim = {k: torch.optim.Adam([p], lr=lrs[k], eps=1e-15) for k, p in params.items()}
torch.cuda.reset_peak_memory_stats()
t_train = time.time()
seq, done = [], 0
for it in range(1, 0 if a.eval_only else a.iters + 1):
    if not seq:
        seq = list(np.random.permutation(train))
    n = int(seq.pop())
    gt = IM[n].to(dev, non_blocking=True).float() / 255
    img, ok = render(n, views_all[n], pose_idx=n)
    ren = torch.exp(params["gain"][n]) * img + params["off"][n]
    m = ok
    l1 = (ren - gt).abs()[m].mean()
    ssim = fused_ssim(torch.where(m[..., None], ren, gt).permute(2, 0, 1)[None].contiguous(),
                      gt.permute(2, 0, 1)[None].contiguous())
    loss = (1 - a.dssim) * l1 + a.dssim * (1 - ssim)
    if a.disp_mm > 0:                           # smooth displacement (second differences), small
        hdm = a.disp_max_mm * torch.tanh(params["disp"][0, 0])            # mm
        lap = (hdm[:, 2:] - 2 * hdm[:, 1:-1] + hdm[:, :-2]).pow(2).mean() +               (torch.roll(hdm, 1, 0) - 2 * hdm + torch.roll(hdm, -1, 0)).pow(2).mean()
        loss = loss + 1e-3 * lap
    if a.pose_opt:                              # weak prior: corrections of a few mm / mrad
        loss = loss + 1e-4 * ((params["pose_c"][n].norm() / 0.002) ** 2 + (params["pose_w"][n].norm() / 0.002) ** 2)
    loss.backward()
    for opt in optim.values():
        opt.step()
        opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        if a.tex_clamp:
            params["tex"].clamp_(0, 1.5)
            if CAP_N:
                params["cap"].clamp_(0, 1.5)
        else:
            params["tex"].clamp_(-12, 4)
            if CAP_N:
                params["cap"].clamp_(-12, 4)
    if it == a.iters // 2:                      # texture learning rate x 0.3 for the second half
        for k in ("tex", "cap"):
            if k in optim:
                for g in optim[k].param_groups:
                    g["lr"] = a.lr * 0.3
    done = it
    if it % 500 == 0 or it == 1:
        with torch.no_grad():
            ps = -10 * torch.log10(((ren - gt) ** 2)[m].mean()).item()
        extra = ""
        if a.pose_opt:
            extra = (f" | pose corr per frame median {params['pose_c'].norm(dim=1).median().item() * 1000:.2f} mm "
                     f"{params['pose_w'].norm(dim=1).median().item() * 1000:.2f} mrad, shared "
                     f"{(params['pose_cg'] * 1000).round(decimals=2).tolist()} mm "
                     f"{(params['pose_wg'] * 1000).round(decimals=2).tolist()} mrad")
        log(f"iter {it:6d} loss {loss.item():.4f} L1 {l1.item():.4f} SSIM {ssim.item():.3f} PSNR {ps:.2f}"
            f"{extra} | {it / (time.time() - t_train):.1f} it/s | peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
train_s = time.time() - t_train
peak = torch.cuda.max_memory_allocated() / 1e9
if not a.eval_only:
    torch.save({k: v.detach().cpu() for k, v in params.items()} | dict(r_ref=r_ref, U0=U0, U1=U1, du=du, dth=dth,
               NU=NU, NTH=NTH, cap_x=cap_x, R_mean=R_mean), os.path.join(a.out, "tex.pt"))
    if a.disp_mm > 0:
        hmm = (a.disp_max_mm * torch.tanh(params["disp"][0, 0])).detach().cpu().numpy()
        np.save(os.path.join(a.out, "disp_mm.npy"), hmm)
        log(f"displacement: |h| median {np.median(np.abs(hmm)):.2f} mm, p95 {np.percentile(np.abs(hmm), 95):.2f} mm, "
            f"range {hmm.min():.2f}..{hmm.max():.2f} mm (grid {hmm.shape[0]} x {hmm.shape[1]})")
    log(f"trained {done} iterations in {train_s / 60:.1f} min, peak {peak:.2f} GB"
        + (f"; lamp offset {(params['lamp_t'] * 1000).round().tolist()} mm" if lamp_on else ""))
if a.export or not a.eval_only:
    # the texture's overall scale is arbitrary (the per-frame gains absorb it): written as 8-bit with its
    # 99.5th percentile at 250 (the scale is kept in texture.json); before X4c it was clipped at 1.0 (22 % white)
    tl = lin(params["tex"][0]).detach()
    seen = (params["tex"][0].detach() - TEX0).abs().amax(0) > 1e-4
    scale = float(torch.quantile(tl.mean(0)[seen][torch.randperm(int(seen.sum()), device=dev)[:2000000]], 0.995)) / 250 * 255
    t8 = (tl / scale).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
    cv2.imwrite(os.path.join(a.out, "texture.png"), t8.round().astype(np.uint8)[:, :, ::-1])
    if CAP_N:
        c8 = (lin(params["cap"][0]).detach() / scale).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
        cv2.imwrite(os.path.join(a.out, "cap.png"), c8.round().astype(np.uint8)[:, :, ::-1])
    json.dump(dict(U0=U0, U1=U1, du=du, dth=dth, NU=NU, NTH=NTH, cap_x=cap_x, cap_half=0.2, R_mean=R_mean, png_scale=scale,
                   inpipe=bool(inpipe), note="texture.png: column c covers u = U0 + c du, row r covers theta = -pi + r dth (rad); "
                   "cap.png: column = (y + 0.2) / du, row = (z + 0.2) / du"),
              open(os.path.join(a.out, "texture.json"), "w"), indent=1)

# ------------------------------------------------------------------------------ scoring (ruler v2) ------------
train_ids, train_t = d["train_ids"], d["train_t"]


def neighbours(fid, ft):
    """the two training frames around a held-out frame (by time for InPipe, by frame number otherwise) and weight"""
    key = train_t if inpipe else train_ids
    x = ft if inpipe else fid
    tr = np.sort(train)
    kk = key[tr]
    j = int(np.searchsorted(kk, x))
    i0, i1 = tr[max(j - 1, 0)], tr[min(j, len(tr) - 1)]
    t_ = 0.0 if key[i1] == key[i0] else float((x - key[i0]) / (key[i1] - key[i0]))
    return (int(i0), int(i1)), min(max(t_, 0.0), 1.0)


@torch.no_grad()
def score(tag, imgs, nxt, views, B, Bm, ids, pos, times, own_idx=None):
    rows = []
    for k in range(len(ids)):
        view = torch.from_numpy(views[k]).to(dev)
        # any frame not trained on (held-out and block) takes the pose correction interpolated between the
        # training frames around it (block frames: the frames before and after the 10 cm gap)
        pi, tw = neighbours(int(ids[k]), int(times[k]))
        img, ok = render(None, view, pose_idx=pi, held_w=tw)
        G = img.clamp(min=0).mul(255).cpu().numpy()
        real = imgs[k].astype(np.float32)
        m = Bm[k] & valid_raw
        row = dict(tag=tag, frame=int(ids[k]), pos=float(pos[k]), coverage=float(m.mean()),
                   cov_G=float(ok.float().mean().item()), cov_B=float(Bm[k].mean()))
        nal = im.linear_match(im.align_to(nxt[k].astype(np.float32), real), real, m)
        den = cv2.fastNlMeansDenoisingColored(np.ascontiguousarray(imgs[k]), None, 5, 5, 7, 21).astype(np.float32)
        mats = {}
        for name, x0 in (("G", G), ("B", B[k].astype(np.float32)), ("Nal", nal), ("D", den)):
            x = im.linear_match(x0, real, m)
            mats[name] = x
            for key, val in im.all_metrics(x, real, m).items():
                row[f"{name}_{key}"] = val
            for key, val in im.ruler2(x, real, nal, m).items():
                row[f"{name}_{key}"] = val
        rows.append(row)
        if a.save and k % a.save == 0:
            trip = np.hstack([real, mats["B"], mats["G"]]).clip(0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(a.out, "renders", f"{tag}_{int(ids[k]):06d}.jpg"), trip[:, :, ::-1],
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
        log(f"  {tag} {k + 1}/{len(ids)} frame {int(ids[k])}: PSNR TX {row['G_psnr']:.2f} B {row['B_psnr']:.2f} "
            f"Nal {row['Nal_psnr']:.2f} | fid fine TX {row['G_fid_fine']:.3f} B {row['B_fid_fine']:.3f}")
    return rows


rng = d["block_ranges"]
in_block = lambda p: bool(np.any([(p >= lo) & (p < hi) for lo, hi in rng]))  # noqa: E731
held_t = d["held_t"] if "held_t" in d.files else np.zeros(len(d["held_ids"]), np.int64)
rows = score("held", d["held_images"], d["held_next"], d["held_viewmats"], d["held_B"], d["held_mask"], d["held_ids"],
             d["held_pos"], held_t)
for r in rows:
    r["in_block"] = in_block(r["pos"])
bi = block_idx
rows_b = score("block", raw_train[bi], d["block_next"], d["block_viewmats"], d["block_B"], d["block_mask"],
               d["block_ids"], d["train_pos"][bi], train_t[bi], own_idx=bi)
for r in rows_b:
    r["in_block"] = True
keys = [k for k in rows[0] if k not in ("tag", "frame", "in_block")]
with open(os.path.join(a.out, "per_frame_r2.csv"), "w", newline="", encoding="utf-8") as fh:
    wr = csv.DictWriter(fh, ["tag", "frame", "in_block"] + keys)
    wr.writeheader()
    wr.writerows(rows + rows_b)
med = lambda rs: {k: float(np.nanmedian([r[k] for r in rs])) for k in keys} if rs else {}  # noqa: E731
summary = dict(scene=a.scene, args=vars(a), model="TX texture", texture=[NTH, NU], train_frames=int(len(train)),
               iterations_done=done, train_minutes=round(train_s / 60, 2), peak_GB=round(peak, 2),
               held_all=med(rows), held_outside_blocks=med([r for r in rows if not r["in_block"]]), block=med(rows_b))
json.dump(summary, open(os.path.join(a.out, "summary_r2.json"), "w"), indent=1)
M = ("psnr", "lpips", "shift_px", "fid_fine", "fid_mid", "fid_coarse", "en_fine", "en_mid", "mse_low")
for title, s in (("in-sequence held-out, outside the blocks", summary["held_outside_blocks"]), ("block frames", summary["block"])):
    log(f"\n{title} (median, ruler v2):")
    for tag, name in (("G", "TX"), ("B", "lampmap B"), ("Nal", "next frame aligned"), ("D", "frame denoised")):
        log(f"  {name:18s} " + "  ".join(f"{k} {s[f'{tag}_{k}']:.3f}" for k in M))
log("done")
