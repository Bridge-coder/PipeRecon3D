"""Gaussian model of a pipe video, trained and scored against lampmap's B (docs/08, docs/09). Runs in the pipe3d WSL distro.
Without options this is exactly step 1's G0 (train_g0.py, kept unchanged for reproducing docs/08); options add one
change at a time:
  --anchor  (docs/09, G0-A) hold the Gaussians on the measured surface (SCENE_DIR/surface.npz from export_surface.py):
            starting points snapped onto the nearest surface triangle (dropped if farther than --anchor-snap-mm),
            Gaussians made thin discs lying in the surface (rotation fixed to the surface normal, thickness <= --thin-mm),
            centres kept within +-(--anchor-clamp-mm) of their snapped point along the normal (they may slide along it),
            and the depth term replaced by a dense one: relative L1 between the rendered depth and the depth of the
            measured surface over every pixel it covers (weight --anchor-depth-w)
  --lamp    (docs/09, step 2; needs --anchor) the robot's lamp: the rendered colour is multiplied per pixel by
            M(pixel) x cos(incidence) / (r / r_ref)^2, with r and the incidence angle from the measured surface's depth
            (normals: the surface's triangle normals averaged over ~--lamp-smooth-mm, rendered as a normal map -
            the LiDAR height field has +-1-2 mm oblique stripes whose raw normals put stripes into the shading,
            docs/09 2.1), the lamp at a learned offset from the camera centre (start 0),
            M a learned camera-fixed map (48 x 36 grid, lamp pattern x vignetting; started so that M x cos / r^2 is 1
            on average at every pixel, i.e. the first render equals the one without the lamp), plus a learned gain and
            offset per training frame and colour channel (exposure). Scored like everything else: the render x lamp
            factor, then each frame's own gain and offset fitted (imgmetrics.linear_match).
  --lamp-ambient  (docs/09 2.3) colour = gain x (lamp factor x Gaussian colour + ambient) + offset, the ambient a
            second, view-independent colour per Gaussian (start ~0.01) that the lamp factor does not scale: light from
            the room entering through the slot and the open end of the InPipe pipe follows the world, not the robot.

usage: TORCH_HOME=/mnt/d/pipe3d_gs/torch_cache /opt/gsenv/bin/python pipesplat/train_gs.py SCENE_DIR OUT_DIR
           [--iters 30000] [--adc] [--anchor] [--no-blocks] [--save 4] [--seed 0] [--eval-only] [--alphas 0.99 0.5]
       SCENE_DIR  output of export_g0.py (scene.npz)
       OUT_DIR    train.log, model.pt, summary.json, per_frame.csv, renders/*.jpg (video | B | G0)

What is Gaussian-LIC2 here (paper section in brackets) and what is not:
  same:  one Gaussian per geometry point, colour from projecting it into a training frame (6.1), scale from its depth
         over the focal length (7.1.1), opacity 0.1, identity rotation, full 3rd-order SH; loss
         0.8 L1 + 0.2 D-SSIM + 0.005 |D/O - D_s| on pixels with geometry depth (eq. 20-22); undistorted pinhole
         frames; no adaptive density control by default (7.1, appendix: ADC hurts their incremental mapping)
  ours:  poses = our SfM / fused poses (as their mapping evaluation, 9.4.1, uses given poses); geometry = OpenMVS
         dense cloud (vision) or the refined LiDAR wall points, depth from the 5 scans nearest to each frame (InPipe)
  differences, each for a stated reason:
    - offline, all training frames at once, 30k iterations with the vanilla 3DGS position-lr decay (their system is
      online; their offline comparison with LetsGo also trains 30k iterations)
    - scale = max(depth / focal, half the distance to the nearest point): our points are a few mm apart, and the
      smoke test (docs/07 4.7) showed the depth/focal rule alone leaves holes near the camera
    - the undistorted frames keep the whole raw picture (a larger pinhole canvas) instead of cropping it
    - no LiDAR blind-area compensation (their 6.2): coverage is measured here and treated in step 5
    - --adc turns on gsplat's default densification (offline it is the standard 3DGS choice) as a second run
  Points never seen by a training frame are dropped (they would never be optimised).
The k1 fold: with k1 < 0, x_d = x_u (1 + k1 r^2) turns back beyond r_u = 1/sqrt(-3 k1); the canvas stays inside 95 %.
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
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from fused_ssim import fused_ssim
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "lampmap"))
import imgmetrics as im  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("scene"), ap.add_argument("out")
ap.add_argument("--iters", type=int, default=30000), ap.add_argument("--adc", action="store_true")
ap.add_argument("--no-blocks", action="store_true", help="train on the block frames too (block test then meaningless)")
ap.add_argument("--save", type=int, default=4), ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--dssim", type=float, default=0.2), ap.add_argument("--depth-w", type=float, default=0.005)
ap.add_argument("--eval-only", action="store_true", help="load OUT_DIR/model.pt and only score it")
ap.add_argument("--alpha-min", type=float, default=0.99, help="pixels scored: model opacity above this (and B's mask)")
ap.add_argument("--suffix", default="", help="appended to summary/per_frame file names (e.g. _alpha050)")
ap.add_argument("--alphas", type=float, nargs="+", default=None,
                help="score at several opacity thresholds in one go (default 0.99 and 0.5; files _alpha050 etc.)")
ap.add_argument("--ruler2", action="store_true",
                help="score with ruler v2 (docs/10 section 2): every model on the same pixels (B's mask, no opacity "
                     "threshold), references Nal (next frame aligned by optical flow) and D (the frame denoised), "
                     "band metrics imgmetrics.ruler2; use with --eval-only --suffix _r2")
ap.add_argument("--scale-cap", type=float, default=0.0,
                help="keep every Gaussian's scale (in-plane scales with --anchor) at most this many times its starting "
                     "scale (0 = off, as docs/08-09). Without ADC nothing else limits it, and a few per cent of the "
                     "Gaussians grew to 5 mm - metres, opaque, and formed most pixels (docs/10 X2)")
ap.add_argument("--raw", action="store_true",
                help="train and render in the raw (distorted) frame through 3DGUT with the radial k1 instead of an "
                     "undistorted pinhole canvas: no resampling of the training frames and of the renders (docs/10 X3)")
ap.add_argument("--smoke", action="store_true", help="code check: train, save, no scoring")
ap.add_argument("--pose-opt", action="store_true",
                help="learn pose corrections (docs/10 X5): one rigid correction shared by all frames (rotation and "
                     "shift in camera axes = a camera-LiDAR / camera-mesh extrinsic error) plus a small one per training "
                     "frame (weak prior); frames not trained on take the correction interpolated between their two "
                     "neighbouring training frames")
ap.add_argument("--test-pose-iters", type=int, default=0,
                help="evaluation only (docs/10 X6c): before scoring, register each scored frame's own pose (6 numbers, "
                     "model frozen) to that frame by optical flow, this many rounds (3) - separates 'the model is wrong' "
                     "from 'the test frame's pose is wrong'. A separate score (suffix _r2tp), never mixed with the strict one")
ap.add_argument("--pose-start", type=int, default=3000,
                help="pose corrections are learned only after this many iterations (early gradients are unreliable: "
                     "starting at once, the shared correction ran off by 4 cm / 1.6 deg in 500 iterations)")
ap.add_argument("--anchor", action="store_true", help="hold the Gaussians on the measured surface (docs/09)")
ap.add_argument("--points-every", type=int, default=1,
                help="start from every N-th geometry point (memory: lab A's dense cloud has 4.1 M points)")
ap.add_argument("--surface", default="surface.npz", help="measured-surface file in SCENE_DIR (export_surface.py)")
ap.add_argument("--anchor-depth-w", type=float, default=1.0)
ap.add_argument("--anchor-clamp-mm", type=float, default=2.0)
ap.add_argument("--anchor-snap-mm", type=float, default=10.0)
ap.add_argument("--thin-mm", type=float, default=0.05)
ap.add_argument("--lamp", action="store_true", help="near-field lamp model (docs/09; needs --anchor)")
ap.add_argument("--lamp-ambient", action="store_true",
                help="add light that does not come from the lamp (room light through the slot / exit): per Gaussian a "
                     "view-independent colour added after the lamp factor")
ap.add_argument("--lamp-smooth-mm", type=float, default=20.0,
                help="surface normals for the lamp's incidence angle averaged over about this radius")
ap.add_argument("--baseline", choices=("3dgs", "mcmc", "2dgs"), default=None,
                help="published methods on the same frames, canvas and scoring (docs/11): "
                     "3dgs = 3D Gaussian Splatting (Kerbl et al. 2023): ADC (gsplat DefaultStrategy), start scale = RMS "
                     "distance to the 3 nearest points, opacity 0.1, SH degree raised every 1000 iterations, no depth; "
                     "mcmc = 3DGS as MCMC (Kheradmand et al. 2024): gsplat MCMCStrategy (relocation + noise, as many "
                     "Gaussians as starting points), opacity 0.5, scale 0.1 x that distance, opacity / scale "
                     "regularisation 0.01; 2dgs = 2D Gaussian Splatting (Huang et al. 2024): surfels "
                     "(rasterization_2dgs), ADC on its own gradient, normal consistency 0.05 from 7k, distortion 0.01 "
                     "from 3k (gsplat's simple_trainer_2dgs defaults). All start from the same dense MVS cloud as ours "
                     "(a stronger start than the sparse SfM points of the papers)")
ap.add_argument("--normal-start", type=int, default=7000, help="--baseline 2dgs: normal consistency from this iteration")
ap.add_argument("--dist-start", type=int, default=3000, help="--baseline 2dgs: depth distortion from this iteration")
ap.add_argument("--opaque", action="store_true",
                help="with --anchor: the surface discs are opaque (opacity fixed at 0.99, not learned) - one layer of "
                     "surfels instead of many semi-transparent ones (docs/10 5.5: median opacity 0.14 looks speckled "
                     "and see-through from outside the pipe)")
a = ap.parse_args()
if a.baseline:
    assert not (a.anchor or a.lamp or a.raw or a.adc or a.scale_cap), "--baseline is a method of its own"
    a.adc, a.depth_w = a.baseline in ("3dgs", "2dgs"), 0.0
assert not a.opaque or a.anchor, "--opaque is for the anchored surface discs"
if a.baseline == "2dgs":
    # this gsplat build (1.6.0 @512d366) hands the 2DGS projection backward a non-contiguous depth gradient
    # ("grad.depths must be contiguous") for every depth render mode; make its inputs contiguous
    import gsplat.cuda._wrapper as _gw
    _lazy = _gw._make_lazy_cuda_func

    def _lazy_contig(name):
        f = _lazy(name)
        if name != "projection_2dgs_fused_bwd":
            return f
        return lambda *args: f(*[x.contiguous() if torch.is_tensor(x) else x for x in args])
    _gw._make_lazy_cuda_func = _lazy_contig
assert not a.lamp or a.anchor, "--lamp uses the measured-surface depth of --anchor"
assert not a.lamp_ambient or a.lamp, "--lamp-ambient extends --lamp"
assert not (a.anchor and a.adc), "--anchor keeps one Gaussian per snapped point; it is not combined with --adc"
assert not (a.scale_cap and a.adc), "--scale-cap is per starting Gaussian; ADC changes their number"
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


# ------------------------------------------------------------------------------ camera canvas ---------------
def undistort_grid(xd, yd, k1):
    """inverse of x_d = x_u (1 + k1 r_u^2) by iteration (render_views.undistort_grid, copied: that module needs the
    Windows paths); NaN beyond 95 % of the radius the model reaches"""
    out = (xd ** 2 + yd ** 2 >= (0.95 * 2 / (3 * np.sqrt(-3 * k1))) ** 2) if k1 < 0 else np.zeros(xd.shape, bool)
    xd, yd = np.where(out, 0.0, xd), np.where(out, 0.0, yd)
    xu, yu = xd.copy(), yd.copy()
    for _ in range(20):
        g = 1 + k1 * (xu ** 2 + yu ** 2)
        xu, yu = xd / g, yd / g
    xu[out], yu[out] = np.nan, np.nan
    return xu, yu


UT = {}                                         # extra gsplat arguments of the camera model (--raw: 3DGUT)
if k1 == 0.0:
    CW, CH, ccx, ccy = w, h, cx, cy
    valid_raw = np.ones((h, w), bool)
    to_canvas = to_raw = None
    valid_canvas = np.ones((h, w), bool)
    RXY = np.meshgrid((np.arange(w) - cx) / fx, (np.arange(h) - cy) / fy)
elif a.raw:                                     # the raw frame itself, distortion inside the rasteriser
    XU, YU = undistort_grid(*np.meshgrid((np.arange(w) - cx) / fx, (np.arange(h) - cy) / fy), k1)
    valid_raw = np.isfinite(XU)
    CW, CH, ccx, ccy = w, h, cx, cy
    to_canvas = to_raw = None
    valid_canvas = valid_raw
    RXY = (np.nan_to_num(XU), np.nan_to_num(YU))
    UT = dict(with_ut=True, with_eval3d=True,
              radial_coeffs=torch.tensor([[k1, 0, 0, 0, 0, 0]], device=dev, dtype=torch.float32))
else:
    XU, YU = undistort_grid(*np.meshgrid((np.arange(w) - cx) / fx, (np.arange(h) - cy) / fy), k1)
    valid_raw = np.isfinite(XU)
    CW, CH = 2 * int(np.ceil(np.nanmax(np.abs(XU)) * fx)) + 4, 2 * int(np.ceil(np.nanmax(np.abs(YU)) * fy)) + 4
    ccx, ccy = CW / 2 - 0.5, CH / 2 - 0.5
    to_raw = (np.nan_to_num(ccx + XU * fx, nan=-10).astype(np.float32), np.nan_to_num(ccy + YU * fy, nan=-10).astype(np.float32))
    xu, yu = np.meshgrid((np.arange(CW) - ccx) / fx, (np.arange(CH) - ccy) / fy)
    g = 1 + k1 * (xu ** 2 + yu ** 2)
    ud, vd = (cx + fx * xu * g).astype(np.float32), (cy + fy * yu * g).astype(np.float32)
    valid_canvas = ((xu ** 2 + yu ** 2) < (0.95 / np.sqrt(-3 * k1)) ** 2) & (ud >= 0) & (ud <= w - 1) & (vd >= 0) & (vd <= h - 1)
    to_canvas = (ud, vd)
    RXY = (xu, yu)
Kc = torch.tensor([[fx, 0, ccx + 0.5], [0, fy, ccy + 0.5], [0, 0, 1]], device=dev, dtype=torch.float32)  # gsplat: centres at +0.5
VALID = torch.from_numpy(valid_canvas).to(dev)
log(f"raw {w}x{h}, k1 {k1}; training canvas {CW}x{CH} ({'raw frame, 3DGUT' if UT else 'pinhole'}), "
    f"valid {valid_canvas.mean():.1%} of it")
RX, RY = (torch.from_numpy(np.ascontiguousarray(v, np.float32)).to(dev) for v in RXY)  # ray x/z, y/z per pixel


def project(c):
    """pixel (column, row) on the training canvas of camera-space points c, and whether they land inside"""
    z = c[:, 2].clamp(min=1e-6)
    xn, yn = c[:, 0] / z, c[:, 1] / z
    ok = c[:, 2] > 0.01
    if UT:                                      # x_d = x_u (1 + k1 r_u^2), only before the fold
        r2 = xn ** 2 + yn ** 2
        ok &= r2 < 0.95 ** 2 / (-3 * k1)
        xn, yn = xn * (1 + k1 * r2), yn * (1 + k1 * r2)
    return torch.round(ccx + fx * xn).long(), torch.round(ccy + fy * yn).long(), ok


def canvas(img):
    return img if to_canvas is None else cv2.remap(img, to_canvas[0], to_canvas[1], cv2.INTER_LINEAR, borderValue=0)


# ------------------------------------------------------------------------------ frames, depth ---------------
views_all = torch.from_numpy(d["train_viewmats"]).to(dev)
eyes = d["train_eyes"]
block_idx = d["block_idx"]
train = np.setdiff1d(np.arange(len(views_all)), [] if a.no_blocks else block_idx)
raw_train = d["train_images"]
IM = {int(i): torch.from_numpy(canvas(raw_train[i])) for i in train}
if sum(v.numel() for v in IM.values()) < 1.0e9:  # pinned for fast upload; WSL refuses ~1.2 GB of pinned memory
    IM = {k: v.pin_memory() for k, v in IM.items()}   # (Pipe1), and the failed CUDA call breaks the next one
else:
    log("training frames not pinned (more than 1 GB)")
log(f"{len(train)} training frames (left out: {0 if a.no_blocks else len(block_idx)} block frames in "
    f"{d['block_ranges'].round(3).tolist()} m)")

DP = torch.from_numpy(d["depth_points"]).to(dev)
if inpipe:                                     # LiDAR: the 5 scans nearest in time to a frame (Gaussian-LIC2 6.1)
    scan = torch.from_numpy(d["depth_scan"].astype(np.int64)).to(dev)
    order = torch.argsort(scan)
    DP, scan = DP[order], scan[order]
    starts = torch.searchsorted(scan, torch.arange(len(d["scan_t"]) + 1, device=dev))
    scan_t = d["scan_t"]

    def depth_points(t):
        near = np.argsort(np.abs(scan_t - t))[:5]
        return torch.cat([DP[starts[s]:starts[s + 1]] for s in near])
else:
    def depth_points(t):
        return DP


def depth_map(view, t):
    """sparse geometry depth on the canvas (0 = none), nearest point per pixel"""
    p = depth_points(t)
    c = p @ view[:3, :3].T + view[:3, 3]
    i, j, k = project(c)
    z = c[:, 2]
    k &= (i >= 0) & (i < CW) & (j >= 0) & (j < CH)
    D = torch.full((CH * CW,), float("inf"), device=dev)
    D.scatter_reduce_(0, j[k] * CW + i[k], z[k], reduce="amin")
    D[torch.isinf(D)] = 0
    return D.view(CH, CW)


def quat_z_to(n):
    """unit quaternions (wxyz) rotating the z axis onto the unit vectors n"""
    z = torch.tensor([0.0, 0.0, 1.0], device=n.device)
    axis = torch.cross(z.expand_as(n), n, dim=1)
    s = axis.norm(dim=1, keepdim=True)
    c = n[:, 2:3].clamp(-1, 1)
    ang = torch.atan2(s, c)
    axis = torch.where(s > 1e-8, axis / s.clamp(min=1e-12), torch.tensor([1.0, 0, 0], device=n.device).expand_as(n))
    return torch.cat([torch.cos(ang / 2), axis * torch.sin(ang / 2)], dim=1)


if a.anchor:
    sf = np.load(os.path.join(a.scene, a.surface))
    S_c = torch.from_numpy(sf["centres"]).to(dev)
    S_n = torch.from_numpy(sf["normals"]).to(dev)
    S_q = quat_z_to(S_n)
    S_s = torch.log(torch.stack([0.6 * torch.from_numpy(sf["size"]).to(dev)] * 2 + [torch.full((len(S_c),), 1e-5, device=dev)], 1))
    S_o = torch.full((len(S_c),), 0.999, device=dev)
    S_col = torch.zeros(len(S_c), 3, device=dev)

    @torch.no_grad()
    def surface_depth(view):
        """depth of the measured surface on the canvas (0 where it does not cover the pixel)"""
        out, al, _ = gsplat.rasterization(S_c, S_q, torch.exp(S_s), S_o, S_col, view[None], Kc[None], CW, CH,
                                          packed=False, render_mode="ED", near_plane=0.01, **UT)
        dep = out[0, ..., 0]
        return torch.where(al[0, ..., 0] > 0.5, dep, torch.zeros_like(dep))

    t_s = time.time()
    DA = {int(n): surface_depth(views_all[n]).half() for n in train}
    chk = []
    for n in list(train)[:: max(1, len(train) // 20)]:
        Da, Ds = DA[int(n)].float(), depth_map(views_all[n], d["train_t"][n])
        m = (Da > 0) & (Ds > 0) & VALID
        chk.append(((Da - Ds).abs() / Ds)[m].median().item())
    log(f"measured surface: {len(S_c):,} triangles; dense depth for {len(DA)} training frames in {time.time() - t_s:.0f} s; "
        f"median relative difference to the step-1 geometry depth {np.median(chk):.2%} (consistency check)")

if a.lamp:
    GY, GX = 36, 48
    r_ref = float(np.median([DA[int(n)][DA[int(n)] > 0].float().median().item() for n in list(train)[::10]]))
    lamp = torch.nn.ParameterDict({
        "logM": torch.nn.Parameter(torch.zeros(1, 1, GY, GX, device=dev)),
        "t": torch.nn.Parameter(torch.zeros(3, device=dev)),                     # lamp in camera coordinates, m
        "gain": torch.nn.Parameter(torch.zeros(len(views_all), 3, device=dev)),  # log gain per frame and channel
        "off": torch.nn.Parameter(torch.zeros(len(views_all), 3, device=dev)),
    })

    def smooth_normals(c, n, radius):
        """structure-tensor average of the triangle normals within about radius (voxels of radius/2, 27 neighbours);
        sign kept as the triangle's own"""
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
        m = n.clone()                     # power iteration from the triangle's own normal
        for _ in range(15):
            m = torch.bmm(A, m[:, :, None])[:, :, 0]
            m = m / m.norm(dim=1, keepdim=True).clamp(min=1e-12)
        return torch.where((m * n).sum(1, keepdim=True) < 0, -m, m)

    S_ns = smooth_normals(S_c, S_n, a.lamp_smooth_mm / 1000)
    ang = torch.rad2deg(torch.acos((S_ns * S_n).sum(1).clamp(-1, 1)))
    log(f"lamp normals averaged over ~{a.lamp_smooth_mm:.0f} mm: change from the raw triangle normals median "
        f"{ang.median().item():.1f} deg, p95 {ang.quantile(0.95).item() if len(ang) < 16_000_000 else float('nan'):.1f} deg")

    @torch.no_grad()
    def surface_dn(view):
        """depth and smoothed normal (camera coordinates) of the measured surface on the canvas, and where it covers"""
        out, al, _ = gsplat.rasterization(S_c, S_q, torch.exp(S_s), S_o, S_ns @ view[:3, :3].T, view[None], Kc[None],
                                          CW, CH, packed=False, render_mode="RGB+ED", near_plane=0.01, **UT)
        a0 = al[0, ..., 0]
        ok = a0 > 0.5
        D = torch.where(ok, out[0, ..., 3], torch.zeros_like(a0))
        N = out[0, ..., :3] / a0[..., None].clamp(min=1e-6)
        return D, N / N.norm(dim=-1, keepdim=True).clamp(min=1e-12), ok

    def lamp_geometry(view):
        """cos(incidence) / (r / r_ref)^2 per canvas pixel; 1 where the measured surface does not cover the pixel"""
        D, N, ok = surface_dn(view)
        X = torch.stack([RX * D, RY * D, D], -1)
        Lv = lamp["t"] - X
        r = Lv.norm(dim=-1).clamp(min=1e-3)
        cos = ((N * Lv).sum(-1) / r).abs().clamp(min=0.05)
        return torch.where(ok, cos / (r / r_ref) ** 2, torch.ones_like(D))

    def lamp_factor(view):
        M = torch.exp(F.interpolate(lamp["logM"], size=(CH, CW), mode="bilinear", align_corners=True))[0, 0]
        return M * lamp_geometry(view)

    with torch.no_grad():                         # start: M x cos / r^2 = 1 on average at every pixel
        lg = torch.stack([torch.log(lamp_geometry(views_all[int(n)])) for n in list(train)[::8]]).mean(0)
        lamp["logM"][:] = -F.adaptive_avg_pool2d(lg[None, None], (GY, GX))
    log(f"lamp model: r_ref {r_ref * 1000:.0f} mm, M on a {GX}x{GY} grid started from the mean geometry factor")

# ------------------------------------------------------------------------------ Gaussians --------------------
pts = torch.from_numpy(d["points"][::a.points_every]).to(dev)
if a.anchor:                                    # snap the starting points onto the measured surface
    dist, idx = cKDTree(sf["centres"]).query(d["points"][::a.points_every], k=1)
    idx = torch.from_numpy(idx).to(dev)
    n0 = S_n[idx]
    off0 = ((pts - S_c[idx]) * n0).sum(1, keepdim=True)
    near = torch.from_numpy(dist <= a.anchor_snap_mm / 1000).to(dev)
    log(f"anchor: {int(near.sum()):,} of {len(pts):,} points within {a.anchor_snap_mm} mm of the surface; "
        f"offset along the normal median {off0.abs().median().item() * 1000:.2f} mm")
    pts, n0 = (pts - off0 * n0)[near], n0[near]
best_z = torch.full((len(pts),), float("inf"), device=dev)
rgb = torch.zeros(len(pts), 3, device=dev)
for n in train:                                 # colour = the training frame that sees the point closest
    v = views_all[n]
    c = pts @ v[:3, :3].T + v[:3, 3]
    z = c[:, 2]
    i, j, ok = project(c)
    ok &= (i >= 8) & (i < CW - 8) & (j >= 8) & (j < CH - 8)
    ok &= VALID[j.clamp(0, CH - 1), i.clamp(0, CW - 1)]
    better = ok & (z < best_z)
    if better.any():
        img = IM[int(n)].to(dev)
        rgb[better] = img[j[better], i[better]].float() / 255
        best_z[better] = z[better]
seen = torch.isfinite(best_z)
pts, rgb, best_z = pts[seen], rgb[seen], best_z[seen]
if a.anchor:
    n0 = n0[seen]
    mu0 = pts.clone()
nn_d, _ = cKDTree(pts.cpu().numpy()).query(pts.cpu().numpy(), k=2)
spacing = torch.from_numpy(nn_d[:, 1].astype(np.float32)).to(dev)
scale0 = torch.maximum(best_z / fx, 0.5 * spacing)
opa0 = 0.1
if a.baseline:                                  # the papers' start: RMS distance to the 3 nearest points
    nn3, _ = cKDTree(pts.cpu().numpy()).query(pts.cpu().numpy(), k=4)
    scale0 = torch.from_numpy(np.sqrt((nn3[:, 1:] ** 2).mean(1)).astype(np.float32)).to(dev).clamp(min=1e-5)
    if a.baseline == "mcmc":
        scale0, opa0 = 0.1 * scale0, 0.5
log(f"{int(seen.sum()):,} of {len(seen):,} points seen by a training frame; scale median {scale0.median().item() * 1000:.2f} mm "
    f"(depth/focal {(best_z / fx).median().item() * 1000:.2f} mm, half spacing {(0.5 * spacing).median().item() * 1000:.2f} mm)")

C0 = 0.28209479177387814
params = torch.nn.ParameterDict({
    "means": torch.nn.Parameter(pts.clone()),
    "scales": torch.nn.Parameter(torch.log(scale0).unsqueeze(1).repeat(1, 3)),
    "quats": torch.nn.Parameter(quat_z_to(n0) if a.anchor else torch.tensor([1.0, 0, 0, 0], device=dev).repeat(len(pts), 1),
                                requires_grad=not a.anchor),
    "opacities": torch.nn.Parameter(torch.logit(torch.full((len(pts),), 0.99 if a.opaque else opa0, device=dev)),
                                    requires_grad=not a.opaque),
    "sh0": torch.nn.Parameter(((rgb - 0.5) / C0).unsqueeze(1)),
    "shN": torch.nn.Parameter(torch.zeros(len(pts), 15, 3, device=dev)),
})
if a.lamp_ambient:
    params["amb"] = torch.nn.Parameter(torch.full((len(pts), 3), -4.6, device=dev))   # sigmoid -> 0.01
THIN = math.log(a.thin_mm / 1000)
CAP = (torch.log(scale0) + math.log(a.scale_cap)) if a.scale_cap > 0 else None
if a.anchor:                                    # thin discs in the surface
    with torch.no_grad():
        params["scales"][:, 2] = THIN
if a.eval_only:                                 # score an already trained model
    saved = torch.load(os.path.join(a.out, "model.pt"))
    params = torch.nn.ParameterDict({k: torch.nn.Parameter(saved[k].to(dev)) for k in params.keys()})
    log(f"eval only: loaded {len(params['means']):,} Gaussians from {a.out}/model.pt; scoring pixels with opacity > {a.alpha_min}")
scene_scale = 1.1 * float(np.max(np.linalg.norm(eyes - eyes.mean(0), axis=1)))
lrs = dict(means=1.6e-4 * scene_scale, scales=5e-3, quats=1e-3, opacities=5e-2, sh0=2.5e-3, shN=2.5e-3 / 20)
if a.lamp_ambient:
    lrs["amb"] = 5e-3
optimizers = {k: torch.optim.Adam([params[k]], lr=lr, eps=1e-15) for k, lr in lrs.items()
              if not (a.anchor and k == "quats") and not (a.opaque and k == "opacities")}
if a.lamp:
    if a.eval_only:
        with torch.no_grad():
            for k in lamp.keys():
                lamp[k][:] = saved["lamp_" + k].to(dev)
    for k, lr in dict(logM=3e-3, t=1e-4, gain=1e-3, off=1e-4).items():
        optimizers["lamp_" + k] = torch.optim.Adam([lamp[k]], lr=lr, eps=1e-15)
POSE = {}
if a.pose_opt:
    POSE = torch.nn.ParameterDict({
        "w": torch.nn.Parameter(torch.zeros(len(views_all), 3, device=dev)),    # per frame rotation vector, rad
        "c": torch.nn.Parameter(torch.zeros(len(views_all), 3, device=dev)),    # per frame camera centre, m (world)
        "wg": torch.nn.Parameter(torch.zeros(3, device=dev)),                   # shared rotation (camera)
        "cg": torch.nn.Parameter(torch.zeros(3, device=dev))})                  # shared shift, camera axes
    if a.eval_only:
        with torch.no_grad():
            for k in POSE.keys():
                POSE[k][:] = saved["pose_" + k].to(dev)
    pose_opt = {k: torch.optim.Adam([POSE[k]], lr=lr, eps=1e-15)
                for k, lr in dict(w=1e-5, c=1e-5, wg=2e-5, cg=2e-5).items()}


def so3(wv):
    th = wv.norm().clamp(min=1e-12)
    k = wv / th
    Kx = torch.zeros(3, 3, device=wv.device)
    Kx[0, 1], Kx[0, 2], Kx[1, 2] = -k[2], k[1], -k[0]
    Kx = Kx - Kx.T
    return torch.eye(3, device=wv.device) + torch.sin(th) * Kx + (1 - torch.cos(th)) * (Kx @ Kx)


def corrected(view, n=None, pair=None):
    """view with the learned correction of training frame n, or interpolated between pair = ((i0, i1), t)"""
    if not a.pose_opt:
        return view
    if pair is None:
        wv, dc = POSE["w"][n], POSE["c"][n]
    else:
        (i0, i1), t_ = pair
        wv, dc = (1 - t_) * POSE["w"][i0] + t_ * POSE["w"][i1], (1 - t_) * POSE["c"][i0] + t_ * POSE["c"][i1]
    R, t = view[:3, :3], view[:3, 3]
    C = -R.T @ t
    R2 = so3(wv + POSE["wg"]) @ R
    C2 = C + dc + R.T @ POSE["cg"]
    top = torch.cat([R2, (-R2 @ C2)[:, None]], 1)
    return torch.cat([top, view[3:4]], 0)


TRAIN_SORTED = np.sort(train)


def neighbours(fid, ft):
    """the two training frames around a frame not trained on (by time for InPipe, by frame number otherwise)"""
    key = d["train_t"] if inpipe else d["train_ids"]
    x = ft if inpipe else fid
    kk = key[TRAIN_SORTED]
    j = int(np.searchsorted(kk, x))
    i0, i1 = TRAIN_SORTED[max(j - 1, 0)], TRAIN_SORTED[min(j, len(TRAIN_SORTED) - 1)]
    t_ = 0.0 if key[i1] == key[i0] else float((x - key[i0]) / (key[i1] - key[i0]))
    return (int(i0), int(i1)), min(max(t_, 0.0), 1.0)


sched = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / a.iters))
strategy = state = None
if a.baseline == "mcmc":
    strategy = MCMCStrategy(cap_max=len(pts), refine_stop_iter=min(25000, a.iters * 5 // 6))
    state = strategy.initialize_state()
elif a.adc:
    strategy = DefaultStrategy(refine_stop_iter=min(15000, a.iters // 2),
                               key_for_gradient="gradient_2dgs" if a.baseline == "2dgs" else "means2d")
    state = strategy.initialize_state(scene_scale=scene_scale)
if strategy is not None:
    strategy.check_sanity(params, optimizers)
SH = [0 if a.baseline else 3]                   # SH degree in use (the baselines raise it every 1000 iterations)
log(f"scene scale {scene_scale:.2f} m; ADC {'on' if a.adc else 'off'}; anchor {'on' if a.anchor else 'off'}; "
    f"{a.iters} iterations" + (f"; baseline {a.baseline}" if a.baseline else ""))


def rasterize(view, mode="RGB+ED"):
    p = params
    if a.baseline == "2dgs":                    # surfels; normals and distortion kept in info for the losses
        if torch.is_grad_enabled() and mode == "RGB+ED":
            mode = "RGB+D"                      # as gsplat's simple_trainer_2dgs (its ED backward fails here)
        out, alpha, nrm, snrm, dist, _, info = gsplat.rasterization_2dgs(
            p["means"], p["quats"], torch.exp(p["scales"]), torch.sigmoid(p["opacities"]),
            torch.cat([p["sh0"], p["shN"]], 1), view[None], Kc[None], CW, CH, sh_degree=SH[0], packed=False,
            render_mode=mode, near_plane=0.01, distloss=True)
        info["n2d"], info["sn2d"], info["dist2d"] = nrm, snrm, dist
        return out, alpha, info
    return gsplat.rasterization(p["means"], p["quats"], torch.exp(p["scales"]), torch.sigmoid(p["opacities"]),
                                torch.cat([p["sh0"], p["shN"]], 1), view[None], Kc[None], CW, CH, sh_degree=SH[0],
                                packed=False, render_mode=mode, near_plane=0.01, **UT)


def ambient(view):
    """the per-Gaussian ambient colour rendered on the canvas (--lamp-ambient)"""
    p = params
    out, _, _ = gsplat.rasterization(p["means"], p["quats"], torch.exp(p["scales"]), torch.sigmoid(p["opacities"]),
                                     torch.sigmoid(p["amb"]), view[None], Kc[None], CW, CH, packed=False,
                                     render_mode="RGB", near_plane=0.01, **UT)
    return out[0]


# ------------------------------------------------------------------------------ training ---------------------
torch.cuda.reset_peak_memory_stats()
t_train = time.time()
seq, done = [], 0
lam, gam = a.dssim, a.depth_w
t_of = d["train_t"]
try:
    for it in range(1, 0 if a.eval_only else a.iters + 1):
        if not seq:
            seq = list(np.random.permutation(train))
        n = int(seq.pop())
        gt = IM[n].to(dev, non_blocking=True).float() / 255
        view = corrected(views_all[n], n)
        if a.baseline:
            SH[0] = min(it // 1000, 3)
        out, alpha, info = rasterize(view)
        if a.adc:
            strategy.step_pre_backward(params, optimizers, state, it, info)
        ren, dep = out[0, ..., :3], out[0, ..., 3]
        if a.lamp:
            lit = lamp_factor(view.detach())[..., None] * ren + (ambient(view) if a.lamp_ambient else 0)
            ren = torch.exp(lamp["gain"][n]) * lit + lamp["off"][n]
        l1 = (ren - gt).abs()[VALID].mean()
        ssim = fused_ssim(torch.where(VALID[..., None], ren, gt).permute(2, 0, 1)[None].contiguous(),
                          gt.permute(2, 0, 1)[None].contiguous())
        if a.anchor:                            # dense relative depth to the measured surface
            Da = surface_depth(view.detach()).float() if a.pose_opt else DA[n].float()
            md = (Da > 0) & VALID
            ld = ((dep - Da).abs() / Da.clamp(min=1e-3))[md].mean() if md.any() else torch.zeros((), device=dev)
            loss = (1 - lam) * l1 + lam * (1 - ssim) + a.anchor_depth_w * ld
        else:
            if gam > 0:
                Ds = depth_map(view.detach(), t_of[n])
                md = (Ds > 0) & VALID
                ld = (dep - Ds).abs()[md].mean() if md.any() else torch.zeros((), device=dev)
                loss = (1 - lam) * l1 + lam * (1 - ssim) + gam * ld
            else:                               # baselines: no depth term (and no zero gradient into 2DGS depth)
                ld = torch.zeros((), device=dev)
                loss = (1 - lam) * l1 + lam * (1 - ssim)
        if a.baseline == "mcmc":                # opacity and scale regularisation of 3DGS-MCMC
            loss = loss + 0.01 * torch.sigmoid(params["opacities"]).mean() + 0.01 * torch.exp(params["scales"]).mean()
        if a.baseline == "2dgs":                # normal consistency (from 7k) and depth distortion (from 3k)
            if it > a.normal_start:
                sn = info["sn2d"][0] * alpha[0].detach()
                loss = loss + 0.05 * (1 - (info["n2d"][0] * sn).sum(-1)).mean()
            if it > a.dist_start:
                loss = loss + 0.01 * info["dist2d"].mean()
        if a.pose_opt:                          # weak priors: per frame a few mm / mrad, shared ~1 cm / 20 mrad
            loss = loss + 1e-4 * ((POSE["c"][n].norm() / 0.002) ** 2 + (POSE["w"][n].norm() / 0.002) ** 2
                                  + (POSE["cg"].norm() / 0.010) ** 2 + (POSE["wg"].norm() / 0.020) ** 2)
        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        if a.pose_opt:
            for opt in pose_opt.values():
                if it > a.pose_start:
                    opt.step()
                opt.zero_grad(set_to_none=True)
        sched.step()
        if a.baseline == "mcmc":
            strategy.step_post_backward(params, optimizers, state, it, info, lr=sched.get_last_lr()[0])
        elif a.adc:
            strategy.step_post_backward(params, optimizers, state, it, info, packed=False)
        if a.anchor:
            with torch.no_grad():                  # stay within the clamp along the normal, stay thin
                off = ((params["means"] - mu0) * n0).sum(1, keepdim=True)
                lim = a.anchor_clamp_mm / 1000
                params["means"] -= (off - off.clamp(-lim, lim)) * n0
                params["scales"][:, 2].clamp_(max=THIN)
        if CAP is not None:
            with torch.no_grad():                  # scale cap relative to the start (in-plane only with --anchor)
                dims = 2 if a.anchor else 3
                params["scales"][:, :dims] = torch.minimum(params["scales"][:, :dims], CAP[:, None])
        done = it
        if it % 500 == 0 or it == 1:
            with torch.no_grad():
                ps = -10 * torch.log10(((ren - gt) ** 2)[VALID].mean()).item()
            dtxt = f"depth {ld.item():.2%}" if a.anchor else f"depth {ld.item() * 1000:.1f} mm"
            if a.pose_opt:
                dtxt += (f" | pose per frame {POSE['c'].norm(dim=1).median().item() * 1000:.2f} mm "
                         f"{POSE['w'].norm(dim=1).median().item() * 1000:.2f} mrad, shared "
                         f"{(POSE['cg'] * 1000).round(decimals=1).tolist()} mm {(POSE['wg'] * 1000).round(decimals=2).tolist()} mrad")
            log(f"iter {it:6d} loss {loss.item():.4f} L1 {l1.item():.4f} SSIM {ssim.item():.3f} {dtxt} "
                f"PSNR {ps:.2f} | {len(params['means']):,} Gaussians | {it / (time.time() - t_train):.1f} it/s | "
                f"peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
except torch.cuda.OutOfMemoryError as e:
    log(f"OUT OF MEMORY at iteration {done + 1} with {len(params['means']):,} Gaussians: {e}")
torch.cuda.synchronize()
SH[0] = 3                                       # scoring (and --eval-only) with the full SH degree
train_s = time.time() - t_train
peak = torch.cuda.max_memory_allocated() / 1e9
if not a.eval_only:
    extra = {"lamp_" + k: v.detach().cpu() for k, v in lamp.items()} if a.lamp else {}
    extra |= {"pose_" + k: v.detach().cpu() for k, v in POSE.items()} if a.pose_opt else {}
    torch.save({k: v.detach().cpu() for k, v in params.items()} | extra | dict(K=Kc.cpu(), size=(CW, CH)),
               os.path.join(a.out, "model.pt"))
    if a.lamp:
        log(f"lamp: offset from the camera centre {(lamp['t'] * 1000).round().tolist()} mm; M range "
            f"{torch.exp(lamp['logM']).min().item():.2f}..{torch.exp(lamp['logM']).max().item():.2f}")
    log(f"trained {done} iterations in {train_s / 60:.1f} min, {len(params['means']):,} Gaussians, peak {peak:.2f} GB")


# ------------------------------------------------------------------------------ evaluation -------------------
@torch.no_grad()
def render_raw(view):
    out, alpha, _ = rasterize(view)
    if a.lamp:                                  # the frame's own gain and offset are fitted by the scoring
        lit = out[..., :3] * lamp_factor(view)[None, ..., None] + (ambient(view)[None] if a.lamp_ambient else 0)
        out = torch.cat([lit, out[..., 3:]], -1)
    ren = (out[0, ..., :3].clamp(min=0) if a.lamp else out[0, ..., :3].clamp(0, 1)).mul(255).cpu().numpy()
    al = alpha[0, ..., 0].cpu().numpy()
    dep_c = out[0, ..., 3]
    if to_raw is not None:
        ren = cv2.remap(ren, *to_raw, cv2.INTER_LINEAR, borderValue=0)
        al = cv2.remap(al, *to_raw, cv2.INTER_LINEAR, borderValue=0)
    return ren, al, dep_c, alpha[0, ..., 0]


def so3_t(wv):
    th = wv.norm().clamp(min=1e-12)
    k = wv / th
    Kx = torch.zeros(3, 3, device=wv.device)
    Kx[0, 1], Kx[0, 2], Kx[1, 2] = -k[2], k[1], -k[0]
    Kx = Kx - Kx.T
    return torch.eye(3, device=wv.device) + torch.sin(th) * Kx + (1 - torch.cos(th)) * (Kx @ Kx)


def pose_step(view, wv, dc):
    """view moved by a rotation vector wv (camera axes) and a camera-centre shift dc (camera axes), as tensors"""
    R, t = view[:3, :3], view[:3, 3]
    C = -R.T @ t
    R2 = so3_t(wv + 1e-12) @ R
    return torch.cat([torch.cat([R2, (-R2 @ (C + R.T @ dc))[:, None]], 1), view[3:4]], 0)


def proj_raw(Xw, view):
    """raw-frame pixel coordinates (float) of world points (numpy), with the radial k1"""
    R, t = view[:3, :3].cpu().numpy().astype(np.float64), view[:3, 3].cpu().numpy().astype(np.float64)
    c = Xw @ R.T + t
    xn, yn = c[:, 0] / c[:, 2], c[:, 1] / c[:, 2]
    g = 1 + k1 * (xn ** 2 + yn ** 2)
    return np.stack([cx + fx * xn * g, cy + fy * yn * g], 1)


def test_pose(view, img, Bmask):
    """test-time registration of one frame (docs/10 X6c), model frozen: render, optical flow from the frame to the
    render on CLAHE-normalised images at its detailed pixels, back-project those pixels with the rendered depth and
    solve the 6 pose numbers whose image motion cancels the flow (numerical Jacobian, Huber-weighted least squares);
    a.test_pose_iters rounds. Detail-driven: smooth shading does not pull the pose (the photometric version did)."""
    real = img.astype(np.float32)
    v0 = view
    xu_raw = (np.arange(w) - cx) / fx
    yu_raw = (np.arange(h) - cy) / fy
    if k1 != 0:
        XUr, YUr = np.nan_to_num(XU), np.nan_to_num(YU)
    else:
        XUr, YUr = np.meshgrid(xu_raw, yu_raw)
    for _ in range(a.test_pose_iters):
        G, aG, dep_c, _ = render_raw(view)
        D = dep_c.cpu().numpy()
        if to_raw is not None:
            D = cv2.remap(D, *to_raw, cv2.INTER_NEAREST, borderValue=0)
        m = Bmask & valid_raw & (aG > 0.5) & (D > 0.01)
        Gm = im.linear_match(G, real, m)
        A = im._CLAHE.apply(np.clip(im.grey(real), 0, 255).astype(np.uint8))
        Bq = im._CLAHE.apply(np.clip(im.grey(Gm), 0, 255).astype(np.uint8))
        fl = im._DIS.calc(A, Bq, None)
        gs = cv2.GaussianBlur(A.astype(np.float32), (0, 0), 1.5)
        gm = np.hypot(cv2.Sobel(gs, cv2.CV_32F, 1, 0), cv2.Sobel(gs, cv2.CV_32F, 0, 1))
        inner = im.erode(m, 15)
        if inner.sum() < 500:
            break
        sel = inner & (gm >= np.percentile(gm[inner], 70)) & (np.hypot(fl[..., 0], fl[..., 1]) < 40)
        ys, xs = np.nonzero(sel)
        if len(ys) > 20000:
            k_ = np.random.default_rng(0).choice(len(ys), 20000, replace=False)
            ys, xs = ys[k_], xs[k_]
        z = D[ys, xs].astype(np.float64)
        Xc = np.stack([XUr[ys, xs] * z, YUr[ys, xs] * z, z], 1)
        R, t = view[:3, :3].cpu().numpy().astype(np.float64), view[:3, 3].cpu().numpy().astype(np.float64)
        Xw = (Xc - t) @ R
        p0 = proj_raw(Xw, view)
        J = np.zeros((len(ys), 2, 6))
        for k_ in range(6):
            e = np.zeros(6)
            e[k_] = 1e-4
            vk = pose_step(view, torch.tensor(e[:3], device=dev, dtype=torch.float32),
                           torch.tensor(e[3:], device=dev, dtype=torch.float32))
            J[:, :, k_] = (proj_raw(Xw, vk) - p0) / 1e-4
        f = fl[ys, xs].astype(np.float64)
        Jm, rhs = J.reshape(-1, 6), (-f).reshape(-1)
        wgt = np.ones(len(rhs))
        for _ in range(4):                         # Huber IRLS, 1 px
            Aw = Jm * wgt[:, None]
            xi = np.linalg.lstsq(Aw.T @ Jm + 1e-6 * np.eye(6), Aw.T @ rhs, rcond=None)[0]
            res = np.abs(Jm @ xi - rhs)
            wgt = np.where(res < 1.0, 1.0, 1.0 / np.maximum(res, 1e-9))
        view = pose_step(view, torch.tensor(xi[:3], device=dev, dtype=torch.float32),
                         torch.tensor(xi[3:], device=dev, dtype=torch.float32))
    R0, t0 = v0[:3, :3], v0[:3, 3]
    R1, t1 = view[:3, :3], view[:3, 3]
    dC = (-R1.T @ t1) - (-R0.T @ t0)
    ang = torch.acos(((torch.trace(R1 @ R0.T) - 1) / 2).clamp(-1, 1))
    return view.detach(), float(dC.norm().item() * 1000), float(ang.item() * 1000)


def score(tag, imgs, nxt, views, B, Bm, ids, pos, times):
    rows = []
    for k in range(len(ids)):
        view = torch.from_numpy(views[k]).to(dev)
        if a.pose_opt:                          # not trained on: correction interpolated from the neighbours
            with torch.no_grad():
                view = corrected(view, pair=neighbours(int(ids[k]), int(times[k])))
        tp = None
        if a.test_pose_iters:
            view, tp_mm, tp_mrad = test_pose(view, imgs[k], Bm[k])
            tp = (tp_mm, tp_mrad)
        G, aG, dep_c, al_c = render_raw(view)
        mG = valid_raw & (aG > a.alpha_min)
        m = Bm[k] & mG
        real = imgs[k].astype(np.float32)
        Bk = B[k].astype(np.float32)
        row = dict(tag=tag, frame=int(ids[k]), pos=float(pos[k]), coverage=float(m.mean()), cov_G=float(mG.mean()),
                   cov_B=float(Bm[k].mean()))
        if tp is not None:
            row["test_pose_mm"], row["test_pose_mrad"] = tp
        mats = {}
        if a.ruler2:                            # same pixels for every model: B's mask, no opacity threshold
            m = Bm[k] & valid_raw
            row["coverage"] = float(m.mean())
            nal = im.linear_match(im.align_to(nxt[k].astype(np.float32), real), real, m)
            den = cv2.fastNlMeansDenoisingColored(np.ascontiguousarray(imgs[k]), None, 5, 5, 7, 21).astype(np.float32)
            cands = (("G", G, m), ("B", Bk, m), ("Nal", nal, m), ("D", den, m))
        else:
            cands = (("G", G, m), ("B", Bk, m), ("N", nxt[k].astype(np.float32), m), ("Bown", Bk, Bm[k]))
        for name, img, mask in cands:
            x = im.linear_match(img, real, mask)
            mats[name] = x
            for key, val in im.all_metrics(x, real, mask).items():
                row[f"{name}_{key}"] = val
            if a.ruler2:
                for key, val in im.ruler2(x, real, nal, mask).items():
                    row[f"{name}_{key}"] = val
        Ds = depth_map(view, times[k])
        md = (Ds > 0) & (al_c > a.alpha_min) & VALID
        rel = ((dep_c - Ds).abs() / Ds)[md]
        row["depth_rel_err"] = float(rel.median()) if len(rel) else np.nan
        if a.anchor:
            Da = surface_depth(view)
            ma = (Da > 0) & (al_c > a.alpha_min) & VALID
            rel = ((dep_c - Da).abs() / Da)[ma]
            row["depth_rel_err_surface"] = float(rel.median()) if len(rel) else np.nan
        rows.append(row)
        if a.save and k % a.save == 0 and not a.eval_only:
            trip = np.hstack([real, mats["B"], mats["G"]]).clip(0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(a.out, "renders", f"{tag}_{int(ids[k]):06d}.jpg"), trip[:, :, ::-1],
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
        ref = "Nal" if a.ruler2 else "N"
        log(f"  {tag} {k + 1}/{len(ids)} frame {int(ids[k])}: PSNR G {row['G_psnr']:.2f} B {row['B_psnr']:.2f} "
            f"{ref} {row[ref + '_psnr']:.2f} | LPIPS G {row['G_lpips']:.3f} B {row['B_lpips']:.3f} | depth err {row['depth_rel_err']:.1%}")
    return rows


def evaluate_all():
  global rows, rows_b, keys
  rng = d["block_ranges"]
  in_block = lambda p: bool(np.any([(p >= lo) & (p < hi) for lo, hi in rng]))  # noqa: E731
  held_t = d["held_t"] if "held_t" in d.files else np.zeros(len(d["held_ids"]), np.int64)
  rows = score("held", d["held_images"], d["held_next"], d["held_viewmats"], d["held_B"], d["held_mask"], d["held_ids"],
               d["held_pos"], held_t)
  for r in rows:
      r["in_block"] = in_block(r["pos"])
  bi = block_idx
  rows_b = score("block", raw_train[bi], d["block_next"], d["block_viewmats"], d["block_B"], d["block_mask"], d["block_ids"],
                 d["train_pos"][bi], t_of[bi])
  for r in rows_b:
      r["in_block"] = True

  keys = [k for k in rows[0] if k not in ("tag", "frame", "in_block")]
  with open(os.path.join(a.out, f"per_frame{a.suffix}.csv"), "w", newline="", encoding="utf-8") as fh:
      wr = csv.DictWriter(fh, ["tag", "frame", "in_block"] + keys)
      wr.writeheader()
      wr.writerows(rows + rows_b)


  def med(rs):
      return {k: float(np.nanmedian([r[k] for r in rs])) for k in keys} if rs else {}


  summary = dict(scene=a.scene, args=vars(a), canvas=[CW, CH], train_frames=int(len(train)), iterations_done=done,
                 train_minutes=round(train_s / 60, 2), it_per_s=round(done / max(train_s, 1e-9), 1), peak_GB=round(peak, 2),
                 gaussians=int(len(params["means"])), block_ranges_m=rng.tolist(),
                 held_all=med(rows), held_outside_blocks=med([r for r in rows if not r["in_block"]]),
                 block=med(rows_b), n=dict(held=len(rows), held_outside_blocks=sum(not r["in_block"] for r in rows),
                                           block=len(rows_b)))
  json.dump(summary, open(os.path.join(a.out, f"summary{a.suffix}.json"), "w"), indent=1)
  M = ("psnr", "lpips", "shift_px", "detail_corr", "hf_ratio", "mse_low", "mse_high")
  for title, s in (("in-sequence held-out, outside the blocks", summary["held_outside_blocks"]), ("block frames", summary["block"])):
      log(f"\n{title} (median; coverage {s['coverage']:.1%}, depth error G {s['depth_rel_err']:.1%}):")
      tags = ((("G", "model"), ("B", "lampmap B"), ("Nal", "next frame aligned"), ("D", "frame denoised"))
              if a.ruler2 else (("G", "G0"), ("B", "lampmap B"), ("N", "next frame"), ("Bown", "B on its own mask")))
      M2 = M + (tuple(f"{q}_{b}" for q in ("fid", "en") for b, _, _ in im.BANDS) if a.ruler2 else ())
      for tag, name in tags:
          log(f"  {name:18s} " + "  ".join(f"{k} {s[f'{tag}_{k}']:.3f}" for k in M2))
  log("done")


if a.smoke:
    log("smoke run: no scoring")
    sys.exit(0)
alphas = [a.alpha_min] if a.alphas is None and (a.eval_only and (a.alpha_min != 0.99 or a.suffix)) else (a.alphas or [0.99, 0.5])
for k_al, al_min in enumerate(alphas):
    a.alpha_min = al_min
    a.suffix = a.suffix if len(alphas) == 1 and a.suffix else ("" if al_min == 0.99 else f"_alpha{int(round(al_min * 100)):03d}")
    if k_al > 0:
        a.save = 0                              # the triptychs once
    log(f"=== scoring pixels with opacity > {al_min} (files *{a.suffix}) ===")
    evaluate_all()
if not a.eval_only:                             # and once with ruler v2 (docs/10 section 2)
    a.ruler2, a.alpha_min, a.suffix, a.save = True, 0.5, "_r2", 0
    log("=== ruler v2: B's mask for every model (files *_r2) ===")
    evaluate_all()
