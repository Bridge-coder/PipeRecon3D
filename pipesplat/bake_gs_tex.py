"""Bake a Gaussian model into the unrolled mesh texture (docs/10 X8): a viewer-ready textured mesh (OBJ + PNG through
export_tex_obj.py) whose colours come from the best rendering model (G0-A, train_gs.py --anchor) instead of the CLAHE
frames (OpenMVS) or a texture learnt together with a lamp model (TX, train_tex.py). Runs in the pipe3d WSL distro.

1. texel -> surface point: the measured mesh (SCENE_DIR/mesh.npz from pipesplat/review/dump_mesh.py: the faces that
   export_tex_obj.py writes) rasterised into the unrolled (u, theta) texture of train_tex.py (same layout, 0.5 mm);
   where faces overlap in (u, theta) the innermost one (what a camera inside the pipe sees). End-cap faces (beyond
   cap_x with an axial normal, as train_tex.py / export_tex_obj.py) go into the planar (y, z) cap texture (nearest x).
2. every training pose: the Gaussian model rendered on its training canvas (as train_gs.py, colour + depth). Each
   texel is projected into it and kept where it is visible (depth within 1.5 % + 2 mm of the surface depth), inside
   the canvas (eroded --margin-px) and the model is opaque (alpha >= --alpha-min). Its footprint = the size of one
   RAW pixel on the wall there: depth / f / cos(incidence to the pipe radius) x the canvas magnification of k1 at that
   pixel (the canvas edge is the raw frame stretched about 2x).
   --model-surface (default; --mesh-surface for the first version): the opaque Gaussians of G0-A sit ~3.6 mm behind
   the OpenMVS mesh (pipesplat/review/gs_offset.py), so the views that see a mesh point sharpest (~10 cm of camera
   travel) look at points of the model's wall ~2 mm apart: the first bake was blurred and shifted by ~2 mm along the
   pipe. Now the model's own surface is used: per texel the radial offset of the model's depth from the mesh along
   the pixel's ray (near views, alpha >= 0.99, smoothed 2 mm; the end cap along x), the mesh vertices moved by it
   (OUT_DIR/mesh_bake.npz + surface_bake.npz: the geometry of the delivered OBJ and of the scoring), the texture
   layout recomputed from the moved surface, the texels rasterised again and visibility tested against the model's
   depth.
3. two weighted means per texel:
   near  the views that see it sharpest, w = exp(-(fp / fp_best - 1) / --near-tau). Sharp, but lit differently all
         around the pipe: the camera rides ~5 cm below the axis and the frame is wider than tall, so the sharpest
         views of the bottom wall are ~0.26 m away, of the top wall ~0.54 m (the lamp falls off ~ cos / r^2)
   ref   the views at one distance ahead, w = exp(-((z - --ref-z) / --ref-sigma)^2) (+1e-4 so every seen texel has
         one): every texel lit from the same place relative to the camera - consistent shading, coarser detail
   texture = near x blur(ref) / blur(near) per channel, blur = Gaussian of --lp-mm in the texture (wrapping in theta):
   the detail of the sharp views with the brightness of the consistent ones (the end cap the same way, no wrap).
   --end-x X (X8b): the end region - the planar cap texels and the wall texels beyond u = X (blended in over --end-ramp)
   - gets its own colours, because the near views of the specular end cap burn lamp highlights into it:
   ref     the ref mean alone (views ~--ref-z away, no near detail)
   tau     the near mean with --end-tau instead of --near-tau (many views), normalised by the ref mean as above
   robust  as tau, then every view's weight (near and ref) times exp(-0.5 max(0, (log L - mu) / --end-hl)^2), L the
           view's luminance at the texel, mu the weighted mean of log L: views that see the texel brighter than the
           others (a highlight) count less
4. empty texels filled (push-pull, for mip-mapping); written as texture.png / cap.png (8 bit, 99.5th percentile at
   250, as train_tex.py) + texture.json, and tex.pt / tex_near.pt in train_tex.py's layout made from the 8-bit
   values (what is scored is the delivered PNG):
   train_tex.py SCENE_DIR SCORE_DIR --eval-only --no-lamp [--lamp-map] --tex-pt OUT_DIR/tex.pt
       [--surface OUT_DIR/surface_bake.npz]
   export_tex_obj.py vision WORK_DIR OUT_DIR [--mesh OUT_DIR/mesh_bake.npz]

usage: TORCH_HOME=/mnt/d/pipe3d_gs/torch_cache /opt/gsenv/bin/python pipesplat/bake_gs_tex.py SCENE_DIR MODEL_RUN OUT_DIR
           [--mesh-surface] [--near-tau 1.0] [--ref-z 0.65] [--ref-sigma 0.1] [--lp-mm 12] [--alpha-min 0.95]
           [--mesh mesh.npz] [--end-x X --end-mode ref|tau|robust [--end-tau 10] [--end-hl 0.15] [--end-ramp 0.02]]
       MODEL_RUN  a train_gs.py run without --lamp (model.pt; its colour is what the camera saw)
writes OUT_DIR/bake.log, texture.png, cap.png, texture.json, tex.pt, tex_near.pt, preview_*.jpg, bake_stats.npz
       (+ mesh_bake.npz, surface_bake.npz with --model-surface)
"""
import argparse
import json
import math
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import gsplat

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("scene"), ap.add_argument("run"), ap.add_argument("out")
ap.add_argument("--mesh", default="mesh.npz", help="mesh file in SCENE_DIR (pipesplat/review/dump_mesh.py)")
ap.add_argument("--surface", default="surface.npz", help="measured-surface discs in SCENE_DIR (export_surface.py)")
ap.add_argument("--mesh-surface", action="store_true",
                help="bake on the measured mesh as it is (the first version, run_bake_v1): no offset to the model's surface")
ap.add_argument("--disp-smooth-mm", type=float, default=2.0)
ap.add_argument("--texel-mm", type=float, default=0.5)
ap.add_argument("--near-tau", type=float, default=1.0, help="0.1 / 0.3 / 1 / 3 / 10 tried (docs/10 X8): 1 scores best")
ap.add_argument("--ref-z", type=float, default=0.65, help="m ahead of the camera (the top wall is visible from ~0.54 m)")
ap.add_argument("--ref-sigma", type=float, default=0.1)
ap.add_argument("--lp-mm", type=float, default=12.0)
ap.add_argument("--alpha-min", type=float, default=0.95)
ap.add_argument("--margin-px", type=int, default=3)
ap.add_argument("--cap-x", type=float, default=None, help="default: last training camera + 0.15 m (train_tex.py)")
ap.add_argument("--views", type=int, nargs="+", default=None, help="diagnosis: bake from these training indices only")
ap.add_argument("--end-x", type=float, default=None,
                help="X8b: own colours for the end region (the cap texels + the wall beyond this u, m); default off")
ap.add_argument("--end-mode", choices=("ref", "tau", "robust"), default="robust")
ap.add_argument("--end-tau", type=float, default=10.0)
ap.add_argument("--end-hl", type=float, default=0.15, help="robust: scale of the highlight down-weighting (log luminance)")
ap.add_argument("--end-ramp", type=float, default=0.02, help="m along the wall over which the end colours blend in")
a = ap.parse_args()
dev = "cuda"
os.makedirs(a.out, exist_ok=True)
T0 = time.time()
_logf = open(os.path.join(a.out, "bake.log"), "a", encoding="utf-8")


def log(msg):
    line = f"{msg} [{time.time() - T0:.0f} s]"
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


log(f"args {vars(a)} | torch {torch.__version__} gsplat {gsplat.__version__}")
d = np.load(os.path.join(a.scene, "scene.npz"))
assert "depth_scan" not in d.files, "vision scenes only (the pipe frame of scene.npz: x along the pipe)"
K = d["K"].astype(np.float64)
k1 = float(d["k1"])
w, h = (int(v) for v in d["size"])
fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]


# ------------------------------------------------------------------------------ training canvas (train_gs.py) ---
def undistort_grid(xd, yd, k1):
    out = (xd ** 2 + yd ** 2 >= (0.95 * 2 / (3 * np.sqrt(-3 * k1))) ** 2) if k1 < 0 else np.zeros(xd.shape, bool)
    xd, yd = np.where(out, 0.0, xd), np.where(out, 0.0, yd)
    xu, yu = xd.copy(), yd.copy()
    for _ in range(20):
        g = 1 + k1 * (xu ** 2 + yu ** 2)
        xu, yu = xd / g, yd / g
    xu[out], yu[out] = np.nan, np.nan
    return xu, yu


if k1 == 0.0:
    CW, CH, ccx, ccy = w, h, cx, cy
    valid_canvas = np.ones((h, w), bool)
    R2MAX = float("inf")
else:
    XU, YU = undistort_grid(*np.meshgrid((np.arange(w) - cx) / fx, (np.arange(h) - cy) / fy), k1)
    CW, CH = 2 * int(np.ceil(np.nanmax(np.abs(XU)) * fx)) + 4, 2 * int(np.ceil(np.nanmax(np.abs(YU)) * fy)) + 4
    ccx, ccy = CW / 2 - 0.5, CH / 2 - 0.5
    xu, yu = np.meshgrid((np.arange(CW) - ccx) / fx, (np.arange(CH) - ccy) / fy)
    g = 1 + k1 * (xu ** 2 + yu ** 2)
    ud, vd = cx + fx * xu * g, cy + fy * yu * g
    valid_canvas = ((xu ** 2 + yu ** 2) < (0.95 / np.sqrt(-3 * k1)) ** 2) & (ud >= 0) & (ud <= w - 1) & (vd >= 0) & (vd <= h - 1)
    R2MAX = 0.95 ** 2 / (-3 * k1)
Kc = torch.tensor([[fx, 0, ccx + 0.5], [0, fy, ccy + 0.5], [0, 0, 1]], device=dev, dtype=torch.float32)
m_ = 2 * a.margin_px + 1
VALID = torch.from_numpy(cv2.erode(valid_canvas.astype(np.uint8), np.ones((m_, m_), np.uint8)) > 0).to(dev)
log(f"raw {w}x{h}, k1 {k1}; canvas {CW}x{CH}, valid (eroded {a.margin_px} px) {VALID.float().mean().item():.1%}")

# ------------------------------------------------------------------------------ Gaussian model -----------------
sv = torch.load(os.path.join(a.run, "model.pt"))
assert tuple(int(v) for v in sv["size"]) == (CW, CH) and torch.allclose(sv["K"].float(), Kc.cpu()), "canvas differs"
assert "lamp_logM" not in sv, "a model with --lamp: its colour is not what the camera saw (needs the lamp factor)"
GS = {k: sv[k].to(dev) for k in ("means", "quats", "scales", "opacities", "sh0", "shN")}
del sv
log(f"model {a.run}: {len(GS['means']):,} Gaussians")


@torch.no_grad()
def render_model(view):
    """colour [3, CH, CW], expected depth and alpha of the model on the canvas"""
    out, al, _ = gsplat.rasterization(GS["means"], GS["quats"], torch.exp(GS["scales"]), torch.sigmoid(GS["opacities"]),
                                      torch.cat([GS["sh0"], GS["shN"]], 1), view[None], Kc[None], CW, CH, sh_degree=3,
                                      packed=False, render_mode="RGB+ED", near_plane=0.01)
    return out[0, ..., :3].clamp(0, 1).permute(2, 0, 1).contiguous(), out[0, ..., 3].contiguous(), al[0, ..., 0].contiguous()


# ------------------------------------------------------------------------------ measured surface (visibility) --
def quat_z_to(n):
    z = torch.tensor([0.0, 0.0, 1.0], device=n.device)
    axis = torch.cross(z.expand_as(n), n, dim=1)
    s = axis.norm(dim=1, keepdim=True)
    c = n[:, 2:3].clamp(-1, 1)
    ang = torch.atan2(s, c)
    axis = torch.where(s > 1e-8, axis / s.clamp(min=1e-12), torch.tensor([1.0, 0, 0], device=n.device).expand_as(n))
    return torch.cat([torch.cos(ang / 2), axis * torch.sin(ang / 2)], dim=1)


sf = np.load(os.path.join(a.scene, a.surface))
S_c = torch.from_numpy(sf["centres"]).to(dev)
S_q = quat_z_to(torch.from_numpy(sf["normals"]).to(dev))
S_s = torch.stack([0.6 * torch.from_numpy(sf["size"]).to(dev)] * 2 + [torch.full((len(S_c),), 1e-5, device=dev)], 1)
S_o = torch.full((len(S_c),), 0.999, device=dev)
S_col = torch.zeros(len(S_c), 3, device=dev)


@torch.no_grad()
def surface_depth(view):
    out, al, _ = gsplat.rasterization(S_c, S_q, S_s, S_o, S_col, view[None], Kc[None], CW, CH, packed=False,
                                      render_mode="ED", near_plane=0.01)
    dep = out[0, ..., 0]
    return torch.where(al[0, ..., 0] > 0.5, dep, torch.zeros_like(dep))


# ------------------------------------------------------------------------------ texture layout (train_tex.py) --
du = a.texel_mm / 1000
cap_x = a.cap_x if a.cap_x is not None else float(d["train_eyes"][:, 0].max()) + 0.15
CAP_N = int(math.ceil(2 * 0.2 / du))


def set_layout(centres_t, centres_np):
    """train_tex.py's texture layout from a surface's triangle centres (sets the globals U0, U1, R_mean, NU, NTH, dth)"""
    global U0, U1, R_mean, NU, NTH, dth
    U0, U1 = float(centres_t[:, 0].min()) - 0.01, float(centres_t[:, 0].max()) + 0.01
    R_mean = float(np.median(np.hypot(centres_np[:, 1], centres_np[:, 2])))
    dth = du / R_mean
    NU, NTH = int(math.ceil((U1 - U0) / du)), int(math.ceil(2 * math.pi / dth))
    dth = 2 * math.pi / NTH
    log(f"texture {NTH} x {NU} texels, u {U0:.4f}..{U1:.4f} m, R_mean {R_mean * 1000:.2f} mm; cap beyond x {cap_x:.3f} m: "
        f"{CAP_N} x {CAP_N}")


set_layout(S_c, sf["centres"])


# ------------------------------------------------------------------------------ texel -> surface point ---------
def raster(P, key, X, H, W, wrap, chunk=150000, maxc=20000):
    """surface point of every texel centre (row r, column c) covered by a face: P [n, 3 corners, (column, row)] in
    texel units, key [n, 3] (the smallest wins where faces overlap), X [n, 3, 3] corner positions"""
    best = torch.full((H * W,), float("inf"), device=dev, dtype=torch.float64)
    pos = torch.full((H * W, 3), float("nan"), device=dev, dtype=torch.float32)
    skipped = 0
    for s in range(0, len(P), chunk):
        p, k, x = P[s:s + chunk], key[s:s + chunk], X[s:s + chunk]
        lo, hi = torch.ceil(p.amin(1)).long(), torch.floor(p.amax(1)).long()
        nc, nr = (hi[:, 0] - lo[:, 0] + 1).clamp(min=0), (hi[:, 1] - lo[:, 1] + 1).clamp(min=0)
        cnt = nc * nr
        big = cnt > maxc
        skipped += int(big.sum())
        cnt = torch.where(big, torch.zeros_like(cnt), cnt)
        tot = int(cnt.sum())
        if tot == 0:
            continue
        fi = torch.repeat_interleave(torch.arange(len(p), device=dev), cnt)
        loc = torch.arange(tot, device=dev) - (torch.cumsum(cnt, 0) - cnt)[fi]
        cc = lo[fi, 0] + loc % nc[fi]
        rr = lo[fi, 1] + torch.div(loc, nc[fi], rounding_mode="floor")
        a0 = p[fi, 0]
        v0, v1 = p[fi, 1] - a0, p[fi, 2] - a0
        q = torch.stack([cc, rr], 1).double() - a0
        den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        ok = den.abs() > 1e-12
        den = torch.where(ok, den, torch.ones_like(den))
        l1 = (q[:, 0] * v1[:, 1] - v1[:, 0] * q[:, 1]) / den
        l2 = (v0[:, 0] * q[:, 1] - q[:, 0] * v0[:, 1]) / den
        l0 = 1 - l1 - l2
        ok &= (l0 >= -1e-9) & (l1 >= -1e-9) & (l2 >= -1e-9)
        if wrap:
            rr = torch.remainder(rr, H)
        ok &= (cc >= 0) & (cc < W) & (rr >= 0) & (rr < H)
        bary = torch.stack([l0, l1, l2], 1)[ok]
        fi, lin = fi[ok], (rr * W + cc)[ok]
        kv = (bary * k[fi]).sum(1)
        best.scatter_reduce_(0, lin, kv, reduce="amin")
        win = kv <= best[lin]
        pos[lin[win]] = (bary[win, :, None] * x[fi[win]]).sum(1).float()
    return pos.view(H, W, 3), skipped


def cap_faces(V, TRI):
    A_, B_, C_ = V[TRI[:, 0]], V[TRI[:, 1]], V[TRI[:, 2]]
    fn = torch.cross(B_ - A_, C_ - A_, dim=1)
    fn = fn / fn.norm(dim=1, keepdim=True).clamp(min=1e-30)
    return (((A_ + B_ + C_) / 3)[:, 0] > cap_x) & (fn[:, 0].abs() > 0.5)


def texel_points(V, TRI, is_cap):
    """surface points of the wall texels [NTH, NU, 3] and of the cap texels [CAP_N, CAP_N, 3] (NaN = no face)"""
    t_r = time.time()
    wt = TRI[~is_cap]
    th_c = torch.atan2(-V[:, 1], V[:, 2])[wt]                # theta of each corner (train_tex.py to_pipe)
    seam = (th_c.amax(1) - th_c.amin(1)) > math.pi
    th_c = torch.where(seam[:, None] & (th_c < 0), th_c + 2 * math.pi, th_c)
    P_w = torch.stack([(V[:, 0][wt] - U0) / du - 0.5, (th_c + math.pi) / dth - 0.5], -1)
    pos, sk_w = raster(P_w, torch.hypot(V[:, 1], V[:, 2])[wt], V[wt], NTH, NU, wrap=True)
    ct = TRI[is_cap]
    P_c = torch.stack([(V[:, 1][ct] + 0.2) / du - 0.5, (V[:, 2][ct] + 0.2) / du - 0.5], -1)
    pos_c, sk_c = raster(P_c, V[:, 0][ct], V[ct], CAP_N, CAP_N, wrap=False)
    hw, hc = torch.isfinite(pos[..., 0]), torch.isfinite(pos_c[..., 0])
    # check: the surface point of a texel maps back onto the texel centre
    rows, cols = torch.nonzero(hw, as_tuple=True)
    pp = pos[rows, cols]
    e_u = (pp[:, 0] - (U0 + (cols.double() + 0.5) * du)).abs().max().item() * 1000
    e_t = ((torch.remainder(torch.atan2(-pp[:, 1], pp[:, 2]) - (-math.pi + (rows.double() + 0.5) * dth) + math.pi, 2 * math.pi)
            - math.pi).abs() * R_mean).max().item() * 1000
    log(f"mesh {len(V):,} vertices, {len(TRI):,} faces ({int(is_cap.sum()):,} end cap, {int(seam.sum()):,} across the seam, "
        f"{sk_w + sk_c} skipped as too large) rasterised in {time.time() - t_r:.0f} s: {hw.float().mean().item():.1%} of the "
        f"wall texels and {hc.float().mean().item():.1%} of the cap texels lie on a face; texel centre check max |du| "
        f"{e_u:.4f} mm, |dtheta| {e_t:.4f} mm")
    return pos, pos_c


ms = np.load(os.path.join(a.scene, a.mesh))
V = torch.from_numpy(ms["xyz"]).to(dev)                      # float64
TRI = torch.from_numpy(ms["tri"]).to(dev)
IS_CAP = cap_faces(V, TRI)
POS, POS_C = texel_points(V, TRI, IS_CAP)

# ------------------------------------------------------------------------------ projection into the views ------
views_all = torch.from_numpy(d["train_viewmats"]).to(dev)
train = np.setdiff1d(np.arange(len(views_all)), d["block_idx"])
train = train[np.argsort(d["train_eyes"][train, 0])]
if a.views:
    train = np.array([n for n in train if n in set(a.views)])
log(f"{len(train)} training poses (the {len(d['block_idx'])} block frames' poses are not used)")


def project(P, R_p, view, Dvis, Dm, img, alpha, cap, amin):
    """texel points P [n, 3] (radius from the pipe axis R_p) seen in this view: kept indices, footprint (m per raw
    pixel), depth, colour and the model's depth along the same ray"""
    R, t = view[:3, :3], view[:3, 3]
    C = -R.T @ t
    Xc = P @ R.T + t
    z = Xc[:, 2]
    ok = torch.isfinite(z) & (z > 0.05)
    zs = torch.where(ok, z, torch.ones_like(z))
    xn, yn = torch.nan_to_num(Xc[:, 0] / zs), torch.nan_to_num(Xc[:, 1] / zs)
    r2 = xn * xn + yn * yn
    ok &= r2 < R2MAX
    px, py = ccx + fx * xn, ccy + fy * yn
    ix, iy = px.round().long().clamp(0, CW - 1), py.round().long().clamp(0, CH - 1)
    ok &= (px > -0.5) & (px < CW - 0.5) & (py > -0.5) & (py < CH - 0.5) & VALID[iy, ix]
    Ds = Dvis[iy, ix]
    ok &= (Ds > 0) & ((z - Ds).abs() < 0.015 * Ds + 0.002)
    sel = torch.nonzero(ok)[:, 0]
    if len(sel) == 0:
        return sel, None, None, None, None
    grid = torch.stack([(px[sel] + 0.5) / CW * 2 - 1, (py[sel] + 0.5) / CH * 2 - 1], -1)[None, None]
    al = F.grid_sample(alpha[None, None], grid, align_corners=False)[0, 0, 0]
    col = F.grid_sample(img[None], grid, align_corners=False)[0, :, 0].T
    ray = P[sel] - C
    dist = ray.norm(dim=1).clamp(min=1e-6)
    if cap:
        cos = ray[:, 0].abs() / dist
    else:
        cos = (ray[:, 1] * P[sel, 1] + ray[:, 2] * P[sel, 2]).abs() / (dist * R_p[sel].clamp(min=1e-6))
    mag = 1 / (1 + 3 * k1 * r2[sel]) if k1 < 0 else torch.ones_like(cos)
    fp = z[sel] / fx / cos.clamp(min=0.05) * mag
    keep = al >= amin
    return sel[keep], fp[keep], z[sel][keep], col[keep], Dm[iy[sel], ix[sel]][keep]


def wall_window(view):
    C = -view[:3, :3].T @ view[:3, 3]
    c0 = max(0, int(math.floor((float(C[0]) - 0.2 - U0) / du)))
    c1 = min(NU, int(math.ceil((float(C[0]) + 1.5 - U0) / du)))
    if c1 <= c0:
        return None
    rows = torch.arange(NTH, device=dev)[:, None] * NU
    return (rows + torch.arange(c0, c1, device=dev)[None]).reshape(-1)


def end_col0():
    """first wall column of the end region's accumulators (3 cm before --end-x, room for the brightness blur)"""
    return min(NU - 1, max(0, int(math.floor((a.end_x - 0.03 - U0) / du))))


def passes(todo, vis):
    """run the passes in todo over the training views: 'fp_best', 'disp' (radial offset of the model's surface from
    the texel point along the ray), 'accumulate' (near / ref colour means), 'end' / 'endrob' (the end region's means,
    --end-x); vis = 'mesh' or 'model' depth for the visibility test. Returns the accumulators of the wall ('w') and
    the cap ('c'); the end region's are in acc[part]['end'] (wall: columns from end_col0() on)."""
    POS_f, POS_Cf = POS.view(-1, 3), POS_C.view(-1, 3)
    pts = {"w": (POS_f, torch.hypot(POS_f[:, 1], POS_f[:, 2])), "c": (POS_Cf, None)}
    acc = {}
    ec0 = end_col0() if a.end_x is not None else NU
    for part, n_ in (("w", NTH * NU), ("c", CAP_N * CAP_N)):
        acc[part] = dict(fpb=torch.full((n_,), float("inf"), device=dev), sN=torch.zeros(n_, device=dev),
                         sN2=torch.zeros(n_, device=dev), cN=torch.zeros(n_, 3, device=dev),
                         sF=torch.zeros(n_, device=dev), cF=torch.zeros(n_, 3, device=dev),
                         nv=torch.zeros(n_, device=dev), sD=torch.zeros(n_, device=dev), cD=torch.zeros(n_, device=dev))
        if any(p in todo for p in ("end", "endrob")):
            ne = NTH * (NU - ec0) if part == "w" else n_
            acc[part]["end"] = {k: torch.zeros((ne, 3) if k[0] == "c" else ne, device=dev)
                                for k in ("sN", "cN", "lN", "sF", "lF", "sR", "cR", "sFR", "cFR")}
    cap_all = torch.arange(CAP_N * CAP_N, device=dev)
    for pas in todo:
        t_p = time.time()
        for j, n in enumerate(train):
            view = views_all[int(n)]
            img, Dm, alpha = render_model(view)
            Dvis = surface_depth(view) if vis == "mesh" else torch.where(alpha > 0.5, Dm, torch.zeros_like(Dm))
            C = -view[:3, :3].T @ view[:3, 3]
            for part, gidx in (("w", wall_window(view)), ("c", cap_all)):
                if gidx is None:
                    continue
                if part == "w" and pas in ("end", "endrob"):
                    gidx = gidx[gidx % NU >= ec0]
                    if len(gidx) == 0:
                        continue
                P_all, R_all = pts[part]
                P = P_all[gidx]
                sel, fp, z, col, zm = project(P, None if R_all is None else R_all[gidx], view, Dvis, Dm, img, alpha,
                                              part == "c", 0.99 if pas == "disp" else a.alpha_min)
                if len(sel) == 0:
                    continue
                gi = gidx[sel]
                A = acc[part]
                if pas == "fp_best":
                    A["fpb"][gi] = torch.minimum(A["fpb"][gi], fp)
                    continue
                if pas in ("end", "endrob"):           # the end region: many views, highlights down-weighted
                    li = gi if part == "c" else torch.div(gi, NU, rounding_mode="floor") * (NU - ec0) + gi % NU - ec0
                    E = A["end"]
                    wE = torch.exp(-(fp / A["fpb"][gi] - 1).clamp(min=0) / a.end_tau)
                    wF = torch.exp(-((z - a.ref_z) / a.ref_sigma) ** 2) + 1e-4
                    lum = torch.log(col.mean(1).clamp(min=1e-3))
                    if pas == "end":
                        E["sN"][li] += wE
                        E["cN"][li] += wE[:, None] * col
                        E["lN"][li] += wE * lum
                        E["sF"][li] += wF
                        E["lF"][li] += wF * lum
                    else:
                        rN = torch.exp(-0.5 * ((lum - E["lN"][li] / E["sN"][li].clamp(min=1e-12)) / a.end_hl).clamp(min=0) ** 2)
                        rF = torch.exp(-0.5 * ((lum - E["lF"][li] / E["sF"][li].clamp(min=1e-12)) / a.end_hl).clamp(min=0) ** 2)
                        E["sR"][li] += wE * rN
                        E["cR"][li] += (wE * rN)[:, None] * col
                        E["sFR"][li] += wF * rF
                        E["cFR"][li] += (wF * rF)[:, None] * col
                    continue
                wN = torch.exp(-(fp / A["fpb"][gi] - 1).clamp(min=0) / a.near_tau)
                if pas == "disp":                   # where the model's surface is along this ray, radially / along x
                    Pk = P[sel]
                    Xg = C + (Pk - C) * (zm / z)[:, None]
                    dd = (Xg[:, 0] - Pk[:, 0]) if part == "c" else (torch.hypot(Xg[:, 1], Xg[:, 2]) - R_all[gi])
                    good = (zm > 0) & (dd.abs() < 0.01)
                    A["sD"][gi] += wN * good
                    A["cD"][gi] += wN * good * dd
                    continue
                wF = torch.exp(-((z - a.ref_z) / a.ref_sigma) ** 2) + 1e-4
                A["sN"][gi] += wN
                A["sN2"][gi] += wN * wN
                A["cN"][gi] += wN[:, None] * col
                A["sF"][gi] += wF
                A["cF"][gi] += wF[:, None] * col
                A["nv"][gi] += 1
            if j % 140 == 0:
                log(f"  {pas}: view {j + 1}/{len(train)} (x {float(d['train_eyes'][int(n), 0]):.2f} m), "
                    f"peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        log(f"pass {pas} (visibility from the {vis} depth) over {len(train)} views in {time.time() - t_p:.0f} s")
    return acc


# ------------------------------------------------------------------------------ texture-space filters ---------
def gauss_blur(x, m, sig, wrap):
    """normalised Gaussian blur of x [C, H, W] with weights m [H, W] (rows wrap with wrap=True)"""
    R = int(math.ceil(3 * sig))
    k = torch.exp(-0.5 * (torch.arange(-R, R + 1, device=dev, dtype=torch.float32) / sig) ** 2)
    k = k / k.sum()
    z = torch.cat([x * m[None], m[None]], 0)[None]
    nch = z.shape[1]
    z = F.conv2d(F.pad(z, (R, R, 0, 0)), k.view(1, 1, 1, -1).repeat(nch, 1, 1, 1), groups=nch)
    z = F.pad(z, (0, 0, R, R), mode="circular") if wrap else F.pad(z, (0, 0, R, R))
    z = F.conv2d(z, k.view(1, 1, -1, 1).repeat(nch, 1, 1, 1), groups=nch)[0]
    return z[:-1] / z[-1:].clamp(min=1e-8)


def fill(x, m):
    """push-pull fill of x [C, H, W] where m [H, W] is False"""
    pyr = [(x * m[None], m.float())]
    while min(pyr[-1][1].shape) > 1:
        x_, m2 = pyr[-1]
        pyr.append((F.avg_pool2d(x_[None], 2, ceil_mode=True)[0], F.avg_pool2d(m2[None, None], 2, ceil_mode=True)[0, 0]))
    out = pyr[-1][0] / pyr[-1][1].clamp(min=1e-12)
    for x_, m2 in reversed(pyr[:-1]):
        up = F.interpolate(out[None], size=m2.shape, mode="bilinear", align_corners=False)[0]
        out = torch.where(m2[None] > 0, x_ / m2.clamp(min=1e-12)[None], up)
    return out


def sample_map(mp, gu, gv, wrap):
    """bilinear sample of a map [H, W] at texel coordinates (gu column, gv row), rows circular with wrap"""
    H, W = mp.shape
    t = mp[None, None]
    if wrap:
        t = torch.cat([t[..., -1:, :], t, t[..., :1, :]], dim=-2)
        gv = torch.remainder(gv + 0.5, H) - 0.5 + 1
        H += 2
    grid = torch.stack([(gu + 0.5) / W * 2 - 1, (gv + 0.5) / H * 2 - 1], -1)[None, None].float()
    return F.grid_sample(t, grid, mode="bilinear", padding_mode="border", align_corners=False)[0, 0, 0]


def qs(t, ps):
    t = t[torch.randperm(len(t), device=dev)[:2000000]]
    return [round(float(v), 2) for v in torch.quantile(t.float(), torch.tensor(ps, device=dev))]


# ------------------------------------------------------------------------------ the model's surface ------------
vis = "mesh"
if not a.mesh_surface:
    acc0 = passes(["fp_best", "disp"], "mesh")
    disp = {}
    for part, (H, W, wrap) in (("w", (NTH, NU, True)), ("c", (CAP_N, CAP_N, False))):
        A = acc0[part]
        ok = (A["sD"] > 1e-3).view(H, W)
        dmap = (A["cD"] / A["sD"].clamp(min=1e-12)).view(1, H, W)
        dmap = fill(gauss_blur(dmap, ok.float(), a.disp_smooth_mm / a.texel_mm, wrap), ok)[0]
        disp[part] = dmap
        log(f"model surface {'wall (radial)' if part == 'w' else 'cap (along x)'}: offset from the mesh known for "
            f"{ok.float().mean().item():.1%} of the texels, p5/25/50/75/95 {qs(dmap[ok] * 1000, [0.05, 0.25, 0.5, 0.75, 0.95])} mm")
    np.save(os.path.join(a.out, "disp_wall_mm.npy"), (disp["w"][::4, ::4] * 1000).cpu().numpy().astype(np.float16))
    del acc0
    # move the mesh vertices: wall radially by the wall map, end-cap vertices along x by the cap map
    th_v = torch.atan2(-V[:, 1], V[:, 2])
    r_v = torch.hypot(V[:, 1], V[:, 2]).clamp(min=1e-6)
    dw = sample_map(disp["w"], (V[:, 0] - U0) / du - 0.5, (th_v + math.pi) / dth - 0.5, True).double()
    V2 = V.clone()
    V2[:, 1] += dw * V[:, 1] / r_v
    V2[:, 2] += dw * V[:, 2] / r_v
    cv = torch.unique(TRI[IS_CAP])
    dc = sample_map(disp["c"], (V[cv, 1] + 0.2) / du - 0.5, (V[cv, 2] + 0.2) / du - 0.5, False).double()
    V2[cv] = V[cv]
    V2[cv, 0] += dc
    np.savez(os.path.join(a.out, "mesh_bake.npz"), xyz=V2.cpu().numpy(), tri=TRI.cpu().numpy())
    A_, B_, C_ = V2[TRI[:, 0]], V2[TRI[:, 1]], V2[TRI[:, 2]]
    nrm = torch.cross(B_ - A_, C_ - A_, dim=1)
    area2 = nrm.norm(dim=1)
    keep = area2 > 1e-14
    cen = ((A_ + B_ + C_) / 3)[keep].float()
    np.savez(os.path.join(a.out, "surface_bake.npz"), centres=cen.cpu().numpy(),
             normals=(nrm[keep] / area2[keep, None]).float().cpu().numpy(), size=torch.sqrt(area2[keep]).float().cpu().numpy())
    log(f"mesh moved onto the model's surface: vertex offset p5/50/95 {qs(dw[r_v > 0.05] * 1000, [0.05, 0.5, 0.95])} mm "
        f"(wall), {qs(dc * 1000, [0.05, 0.5, 0.95])} mm (cap, along x); {int(keep.sum()):,} faces -> mesh_bake.npz, "
        f"surface_bake.npz")
    set_layout(cen, cen.cpu().numpy())
    V = V2
    IS_CAP = cap_faces(V, TRI)
    POS, POS_C = texel_points(V, TRI, IS_CAP)
    vis = "model"
    del A_, B_, C_, nrm, cen
END_PASSES = {None: [], "ref": [], "tau": ["end"], "robust": ["end", "endrob"]}[a.end_mode if a.end_x is not None else None]
acc = passes(["fp_best", "accumulate"] + END_PASSES, vis)
del GS, S_c, S_q, S_s, S_o, S_col
torch.cuda.empty_cache()

# ------------------------------------------------------------------------------ combination -------------------
sig = a.lp_mm / a.texel_mm
res = {}
for part, (H, W, wrap) in (("w", (NTH, NU, True)), ("c", (CAP_N, CAP_N, False))):
    A = acc[part]
    seen = (A["sN"] > 0).view(H, W)
    Nn = (A["cN"] / A["sN"].clamp(min=1e-12)[:, None]).T.reshape(3, H, W)
    Fr = (A["cF"] / A["sF"].clamp(min=1e-12)[:, None]).T.reshape(3, H, W)
    m = seen.float()
    ratio = (gauss_blur(Fr, m, sig, wrap) / gauss_blur(Nn, m, sig, wrap).clamp(min=1e-4)).clamp(0.1, 10)
    Tt = torch.where(seen[None], Nn * ratio, torch.zeros_like(Nn))
    res[part] = dict(T=fill(Tt, seen), N=fill(Nn, seen), F=fill(Fr, seen), seen=seen)
    if a.end_x is not None:                     # the end region's own colours (--end-x), blended in along the wall
        c0 = 0 if part == "c" else end_col0()
        We, se = W - c0, seen[:, c0:]
        Fe = Fr[:, :, c0:]
        if a.end_mode == "ref":
            Te = Fe
        else:
            E = acc[part]["end"]
            sN_, cN_ = (E["sR"], E["cR"]) if a.end_mode == "robust" else (E["sN"], E["cN"])
            Ne = (cN_ / sN_.clamp(min=1e-12)[:, None]).T.reshape(3, H, We)
            if a.end_mode == "robust":
                Fe = (E["cFR"] / E["sFR"].clamp(min=1e-12)[:, None]).T.reshape(3, H, We)
                kept = (E["sR"] / E["sN"].clamp(min=1e-12)).view(H, We)[se]
                log(f"{'cap' if part == 'c' else 'wall end'}: highlight down-weighting kept p5/50/95 "
                    f"{qs(kept, [0.05, 0.5, 0.95])} of the near weight")
            Te = Ne * (gauss_blur(Fe, se.float(), sig, wrap) / gauss_blur(Ne, se.float(), sig, wrap).clamp(min=1e-4)).clamp(0.1, 10)
        Te = fill(torch.where(se[None], Te, torch.zeros_like(Te)), se)
        if part == "c":
            s_ = torch.ones(We, device=dev)
        else:
            s_ = ((U0 + (c0 + torch.arange(We, device=dev) + 0.5) * du - a.end_x) / a.end_ramp).clamp(0, 1)
        T0_ = res[part]["T"][:, :, c0:]
        l0_, l1_ = T0_.mean(0)[se], Te.mean(0)[se]
        log(f"{'cap' if part == 'c' else f'wall beyond u {a.end_x:.3f} m'} ({a.end_mode}): luminance p50/95/99/99.9 "
            f"{qs(l0_, [0.5, 0.95, 0.99, 0.999])} -> {qs(l1_, [0.5, 0.95, 0.99, 0.999])}")
        res[part]["T"][:, :, c0:] = (1 - s_) * T0_ + s_ * Te
        del Te, T0_, Fe
    if part == "w":
        sv_ = seen.view(-1)
        has_w = torch.isfinite(POS[..., 0])
        log(f"wall: {seen.float().mean().item():.1%} of the texels seen ({(seen & has_w).float().sum().item() / max(has_w.float().sum().item(), 1):.1%} "
            f"of those on a face); best footprint p5/50/95 {qs(A['fpb'][sv_] * 1000, [0.05, 0.5, 0.95])} mm per raw pixel; "
            f"views per texel median {float(A['nv'][sv_].median()):.0f}, effective near views p5/50/95 "
            f"{qs((A['sN'] ** 2 / A['sN2'].clamp(min=1e-12))[sv_], [0.05, 0.5, 0.95])}; ref/near brightness ratio p5/50/95 "
            f"{qs(ratio.mean(0)[seen], [0.05, 0.5, 0.95])}")
        np.savez_compressed(os.path.join(a.out, "bake_stats.npz"),
                            fp_best_mm=(A["fpb"].view(H, W)[::4, ::4] * 1000).cpu().numpy().astype(np.float16),
                            neff=(A["sN"] ** 2 / A["sN2"].clamp(min=1e-12)).view(H, W)[::4, ::4].cpu().numpy().astype(np.float16),
                            ratio=ratio.mean(0)[::4, ::4].cpu().numpy().astype(np.float16))
    else:
        log(f"cap: {seen.float().mean().item():.1%} of the texels seen")
    del ratio, Tt, Nn, Fr


# ------------------------------------------------------------------------------ outputs -----------------------
def to8(t, scale):
    return ((t / scale).clamp(0, 1) * 255).round().to(torch.uint8)


def png_scale(t, seen):
    vals = t.mean(0)[seen]
    return float(torch.quantile(vals[torch.randperm(len(vals), device=dev)[:2000000]], 0.995)) / 250 * 255


def save_png(path, t8):
    cv2.imwrite(path, t8.permute(1, 2, 0).cpu().numpy()[:, :, ::-1])


extra = dict(gain=torch.zeros(len(views_all), 3), off=torch.zeros(len(views_all), 3), r_ref=None, U0=U0, U1=U1, du=du,
             dth=dth, NU=NU, NTH=NTH, cap_x=cap_x, R_mean=R_mean)
for kind, fname in (("T", "tex.pt"), ("N", "tex_near.pt")):
    tw, tc = res["w"][kind], res["c"][kind]
    scale = png_scale(tw, res["w"]["seen"])
    w8, c8 = to8(tw, scale), to8(tc, scale)
    lin = lambda t8: torch.log((t8.float() / 255 * scale).clamp(min=1e-4))[None].cpu()  # noqa: E731
    torch.save(dict(tex=lin(w8), cap=lin(c8)) | extra | dict(png_scale=scale, baked=kind), os.path.join(a.out, fname))
    cv2.imwrite(os.path.join(a.out, f"preview_{kind}.jpg"), w8[:, ::4, ::4].permute(1, 2, 0).cpu().numpy()[:, :, ::-1],
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    if kind == "T":
        save_png(os.path.join(a.out, "texture.png"), w8)
        save_png(os.path.join(a.out, "cap.png"), c8)
        json.dump(dict(U0=U0, U1=U1, du=du, dth=dth, NU=NU, NTH=NTH, cap_x=cap_x, cap_half=0.2, R_mean=R_mean, png_scale=scale,
                       inpipe=False, baked_from=a.run,
                       surface=None if a.mesh_surface else os.path.join(a.out, "surface_bake.npz"),
                       mesh=None if a.mesh_surface else os.path.join(a.out, "mesh_bake.npz"),
                       bake=dict(model_surface=not a.mesh_surface, near_tau=a.near_tau, ref_z=a.ref_z, ref_sigma=a.ref_sigma,
                                 lp_mm=a.lp_mm, alpha_min=a.alpha_min, disp_smooth_mm=a.disp_smooth_mm, mesh_in=a.mesh,
                                 end_x=a.end_x, end_mode=a.end_mode if a.end_x is not None else None,
                                 end_tau=a.end_tau, end_hl=a.end_hl, end_ramp=a.end_ramp),
                       note="texture.png: column c covers u = U0 + c du, row r covers theta = -pi + r dth (rad); "
                            "cap.png: column = (y + 0.2) / du, row = (z + 0.2) / du; video colour values (not CLAHE)"),
                  open(os.path.join(a.out, "texture.json"), "w", newline="\n"), indent=1)
    log(f"{fname}: png scale {scale:.4f}; {(w8 == 255).all(0).float().mean().item():.2%} of the wall texels white, "
        f"{(w8 == 0).all(0).float().mean().item():.2%} black")
fr = res["w"]["F"]
cv2.imwrite(os.path.join(a.out, "preview_F.jpg"), to8(fr, png_scale(fr, res["w"]["seen"]))[:, ::4, ::4]
            .permute(1, 2, 0).cpu().numpy()[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
log("done")
