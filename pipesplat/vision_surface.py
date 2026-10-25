"""A clean, image-consistent pipe wall from the vision dense cloud (docs/11 Y1): a height field r(u, theta) around a
centerline fitted to the vision points themselves. Runs in the pipe3d WSL distro (numpy + scipy).

Why (docs/10 X7, 5.3 item 1, 5.5): on InPipe the Gaussian model is anchored on the OpenMVS mesh of the vision
reconstruction - the only geometry that agrees with the images at the mm level - but on the dark, smooth wall that
mesh has holes and flaps (the dense cloud has 4-12 k points per 10 cm there, 100-300 k where the wall is textured),
so the renders show dark blotches and stripes, and from outside the model is a cloud of floaters. The LiDAR height
field R3 is clean but 13-15 % too small and ~14 mm off-centre against the images (X6). This script keeps the vision
geometry and adds the prior that the wall is a pipe:
  1. axis: start from the R2 centerline (--centerline, LiDAR; InPipe) or from the smoothed path of the training
     cameras (no --centerline: Pipe1, where the robot rides off-axis) and move it onto the vision points: per 1 cm node
     r(theta) = R + a cos(theta) + b sin(theta) + c cos(2 theta) + d sin(2 theta) (radius, centre offset, ellipse),
     one sparse robust least-squares fit over all points with a second-difference smoothness along u (dense parts
     do not outvote sparse ones: every 1 cm bin weighs at most --bin-cap points), Huber IRLS; the centerline is then
     moved by (a, b) and the fit repeated (--rounds), so the offset is not a first-order approximation
  2. detail: the residual of the points from that smooth model, cell medians on a --du-mm x --dth-deg grid, cells
     that disagree with their neighbourhood by more than --spike-mm dropped, smoothed (normalised Gaussian) and faded
     to 0 where the cloud is sparse (confidence n / (n + --conf-n)), clipped to +-(--detail-max-mm): joints and dents
     where the cloud sees them, the smooth pipe where it does not (no holes, no flaps). Points farther than
     --detail-window-mm from the smooth model are not detail (InPipe 12 mm; Pipe1 larger: the water surface)
  3. openings (--r3, InPipe): the bottom slot and the pipe's ends are where the R3 LiDAR mesh (InPipe's surface.npz,
     discs of the R3 mesh up to the exit) has no wall - taken from it in the new frame (2 deg x 1 cm occupancy,
     closed by 1 cell), so the new wall is open exactly where the real pipe is. Without --r3: a closed tube over
     the 0.5-99.5 % range of the wall points along u
  4. outputs in SCENE_DIR: surface_vhf.npz (the measured surface for train_gs.py --anchor --surface: the height-field
     triangles plus the vision mesh's triangles that lie outside the pipe - beyond the exit or in the slot - so the
     AprilTag board and the floor seen through the slot keep their geometry), mesh_vhf.npz (the tube only: xyz, tri,
     the grid r(u, theta) and every vertex's cell, for baking / viewers: bake_vhf.py), centerline_vhf.npz
     (Centerline layout), vhf.json (fit report), vhf_*.png (maps)

usage: /opt/gsenv/bin/python pipesplat/vision_surface.py SCENE_DIR [--centerline CL.npz --r3 R3_SURFACE.npz]
           [--band LO HI] [--rounds 3] [--du-mm 2] [--dth-deg 1] [--node-cm 1] [--smooth-cm 5] [--detail-max-mm 8]
           [--detail-window-mm 12] [--no-detail]
  InPipe (docs/11 Y1): --centerline centerline_r2.npz --r3 ../inpipe_172011/surface.npz --band 0.065 0.150
  Pipe1  (docs/11 Y3): --rounds 6 --detail-window-mm 40 --detail-max-mm 30
"""
import argparse
import json
import math
import os
import time

import cv2
import numpy as np
import scipy.sparse as sp
from scipy.ndimage import binary_closing, gaussian_filter
from scipy.spatial import cKDTree
from scipy.sparse.linalg import spsolve

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("scene")
ap.add_argument("--centerline", default=None, help="starting axis (recon_common.Centerline file); default: camera path")
ap.add_argument("--r3", default=None, help="R3 LiDAR surface discs: where the real pipe is open (slot, ends)")
ap.add_argument("--band", type=float, nargs=2, default=None,
                help="first round: wall points between these radii (m); default 0.45..1.6 x the median radius")
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--axis-smooth-m", type=float, default=0.5, help="camera path smoothing for the starting axis")
ap.add_argument("--keep-mm", type=float, default=15.0, help="later rounds: points within this of the model")
ap.add_argument("--detail-window-mm", type=float, default=12.0)
ap.add_argument("--du-mm", type=float, default=2.0)
ap.add_argument("--dth-deg", type=float, default=1.0)
ap.add_argument("--node-cm", type=float, default=1.0)
ap.add_argument("--smooth-cm", type=float, default=5.0, help="smoothing length of the axis / radius fit along u")
ap.add_argument("--bin-cap", type=int, default=2000)
ap.add_argument("--spike-mm", type=float, default=3.0)
ap.add_argument("--conf-n", type=float, default=12.0)
ap.add_argument("--detail-max-mm", type=float, default=8.0)
ap.add_argument("--no-detail", action="store_true", help="the smooth model only (no residual height field)")
ap.add_argument("--name", default="vhf", help="output names surface_<name>.npz, mesh_<name>.npz, ...")
a = ap.parse_args()
T0 = time.time()


def log(msg):
    print(f"{msg} [{time.time() - T0:.0f} s]", flush=True)


# ------------------------------------------------------------------------------ centerline -------------------
class CL:
    """centerline samples every ds (u, C, T, N, Bn, th0) with coords() / point() as recon_common.Centerline"""

    def __init__(self, u, C, up, th0):
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))])
        ds = 0.001
        a_new = np.arange(0.0, arc[-1], ds)
        C = np.column_stack([np.interp(a_new, arc, C[:, j]) for j in range(3)])
        self.u = u[0] + a_new
        T = np.gradient(C, axis=0)
        T /= np.linalg.norm(T, axis=1, keepdims=True)
        N = np.zeros_like(C)
        n = up - (up @ T[0]) * T[0]
        N[0] = n / np.linalg.norm(n)
        for i in range(1, len(C)):                         # parallel transport, as recon_common.Centerline.build
            n = N[i - 1] - (N[i - 1] @ T[i]) * T[i]
            N[i] = n / np.linalg.norm(n)
        self.C, self.T, self.N, self.Bn, self.th0, self.ds = C, T, N, np.cross(T, N), th0, ds

    def coords(self, X):
        """(u m, theta rad in (-pi, pi], r m)"""
        u = self.u[::10][cKDTree(self.C[::10]).query(X)[1]]      # nearest of every 10th sample
        for _ in range(4):
            i = np.clip(np.round((u - self.u[0]) / self.ds).astype(int), 0, len(self.u) - 1)
            u = self.u[i] + np.einsum("ij,ij->i", X - self.C[i], self.T[i])
        i = np.clip(np.round((u - self.u[0]) / self.ds).astype(int), 0, len(self.u) - 1)
        q = X - self.C[i]
        x, y = np.einsum("ij,ij->i", q, self.N[i]), np.einsum("ij,ij->i", q, self.Bn[i])
        th = np.remainder(np.arctan2(y, x) - math.radians(self.th0) + math.pi, 2 * math.pi) - math.pi
        return u, th, np.hypot(x, y)

    def frame(self, u):
        i = np.clip((np.asarray(u, float) - self.u[0]) / self.ds, 0, len(self.u) - 1.000001)
        i0 = np.floor(i).astype(int)
        w = (i - i0)[..., None]
        return [(1 - w) * M[i0] + w * M[i0 + 1] for M in (self.C, self.N, self.Bn, self.T)]

    def point(self, u, th, r):
        C, N, Bn, _ = self.frame(u)
        t = np.asarray(th, float) + math.radians(self.th0)
        return C + np.asarray(r, float)[..., None] * (np.cos(t)[..., None] * N + np.sin(t)[..., None] * Bn)

    def save(self, path):
        np.savez(path, u=self.u, C=self.C, T=self.T, N=self.N, Bn=self.Bn, th0=self.th0)


d = np.load(os.path.join(a.scene, "scene.npz"))
P = d["points"].astype(np.float64)
eyes = d["train_eyes"].astype(np.float64)
ext = 0.6
if a.centerline:
    z = np.load(a.centerline)
    u_r2, C_r2, t_lo, t_hi, up, th0 = z["u"], z["C"], z["T"][0], z["T"][-1], z["N"][0], float(z["th0"])
    what = "R2 centerline"
else:                                           # the training cameras' path, smoothed, as the starting axis
    e = eyes[np.argsort((eyes - eyes.mean(0)) @ np.linalg.svd(eyes - eyes.mean(0))[2][0])]
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(e, axis=0), axis=1))])
    uu = np.arange(0.0, arc[-1], 0.001)
    C_r2 = np.column_stack([np.interp(uu, arc, e[:, j]) for j in range(3)])
    k = int(a.axis_smooth_m / 0.001) | 1
    C_r2 = np.column_stack([np.convolve(np.pad(C_r2[:, j], k // 2, mode="edge"), np.ones(k) / k, mode="valid")
                            for j in range(3)])
    u_r2 = uu
    T_ = np.gradient(C_r2, axis=0)
    t_lo, t_hi = T_[0] / np.linalg.norm(T_[0]), T_[-1] / np.linalg.norm(T_[-1])
    up, th0 = np.cross(t_lo, [0.0, 0.0, 1.0]) if abs(t_lo[2]) < 0.9 else np.cross(t_lo, [1.0, 0.0, 0.0]), 0.0
    what = f"camera path (smoothed {a.axis_smooth_m} m)"
# extend the starting axis straight along its end tangents so that it covers every camera and point
k_lo, k_hi = int(ext / 0.001), int(ext / 0.001)
C_ext = np.concatenate([C_r2[0] - np.outer(np.arange(k_lo, 0, -1) * 0.001, t_lo), C_r2,
                        C_r2[-1] + np.outer(np.arange(1, k_hi + 1) * 0.001, t_hi)])
u_ext = np.concatenate([u_r2[0] - np.arange(k_lo, 0, -1) * 0.001, u_r2, u_r2[-1] + np.arange(1, k_hi + 1) * 0.001])
cl = CL(u_ext, C_ext, np.asarray(up, float), th0)
log(f"{len(P):,} vision dense points, {len(eyes)} training cameras; starting axis = {what}, u {u_r2[0]:.3f}..{u_r2[-1]:.3f} m "
    f"extended to {cl.u[0]:.3f}..{cl.u[-1]:.3f}")

# ------------------------------------------------------------------------------ openings from R3 -------------
r3 = np.load(a.r3)["centres"].astype(np.float64) if a.r3 else None


def openings(cl):
    """(u range of the R3 wall, occupancy grid 1 cm x 2 deg of R3 in this frame, its u0); without R3 a closed
    tube (nothing outside the u range of the points)"""
    if r3 is None:
        return -np.inf, np.inf, np.ones((1, 180), bool)
    u, th, _ = cl.coords(r3)
    lo, hi = np.percentile(u, 0.05), np.percentile(u, 99.95)
    nu, nt = int(math.ceil((hi - lo) / 0.01)) + 1, 180
    iu = np.clip(((u - lo) / 0.01).astype(int), 0, nu - 1)
    it = np.clip(((th + math.pi) / (2 * math.pi) * nt).astype(int), 0, nt - 1)
    occ = np.zeros((nu, nt), bool)
    occ[iu, it] = True
    occ = binary_closing(np.pad(occ, ((1, 1), (1, 1)), mode="wrap"), np.ones((3, 3)))[1:-1, 1:-1] | occ
    return lo, hi, occ


# ------------------------------------------------------------------------------ smooth model fit -------------
def basis(th):
    return np.stack([np.ones_like(th), np.cos(th), np.sin(th), np.cos(2 * th), np.sin(2 * th)], 1)


def fit(cl, u, th, r, keep, prev=None):
    """per-node (R, a, b, c, d) with smoothness; returns node u, params (n, 5), point weights used"""
    h = a.node_cm / 100
    lo, hi = np.floor(u[keep].min() / h) * h - h, np.ceil(u[keep].max() / h) * h + h
    nodes = np.arange(lo, hi + h / 2, h)
    n = len(nodes)
    s = (u - lo) / h
    i0 = np.clip(np.floor(s).astype(int), 0, n - 2)
    w1 = np.clip(s - i0, 0, 1)
    B = basis(th)
    sel = np.flatnonzero(keep)
    cnt = np.bincount(i0[sel], minlength=n).astype(float)
    wb = np.minimum(1.0, a.bin_cap / np.maximum(cnt[i0[sel]], 1))          # every 1 cm bin weighs <= bin_cap points
    delta = 0.010 if prev is None else 0.003
    wr = np.ones(len(sel))
    for it in range(6):
        ww = wb * wr
        rows = np.repeat(np.arange(len(sel)), 10)
        cols = np.concatenate([(i0[sel, None] * 5 + np.arange(5)), ((i0[sel] + 1)[:, None] * 5 + np.arange(5))], 1).ravel()
        vals = np.concatenate([(1 - w1[sel])[:, None] * B[sel], w1[sel][:, None] * B[sel]], 1).ravel()
        A = sp.csr_matrix((vals, (rows, cols)), shape=(len(sel), 5 * n))
        W = sp.diags(ww)
        lam = a.bin_cap * (a.smooth_cm / a.node_cm) ** 4
        D2 = sp.diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n))
        S = sp.kron(D2.T @ D2, sp.identity(5)) * lam
        ridge = sp.diags(np.tile([0, 0, 0, 1, 1], n) * 50.0 + 1e-6)          # ellipse terms only where the data say so
        M = (A.T @ W @ A + S + ridge).tocsc()
        x = spsolve(M, A.T @ (ww * r[sel]))
        res = A @ x - r[sel]
        wr = np.where(np.abs(res) < delta, 1.0, delta / np.maximum(np.abs(res), 1e-9))
    return nodes, x.reshape(n, 5), np.abs(res)


def model_r(nodes, prm, u, th):
    pu = np.column_stack([np.interp(u, nodes, prm[:, j]) for j in range(5)])
    return (pu * basis(th)).sum(1)


u_lo3, u_hi3, occ = openings(cl)
prev = None
for rnd in range(a.rounds):
    u, th, r = cl.coords(P)
    if prev is None:
        if a.band:
            band_lo, band_hi = a.band
        else:                                   # around the median distance of the points near the cameras
            near = np.abs(u - np.clip(u, *np.percentile(cl.coords(eyes)[0], [0, 100]))) < 0.01
            rm = float(np.median(r[near & (r < 2.0)]))
            band_lo, band_hi = 0.45 * rm, 1.6 * rm
        keep = (r > band_lo) & (r < band_hi)
        log(f"first round: wall band {band_lo * 1000:.0f}..{band_hi * 1000:.0f} mm")
    else:
        keep = np.abs(r - model_r(*prev, u, th)) < a.keep_mm / 1000
    if r3 is not None:
        # not in the slot / beyond the R3 wall's ends (points there are the floor, the board, the room)
        iu = ((u - u_lo3) / 0.01).astype(int)
        it = np.clip(((th + math.pi) / (2 * math.pi) * occ.shape[1]).astype(int), 0, occ.shape[1] - 1)
        inside_u = (iu >= 0) & (iu < occ.shape[0])
        wall = np.zeros(len(P), bool)
        wall[inside_u] = occ[iu[inside_u], it[inside_u]]
        keep &= wall | ~inside_u & (u < u_lo3)              # before the R3 wall starts: the vision cloud alone
    nodes, prm, ares = fit(cl, u, th, r, keep, prev)
    # move the centerline by the fitted offset (a, b) and refit
    Cn, Nn, Bnn, _ = cl.frame(cl.u)
    t0 = math.radians(cl.th0)
    av, bv = np.interp(cl.u, nodes, prm[:, 1]), np.interp(cl.u, nodes, prm[:, 2])
    # r ~ R + a cos(th) + b sin(th): the centre is at (a, b) in the (theta) frame -> world offset
    off = np.outer(av, 1) * (math.cos(t0) * Nn + math.sin(t0) * Bnn) + np.outer(bv, 1) * (-math.sin(t0) * Nn + math.cos(t0) * Bnn)
    log(f"round {rnd + 1}: {keep.sum():,} wall points, robust residual median {np.median(ares) * 1000:.2f} mm; "
        f"centre offset median {np.median(np.hypot(prm[:, 1], prm[:, 2])) * 1000:.1f} mm (max "
        f"{np.max(np.hypot(prm[:, 1], prm[:, 2])) * 1000:.1f}); radius median {np.median(prm[:, 0]) * 1000:.1f} mm")
    cl = CL(cl.u, Cn + off, Nn[0], cl.th0)
    u_lo3, u_hi3, occ = openings(cl)
    prev = (nodes, prm * np.array([1, 0, 0, 1, 1]))      # next round: offset now in the axis
u, th, r = cl.coords(P)
nodes, prm, ares = fit(cl, u, th, r, np.abs(r - model_r(*prev, u, th)) < a.keep_mm / 1000, prev)
prev = (nodes, prm)
eu, eth, er = cl.coords(eyes)
log(f"final axis: residual offset median {np.median(np.hypot(prm[:, 1], prm[:, 2])) * 1000:.2f} mm; cameras r median "
    f"{np.median(er) * 1000:.1f} mm (max {er.max() * 1000:.1f}); R3 wall u {u_lo3:.3f}..{u_hi3:.3f} m in the new frame")

# ------------------------------------------------------------------------------ height field ----------------
on_wall = np.abs(r - model_r(*prev, u, th)) < 0.01
u_start = float(np.percentile(u[on_wall], 0.5)) if r3 is None else min(u_lo3, float(np.percentile(u[on_wall], 0.5)))
u_end = u_hi3 if r3 is not None else float(np.percentile(u[on_wall], 99.5))
du, dth = a.du_mm / 1000, math.radians(a.dth_deg)
nu, nt = int(math.ceil((u_end - u_start) / du)), int(round(2 * math.pi / dth))
uc = u_start + (np.arange(nu) + 0.5) * du
tc = -math.pi + (np.arange(nt) + 0.5) * dth
UU, TT = np.meshgrid(uc, tc, indexing="ij")
R0 = model_r(nodes, prm, UU.ravel(), TT.ravel()).reshape(nu, nt)
res = r - model_r(nodes, prm, u, th)
m = (np.abs(res) < a.detail_window_mm / 1000) & (u >= u_start) & (u < u_end)
iu = ((u[m] - u_start) / du).astype(int)
it = np.clip(((th[m] + math.pi) / dth).astype(int), 0, nt - 1)
key = iu * nt + it
o = np.lexsort((res[m], key))
ks, vs = key[o], res[m][o]
st = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
cn = np.diff(np.r_[st, len(ks)])
med = np.full(nu * nt, np.nan)
N = np.zeros(nu * nt)
med[ks[st]] = vs[st + (cn - 1) // 2]
N[ks[st]] = cn
med, N = med.reshape(nu, nt), N.reshape(nu, nt)
med[N < 2] = np.nan
filled = np.where(np.isfinite(med), med, 0.0)
wgt = np.isfinite(med).astype(float)
sm = gaussian_filter(filled * wgt, (5, 5), mode=("nearest", "wrap"))
sw = gaussian_filter(wgt, (5, 5), mode=("nearest", "wrap"))
ref = np.where(sw > 1e-3, sm / np.maximum(sw, 1e-9), 0.0)
spike = np.isfinite(med) & (np.abs(med - ref) > a.spike_mm / 1000)
med[spike] = np.nan
wgt = np.where(np.isfinite(med), np.minimum(N, 10.0), 0.0)
sig = 1.5
sm = gaussian_filter(np.where(np.isfinite(med), med, 0) * wgt, sig, mode=("nearest", "wrap"))
sw = gaussian_filter(wgt, sig, mode=("nearest", "wrap"))
detail = np.where(sw > 1e-6, sm / np.maximum(sw, 1e-9), 0.0)
conf = sw * (2 * math.pi * sig ** 2) / (sw * (2 * math.pi * sig ** 2) + a.conf_n)
detail = np.clip(detail * conf, -a.detail_max_mm / 1000, a.detail_max_mm / 1000)
if a.no_detail:
    detail[:] = 0
RH = R0 + detail
log(f"height field {nu} x {nt} cells ({a.du_mm} mm x {a.dth_deg} deg), u {u_start:.3f}..{u_end:.3f} m; cells with "
    f"points {np.isfinite(med).mean():.1%} (spikes dropped {spike.sum():,}); detail |h| median "
    f"{np.median(np.abs(detail[conf > 0.5])) * 1000 if (conf > 0.5).any() else 0:.2f} mm, p95 "
    f"{np.percentile(np.abs(detail), 95) * 1000:.2f} mm; confidence > 0.5 on {np.mean(conf > 0.5):.1%} of the wall")

# openings: R3 occupancy in this frame (the slot, the ends)
if r3 is not None:
    iu3 = np.clip(((uc - u_lo3) / 0.01).astype(int), 0, occ.shape[0] - 1)
    it3 = np.clip(((tc + math.pi) / (2 * math.pi) * occ.shape[1]).astype(int), 0, occ.shape[1] - 1)
    open_ = ~occ[iu3][:, it3]
    before = uc < u_lo3                                 # before the R3 wall: closed tube (vision only)
    open_[before] = False
    open_ = binary_closing(open_, np.ones((3, 3))) & ~before[:, None]
else:
    open_ = np.zeros((nu, nt), bool)
log(f"openings (slot, exit) {open_.mean():.1%} of the cells")

# ------------------------------------------------------------------------------ mesh + outputs --------------
vid = np.full((nu, nt), -1, np.int64)
good = ~open_
vid[good] = np.arange(good.sum())
xyz = cl.point(UU[good], TT[good], RH[good])
i, j = np.meshgrid(np.arange(nu - 1), np.arange(nt), indexing="ij")
jn = (j + 1) % nt                                       # wrap around in theta
q = np.stack([vid[i, j], vid[i + 1, j], vid[i + 1, jn], vid[i, jn]], -1).reshape(-1, 4)
q = q[(q >= 0).all(1)]
tri = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
A_, B_, C_ = xyz[tri[:, 0]], xyz[tri[:, 1]], xyz[tri[:, 2]]
nrm = np.cross(B_ - A_, C_ - A_)
area2 = np.linalg.norm(nrm, axis=1)
cen = (A_ + B_ + C_) / 3
# orient the normals towards the axis (the camera side), as OpenMVS meshes are seen from inside
Cc = cl.frame(cl.coords(cen)[0])[0]
nrm = nrm / np.maximum(area2[:, None], 1e-15)
flip = np.einsum("ij,ij->i", nrm, Cc - cen) < 0
tri[flip] = tri[flip][:, [0, 2, 1]]
nrm[flip] *= -1
vi, vj = np.nonzero(good)                               # grid cell of every vertex (same order as xyz)
np.savez(os.path.join(a.scene, f"mesh_{a.name}.npz"), xyz=xyz.astype(np.float32), tri=tri.astype(np.int32),
         vi=vi.astype(np.int32), vj=vj.astype(np.int32), u0=u_start, du=du, dth=dth, r=RH.astype(np.float32),
         open=open_)
cl.save(os.path.join(a.scene, f"centerline_{a.name}.npz"))
# measured surface for anchoring: the tube + the vision mesh's triangles outside it (beyond the exit / in the slot)
vs = np.load(os.path.join(a.scene, "surface.npz"))
vu, vth, vr = cl.coords(vs["centres"].astype(np.float64))
viu = np.clip(((vu - u_start) / du).astype(int), 0, nu - 1)
vit = np.clip(((vth + math.pi) / dth).astype(int), 0, nt - 1)
out_tube = (vu >= u_end) | (open_[viu, vit] & (vr > RH[viu, vit] + 0.005) & (vu > u_start))
keep_cen = np.concatenate([cen, vs["centres"][out_tube]]).astype(np.float32)
keep_n = np.concatenate([nrm, vs["normals"][out_tube]]).astype(np.float32)
keep_s = np.concatenate([np.sqrt(area2), vs["size"][out_tube]]).astype(np.float32)
np.savez(os.path.join(a.scene, f"surface_{a.name}.npz"), centres=keep_cen, normals=keep_n, size=keep_s)
log(f"mesh {len(xyz):,} vertices, {len(tri):,} triangles (size median {np.median(np.sqrt(area2)) * 1000:.2f} mm); "
    f"surface_{a.name}.npz = tube + {out_tube.sum():,} of {len(vu):,} vision triangles outside the pipe")

# report: radius (InPipe: per leg, against the independent measurements of docs/10 X6)
rep = dict(points=int(len(P)), wall_points=int(keep.sum()), u_range=[u_start, u_end], r3_u=[float(u_lo3), float(u_hi3)],
           radius_mm_median=float(np.median(prm[:, 0]) * 1000),
           radius_mm_p5_p95=[float(v) for v in np.percentile(prm[:, 0], [5, 95]) * 1000],
           ellipse_mm_median=float(np.median(np.hypot(prm[:, 3], prm[:, 4])) * 1000),
           camera_r_mm_median=float(np.median(er) * 1000), cells_with_points=float(np.isfinite(med).mean()),
           detail_p95_mm=float(np.percentile(np.abs(detail), 95) * 1000), openings=float(open_.mean()),
           vertices=int(len(xyz)), triangles=int(len(tri)), outside_triangles=int(out_tube.sum()), args=vars(a))
if r3 is not None:
    leg1, leg2 = nodes < -0.05, nodes > 0.05
    rep.update(radius_leg1_mm=float(np.median(prm[leg1, 0]) * 1000), radius_leg2_mm=float(np.median(prm[leg2, 0]) * 1000))
    log(f"radius leg 1 {rep['radius_leg1_mm']:.1f} mm, leg 2 {rep['radius_leg2_mm']:.1f} mm "
        f"(RealSense depth 107.5 / 101.4, vision x tag 110.0 / 100.5, R3 LiDAR 94.5 / 89.1)")
log(f"radius median {rep['radius_mm_median']:.1f} mm (p5..p95 {rep['radius_mm_p5_p95'][0]:.1f}..{rep['radius_mm_p5_p95'][1]:.1f}); "
    f"ellipse median {rep['ellipse_mm_median']:.1f} mm")
json.dump(rep, open(os.path.join(a.scene, f"{a.name}.json"), "w"), indent=1)
rlo, rhi = np.percentile(RH, [2, 98])
# maps: radius (model + detail), detail, point count
img = lambda X, lo, hi: cv2.applyColorMap(np.clip((X - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8).T, cv2.COLORMAP_JET)  # noqa: E731
cv2.imwrite(os.path.join(a.scene, f"{a.name}_radius.png"), np.where(open_.T[..., None], 0, img(RH, rlo, rhi)))
cv2.imwrite(os.path.join(a.scene, f"{a.name}_detail.png"), np.where(open_.T[..., None], 0, img(detail, -0.004, 0.004)))
cv2.imwrite(os.path.join(a.scene, f"{a.name}_count.png"), img(np.log1p(N), 0, 4))
log("done")
