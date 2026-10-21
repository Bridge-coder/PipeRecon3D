"""Is the lamp factor of train_gs.py --lamp noisy? Compares, on held-out views, the incidence-angle cosine from
(a) depth differences of the rendered measured surface (what train_gs.py used first) and
(b) a normal map rendered directly from the surface discs (colours = normals in camera coordinates),
and reports the high-frequency share of each (the 1-3 px band, like imgmetrics.detail_corr).

usage: /opt/gsenv/bin/python pipesplat/diag_lamp.py SCENE_DIR RUN_DIR [--frames 0 22 43]
writes RUN_DIR/diag_cos_<frame>.png: [cos from depth | cos from normal map | lamp factor (depth version)]
"""
import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import gsplat

ap = argparse.ArgumentParser()
ap.add_argument("scene"), ap.add_argument("run"), ap.add_argument("--frames", type=int, nargs="+", default=[0, 22, 43])
ap.add_argument("--smooth-mm", type=float, default=0, help="also average the normals over this radius (as train_gs.py --lamp)")
a = ap.parse_args()
dev = "cuda"
d = np.load(os.path.join(a.scene, "scene.npz"))
sf = np.load(os.path.join(a.scene, "surface.npz"))
m = torch.load(os.path.join(a.run, "model.pt"))
Kc, (CW, CH) = m["K"].to(dev), m["size"]
fx, fy, ccx, ccy = Kc[0, 0].item(), Kc[1, 1].item(), Kc[0, 2].item() - 0.5, Kc[1, 2].item() - 0.5
S_c = torch.from_numpy(sf["centres"]).to(dev)
S_n = torch.from_numpy(sf["normals"]).to(dev)


def quat_z_to(n):
    z = torch.tensor([0.0, 0.0, 1.0], device=n.device)
    axis = torch.cross(z.expand_as(n), n, dim=1)
    s = axis.norm(dim=1, keepdim=True)
    ang = torch.atan2(s, n[:, 2:3].clamp(-1, 1))
    axis = torch.where(s > 1e-8, axis / s.clamp(min=1e-12), torch.tensor([1.0, 0, 0], device=n.device).expand_as(n))
    return torch.cat([torch.cos(ang / 2), axis * torch.sin(ang / 2)], dim=1)


S_q = quat_z_to(S_n)


def smooth_normals(c, n, radius):
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


S_nr = smooth_normals(S_c, S_n, a.smooth_mm / 1000) if a.smooth_mm > 0 else S_n
S_s = torch.log(torch.stack([0.6 * torch.from_numpy(sf["size"]).to(dev)] * 2 + [torch.full((len(S_c),), 1e-5, device=dev)], 1))
S_o = torch.full((len(S_c),), 0.999, device=dev)
PX = torch.from_numpy(((np.arange(CW) - ccx) / fx).astype(np.float32)).to(dev)
PY = torch.from_numpy(((np.arange(CH) - ccy) / fy).astype(np.float32)).to(dev)
t = m["lamp_t"].to(dev) if "lamp_t" in m else torch.zeros(3, device=dev)
logM = m["lamp_logM"].to(dev) if "lamp_logM" in m else None


@torch.no_grad()
def surface(view):
    """depth (ED) and camera-frame normal map of the measured surface"""
    nc = S_nr @ view[:3, :3].T                                    # normals in camera coordinates
    out, al, _ = gsplat.rasterization(S_c, S_q, torch.exp(S_s), S_o, nc, view[None], Kc[None], CW, CH,
                                      packed=False, render_mode="RGB+ED", near_plane=0.01)
    ok = al[0, ..., 0] > 0.5
    D = torch.where(ok, out[0, ..., 3], torch.zeros_like(out[0, ..., 3]))
    N = out[0, ..., :3] / al[0].clamp(min=1e-6)
    return D, N / N.norm(dim=-1, keepdim=True).clamp(min=1e-12), ok


def cos_depth(D):
    X = torch.stack([PX[None, :] * D, PY[:, None] * D, D], -1)
    dx, dy = torch.zeros_like(X), torch.zeros_like(X)
    dx[:, 1:-1], dy[1:-1] = X[:, 2:] - X[:, :-2], X[2:] - X[:-2]
    n = torch.cross(dx, dy, dim=-1)
    n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    Lv = t - X
    r = Lv.norm(dim=-1).clamp(min=1e-3)
    return ((n * Lv).sum(-1) / r).abs().clamp(min=0.05), r


def cos_normal(D, N):
    X = torch.stack([PX[None, :] * D, PY[:, None] * D, D], -1)
    Lv = t - X
    r = Lv.norm(dim=-1).clamp(min=1e-3)
    return ((N * Lv).sum(-1) / r).abs().clamp(min=0.05)


def hf_share(img, mask):
    """energy of the 1-3 px band over total variance, inside mask"""
    x = img.astype(np.float32)
    band = cv2.GaussianBlur(x, (0, 0), 1) - cv2.GaussianBlur(x, (0, 0), 3)
    return float((band[mask] ** 2).mean() / max(x[mask].var(), 1e-12))


print(f"lamp offset {[round(v * 1000, 1) for v in t.tolist()]} mm")
for k in a.frames:
    v = torch.from_numpy(d["held_viewmats"][k]).to(dev)
    D, N, ok = surface(v)
    cd, r = cos_depth(D)
    cn = cos_normal(D, N)
    msk = (ok & (D > 0)).cpu().numpy()
    msk = cv2.erode(msk.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    A, B = cd.cpu().numpy(), cn.cpu().numpy()
    print(f"frame {d['held_ids'][k]}: cos from depth: median {np.median(A[msk]):.3f}, 1-3 px band share {hf_share(A, msk):.3f} | "
          f"cos from normal map: median {np.median(B[msk]):.3f}, band share {hf_share(B, msk):.3f} | "
          f"median |difference| {np.median(np.abs(A - B)[msk]):.3f}")
    fac = A / (r.cpu().numpy() / np.median(r.cpu().numpy()[msk])) ** 2
    if logM is not None:
        Mfull = torch.exp(F.interpolate(logM, size=(CH, CW), mode="bilinear", align_corners=True))[0, 0].cpu().numpy()
        fac = fac * Mfull
    vis = [np.clip(A, 0, 1), np.clip(B, 0, 1), np.clip(fac / np.percentile(fac[msk], 99), 0, 1)]
    cv2.imwrite(os.path.join(a.run, f"diag_cos_{int(d['held_ids'][k]):06d}{'_s%d' % a.smooth_mm if a.smooth_mm else ''}.png"), (np.hstack(vis) * 255).astype(np.uint8))
