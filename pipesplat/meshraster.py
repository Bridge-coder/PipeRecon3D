"""Exact triangle rasterisation of a mesh in the RAW (radially distorted) frame on the GPU, for scoring a textured mesh
the way an ordinary viewer draws it (docs/10 X8). train_tex.py renders the measured surface as one gsplat disc per
triangle (0.6 x the triangle size); on the rough OpenMVS mesh the front-most of the overlapping discs lies ~1 mm in
front of the triangles at grazing angles, which moves a pixel's surface point ~2 mm along the pipe - harmless for a
texture learnt through that same renderer (TX), wrong for a texture placed on the triangles (a baked texture, an OBJ).

Vertices are projected with x_d = x_u (1 + k1 r_u^2) (triangles are a few pixels at most, so straight edges between
the projected corners are exact to well under a pixel); each pixel takes the nearest triangle whose projection
contains the pixel centre, and its depth is the intersection of the pixel's own undistorted ray with that triangle's
plane. Triangles with a corner behind the camera or beyond the fold of the k1 model are dropped.
"""
import torch


class MeshRaster:
    def __init__(self, xyz, tri, K, k1, w, h, rays_x, rays_y, dev="cuda"):
        """xyz [n, 3], tri [m, 3] (numpy or torch), K 3x3, rays_x / rays_y [h, w]: the undistorted ray x/z, y/z of
        every raw pixel (pixel centre at its integer index)"""
        self.V = torch.as_tensor(xyz, dtype=torch.float64, device=dev)
        self.T = torch.as_tensor(tri, dtype=torch.int64, device=dev)
        self.fx, self.fy, self.cx, self.cy = float(K[0][0]), float(K[1][1]), float(K[0][2]), float(K[1][2])
        self.k1, self.w, self.h, self.dev = float(k1), int(w), int(h), dev
        self.r2max = 0.95 ** 2 / (-3 * self.k1) if self.k1 < 0 else float("inf")
        self.RX = torch.as_tensor(rays_x, dtype=torch.float64, device=dev)
        self.RY = torch.as_tensor(rays_y, dtype=torch.float64, device=dev)
        a, b, c = self.V[self.T[:, 0]], self.V[self.T[:, 1]], self.V[self.T[:, 2]]
        n = torch.cross(b - a, c - a, dim=1)
        self.N = (n / n.norm(dim=1, keepdim=True).clamp(min=1e-30)).float()     # world face normals

    @torch.no_grad()
    def __call__(self, view, chunk=400000, maxc=4096):
        """depth [h, w] (camera z, 0 = no surface), face index [h, w] (-1 = none) for a world->camera matrix"""
        R, t = view[:3, :3].double(), view[:3, 3].double()
        Vc = self.V @ R.T + t
        z = Vc[:, 2]
        zs = z.clamp(min=1e-9)
        xn, yn = Vc[:, 0] / zs, Vc[:, 1] / zs
        r2 = xn * xn + yn * yn
        g = 1 + self.k1 * r2
        px, py = self.cx + self.fx * xn * g, self.cy + self.fy * yn * g
        vok = (z > 0.01) & (r2 < self.r2max)
        T = self.T[vok[self.T].all(1)]
        P = torch.stack([px, py], 1)[T]                                    # [m, 3, 2]
        lo, hi = torch.ceil(P.amin(1)).long(), torch.floor(P.amax(1)).long()
        lo = torch.maximum(lo, torch.zeros_like(lo))
        hi = torch.minimum(hi, torch.tensor([self.w - 1, self.h - 1], device=self.dev))
        nc, nr = (hi[:, 0] - lo[:, 0] + 1).clamp(min=0), (hi[:, 1] - lo[:, 1] + 1).clamp(min=0)
        cnt = nc * nr
        keep = (cnt > 0) & (cnt <= maxc)
        T, P, lo, nc, cnt = T[keep], P[keep], lo[keep], nc[keep], cnt[keep]
        fid = torch.nonzero(vok[self.T].all(1))[:, 0][keep]                # original face index
        zbuf = torch.full((self.h * self.w,), float("inf"), device=self.dev, dtype=torch.float64)
        face = torch.full((self.h * self.w,), -1, device=self.dev, dtype=torch.int64)
        cand = []
        for s in range(0, len(T), chunk):
            c_ = cnt[s:s + chunk]
            tot = int(c_.sum())
            if tot == 0:
                continue
            fi = torch.repeat_interleave(torch.arange(len(c_), device=self.dev), c_)
            loc = torch.arange(tot, device=self.dev) - (torch.cumsum(c_, 0) - c_)[fi]
            ncf = nc[s:s + chunk][fi]
            ii = lo[s:s + chunk][fi, 0] + loc % ncf
            jj = lo[s:s + chunk][fi, 1] + torch.div(loc, ncf, rounding_mode="floor")
            p = P[s:s + chunk][fi]
            v0, v1 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
            qx, qy = ii.double() - p[:, 0, 0], jj.double() - p[:, 0, 1]
            den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
            ok = den.abs() > 1e-12
            den = torch.where(ok, den, torch.ones_like(den))
            l1 = (qx * v1[:, 1] - v1[:, 0] * qy) / den
            l2 = (v0[:, 0] * qy - qx * v0[:, 1]) / den
            ok &= (l1 >= -1e-7) & (l2 >= -1e-7) & (l1 + l2 <= 1 + 1e-7)
            fi, ii, jj = fi[ok], ii[ok], jj[ok]
            tri = T[s:s + chunk][fi]
            a = Vc[tri[:, 0]]
            nrm = torch.cross(Vc[tri[:, 1]] - a, Vc[tri[:, 2]] - a, dim=1)
            rx, ry = self.RX[jj, ii], self.RY[jj, ii]
            dn = nrm[:, 0] * rx + nrm[:, 1] * ry + nrm[:, 2]
            zz = (nrm * a).sum(1) / torch.where(dn.abs() > 1e-15, dn, torch.full_like(dn, 1e-15))
            good = (zz > 0.01) & torch.isfinite(zz)
            lin = (jj * self.w + ii)[good]
            zz, f_ = zz[good], fid[s:s + chunk][fi[good]]
            zbuf.scatter_reduce_(0, lin, zz, reduce="amin")
            cand.append((lin, zz, f_))
        for lin, zz, f_ in cand:
            win = zz <= zbuf[lin]
            face[lin[win]] = f_[win]
        D = torch.where(torch.isfinite(zbuf), zbuf, torch.zeros_like(zbuf)).view(self.h, self.w).float()
        return D, face.view(self.h, self.w)
