"""Shared set-up for the reconstruction steps R1-R5 (scripts/recon/): paths, extrinsics, the fused camera pose
of f5_fuse.py, and the pipe centerline (a smooth curve through the bend) with pipe coordinates (u, theta, r).

World frame = the fused metric frame of f5_fuse.py (vision axes, AprilTag scale), so the new model can be laid
over the earlier renders (compare_views.py) one to one.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "fusion"))
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402
from timebase import load_offsets, stream_times  # noqa: E402
from vision_model import load_cameras, pose_at  # noqa: E402

EX = paths.INPIPE_EXTRACT                   # extracted bag (lidar/extract_bag.py)
VW = paths.INPIPE_VISION                    # vision work folder (vision/run_vision.py)
RES = paths.INPIPE_RESULTS
FU = os.path.join(RES, "fusion")            # inputs: clocks, extrinsics, scale, vision legs (F1-F4)
OUT = os.path.join(RES, "recon")            # small files and figures
BIG = paths.RECON_BIG                       # point clouds, meshes, textures
os.makedirs(OUT, exist_ok=True)
os.makedirs(BIG, exist_ok=True)

off = load_offsets(os.path.join(FU, "time_offsets.json"))
R_cl = np.array(json.load(open(os.path.join(FU, "extrinsic_rotation.json")))["R_cl"])
t_cl = np.array(json.load(open(os.path.join(FU, "extrinsic_translation_final.json")))["t_cl_m"])
lam = json.load(open(os.path.join(FU, "f4_tag_scale.json")))["mm_per_unit_tag"] / 1000
legs = json.load(open(os.path.join(FU, "vision_legs.json")))
ci = json.load(open(os.path.join(EX, "bag_info.json")))["camera_info"]["color"]
K = np.array(ci["k"]).reshape(3, 3)
W, H = ci["width"], ci["height"]
a1, a2 = np.array(legs["legs"][0]["dir"]), np.array(legs["legs"][1]["dir"])
B_units = np.array(legs["bend_point"])


class FusedPoses:
    """Camera pose of f5_fuse.py at any common time t: orientation from SfM, position on the vision centerline
    (two legs joined at the bend) at the encoder arc length, plus the camera's lateral offset from it."""

    def __init__(self):
        self.cam = cam = load_cameras(EX, VW, off)
        B = B_units

        def centerline_coords(P):
            u1, u2 = (P - B) @ a1, (P - B) @ a2
            on2 = u2 > 0
            u = np.where(on2, u2, np.minimum(u1, 0))
            Q = np.where(on2[:, None], B + np.outer(u2, a2), B + np.outer(np.minimum(u1, 0), a1))
            return u, Q

        self._cp = lambda u: np.where((np.atleast_1d(u) > 0)[:, None], B + np.outer(np.atleast_1d(u), a2),
                                      B + np.outer(np.atleast_1d(u), a1))
        u_cam, Q_cam = centerline_coords(cam["C"])
        self.lat_cam = (cam["C"] - Q_cam) * lam
        tE, rE = stream_times(EX, off, "encoders")
        sE = np.array([(float(r["v6"]) - float(r["v7"])) / 2 for r in rE])
        self.tE, self.sE = tE, sE
        self.s0 = np.interp(cam["t"][0], tE, sE)
        self.u0 = u_cam[0] * lam

    def s_enc(self, t):
        return np.interp(t, self.tE, self.sE) - self.s0

    def pose(self, t):
        """world->camera rotation R, camera centre C (m), arc position (m, 0 at the vision bend point)"""
        cam = self.cam
        R, _ = pose_at(cam, t)
        u_m = self.u0 + self.s_enc(t)
        lat = np.array([np.interp(t, cam["t"], self.lat_cam[:, j]) for j in range(3)])
        C = self._cp(u_m / lam)[0] * lam + lat
        return R, C, u_m


def heightfield_mesh(hf, cl, key="r_mesh"):
    """triangle mesh of a height field (R3): one vertex per filled cell centre of the mesh rows, two triangles per
    quad of four filled cells (not across +-180 deg: the slot is there). UV: u along x, theta along y
    (v = (theta + 180) / 360), over the mesh rows. Returns dict(xyz, tri, vuv, rows, uv_range)."""
    a, b = (int(x) for x in hf["mesh_rows"])
    R = hf[key][a:b]
    uc, tc = hf["u"][a:b], hf["th"]
    nu, nt = R.shape
    ok = np.isfinite(R)
    vid = np.full(R.shape, -1, np.int64)
    vid[ok] = np.arange(ok.sum())
    U, TH = np.meshgrid(uc, tc, indexing="ij")
    xyz = cl.point(U[ok], TH[ok], R[ok])[0]
    i, j = np.meshgrid(np.arange(nu - 1), np.arange(nt - 1), indexing="ij")
    q = np.stack([vid[i, j], vid[i + 1, j], vid[i + 1, j + 1], vid[i, j + 1]], axis=-1).reshape(-1, 4)
    q = q[(q >= 0).all(1)]
    tri = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
    u_lo, u_hi = uc[0] - (uc[1] - uc[0]) / 2, uc[-1] + (uc[1] - uc[0]) / 2
    vuv = np.column_stack([(U[ok] - u_lo) / (u_hi - u_lo), (TH[ok] + 180) / 360])
    return {"xyz": xyz, "tri": tri, "vuv": vuv, "rows": (a, b), "uv_range": (u_lo, u_hi)}


class Centerline:
    """Smooth pipe centerline sampled every DS metres: points C, unit tangents T and a rotation-minimising frame
    (N, Bn) with theta = atan2(q.Bn, q.N), shifted so that the bottom slot is at +-180 deg."""

    def __init__(self, path):
        z = np.load(path)
        self.u, self.C, self.T, self.N, self.Bn = z["u"], z["C"], z["T"], z["N"], z["Bn"]
        self.th0 = float(z["th0"])
        self.ds = float(self.u[1] - self.u[0])

    @staticmethod
    def build(u, P, smooth_pts, up, ds=0.001):
        """centerline through the points P(u) (local axis points), smoothed over smooth_pts samples, resampled every ds"""
        uu = np.arange(u.min(), u.max() + ds / 2, ds)
        C = np.column_stack([np.interp(uu, u, P[:, j]) for j in range(3)])
        k = int(smooth_pts / ds) | 1
        ker = np.ones(k) / k
        C = np.column_stack([np.convolve(np.pad(C[:, j], k // 2, mode="edge"), ker, mode="valid") for j in range(3)])
        # resample at equal arc length (the padded ends of the smoothed curve are bunched up), so that
        # coords() can find a sample by index
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))])
        a_new = np.arange(0.0, arc[-1], ds)
        C = np.column_stack([np.interp(a_new, arc, C[:, j]) for j in range(3)])
        uu = uu[0] + a_new
        T = np.gradient(C, axis=0)
        T /= np.linalg.norm(T, axis=1, keepdims=True)
        N = np.zeros_like(C)
        n = up - (up @ T[0]) * T[0]
        N[0] = n / np.linalg.norm(n)
        for i in range(1, len(C)):                        # parallel transport (double reflection would be finer)
            n = N[i - 1] - (N[i - 1] @ T[i]) * T[i]
            N[i] = n / np.linalg.norm(n)
        Bn = np.cross(T, N)
        return uu, C, T, N, Bn

    def coords(self, X, u_init=None, iters=4):
        """(u, theta deg, r) of world points X: nearest centerline sample by fixed-point iteration from u_init"""
        X = np.asarray(X, float)
        if u_init is None:
            # coarse start: nearest of every 10th sample (chunked)
            sub = self.C[::10]
            u_init = np.empty(len(X))
            for a in range(0, len(X), 200000):
                d2 = ((X[a:a + 200000, None, :] - sub[None]) ** 2).sum(2)
                u_init[a:a + 200000] = self.u[::10][np.argmin(d2, axis=1)]
        u = np.asarray(u_init, float).copy()
        for _ in range(iters):
            i = np.clip(np.round((u - self.u[0]) / self.ds).astype(int), 0, len(self.u) - 1)
            q = X - self.C[i]
            u = self.u[i] + np.einsum("ij,ij->i", q, self.T[i])
        i = np.clip(np.round((u - self.u[0]) / self.ds).astype(int), 0, len(self.u) - 1)
        q = X - self.C[i]
        a, b = np.einsum("ij,ij->i", q, self.N[i]), np.einsum("ij,ij->i", q, self.Bn[i])
        th = (np.degrees(np.arctan2(b, a)) - self.th0 + 540) % 360 - 180
        return u, th, np.hypot(a, b)

    def point(self, u, th, r):
        """world position of pipe coordinates (inverse of coords)"""
        u = np.asarray(u, float)
        i = np.clip((u - self.u[0]) / self.ds, 0, len(self.u) - 1.000001)
        i0 = np.floor(i).astype(int)
        w = (i - i0)[..., None]
        C = (1 - w) * self.C[i0] + w * self.C[i0 + 1]
        N = (1 - w) * self.N[i0] + w * self.N[i0 + 1]
        Bn = (1 - w) * self.Bn[i0] + w * self.Bn[i0 + 1]
        T = (1 - w) * self.T[i0] + w * self.T[i0 + 1]
        # the frame was transported, so the in-plane offset is along N, Bn of the interpolated sample
        t = np.radians(np.asarray(th, float) + self.th0)
        off_ = np.cos(t)[..., None] * N + np.sin(t)[..., None] * Bn
        return C + np.asarray(r, float)[..., None] * off_, off_, T
