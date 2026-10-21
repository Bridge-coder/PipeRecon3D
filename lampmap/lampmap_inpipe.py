"""Lamp-map model (lampmap.py) for the LiDAR + vision model of InPipe-SLAM 172011: the R3 height-field mesh with the
R4 per-pixel texture, drawn from the R2b-corrected fused pose exactly as scripts/recon/r5_compare.py draws it.

usage: python -I lampmap_inpipe.py OUT_DIR [--n 44] [--train 40] [--width 640] [--save 4]

Model frames (lamp map) = the SfM colour frames the texture was made from; held-out frames = the colour frame
halfway to the next SfM frame (+5); next-frame reference = the frame after it (+6). Frames as recorded (RealSense,
no CLAHE), scaled to --width (1280 x 720 -> 640 x 360). Paths from scripts/paths.py (RECON_BIG, INPIPE_EXTRACT, ...).
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts", "recon"))
import lampmap as lm  # noqa: E402
from recon_common import BIG, EX, K, W, H, Centerline, FusedPoses, heightfield_mesh, off, stream_times  # noqa: E402
from vision_model import _expmap  # noqa: E402

rv = lm.rv
U_MAX = 1.005                                   # as r5_compare.py: the mesh ends at the pipe exit


def load_mesh():
    cl = Centerline(os.path.join(BIG, "centerline.npz"))
    m = heightfield_mesh(np.load(os.path.join(BIG, "heightfield.npz")), cl)
    xyz, tri, vuv = m["xyz"], m["tri"], m["vuv"]
    u_lo, u_hi = m["uv_range"]
    tri = tri[(u_lo + vuv[tri, 0] * (u_hi - u_lo) <= U_MAX).all(1)]
    tex = cv2.imdecode(np.fromfile(os.path.join(BIG, "texture_refined.png"), np.uint8), cv2.IMREAD_COLOR)
    return dict(xyz=xyz, tri=tri, uv=vuv[tri], tex=[tex], blank=np.zeros(len(tri), bool),
                centroid=xyz[tri].mean(axis=1))


class PinholeRenderer:
    """the mesh through the colour camera (no distortion), pose = (R world->camera OpenCV axes, centre C)"""

    def __init__(self, mesh, w):
        self.m, self.w, self.h = mesh, w, int(round(H * w / W))
        s = w / W
        self.f, self.cx, self.cy = K[0, 0] * s, K[0, 2] * s, K[1, 2] * s
        self.cw = 2 * int(np.ceil(max(self.cx, self.w - self.cx)))
        self.ch = 2 * int(np.ceil(max(self.cy, self.h - self.cy)))
        self.x0, self.y0 = int(round(self.cw / 2 - self.cx)), int(round(self.ch / 2 - self.cy))

    def __call__(self, pose):
        R, C = pose
        cam = dict(eye=np.asarray(C, float), R=np.array([R[0], -R[1], R[2]]), size=(self.cw * rv.SS, self.ch * rv.SS),
                   focal=self.f * rv.SS)
        sel = np.flatnonzero((self.m["centroid"] - C) @ R[2] > 0.0)
        rgb, cov = lm.textured(self.m, cam, sel)
        sl = (slice(self.y0, self.y0 + self.h), slice(self.x0, self.x0 + self.w))
        return lm.finish(rgb[sl], cov[sl])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out"), ap.add_argument("--n", type=int, default=44), ap.add_argument("--train", type=int, default=40)
    ap.add_argument("--width", type=int, default=640), ap.add_argument("--save", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    render = PinholeRenderer(load_mesh(), a.width)
    fp = FusedPoses()
    sc = np.load(os.path.join(BIG, "scan_corrections.npz"))
    tsc = sc["t"].astype(float)

    def pose_at(t):
        """fused pose with the R2b correction (as the texture was made; r5_compare.poses_at 'now')"""
        R, C, _ = fp.pose(t)
        f = lambda A: np.array([np.interp(float(t), tsc, A[:, j]) for j in range(3)])  # noqa: E731
        d, w, O = f(sc["d"]), f(sc["w"]), f(sc["O"])
        return R @ _expmap(w).T, C + d + np.cross(w, C - O)

    color_t, color_rows = stream_times(EX, off, "color")
    t_of = {int(r["idx"]): t for r, t in zip(color_rows, color_t)}

    def read(idx):
        img = cv2.imread(os.path.join(EX, "images", "color", f"{idx:06d}.png"))
        return cv2.resize(img, (render.w, render.h), interpolation=cv2.INTER_AREA)

    sfm = [int(i) for i in fp.cam["idx"]]
    print(f"{len(sfm)} SfM colour frames, render {render.w}x{render.h} [{time.time() - t0:.0f} s]", flush=True)
    train = [(pose_at(t_of[sfm[j]]), sfm[j]) for j in np.linspace(0, len(sfm) - 1, a.train).round().astype(int)]
    cand = [i + 5 for i in sfm[:-1] if i + 5 not in sfm and i + 5 in t_of and i + 6 in t_of]
    heldout = [(pose_at(t_of[i]), i, i + 1, dict(u=float(fp.pose(t_of[i])[2])))
               for i in (cand[j] for j in np.linspace(0, len(cand) - 1, a.n).round().astype(int))]
    lm.evaluate(render, read, train, heldout, a.out,
                dict(model="InPipe 172011 R3 height field + R4 texture (texture_refined.png)", width=a.width), t0, a.save)


if __name__ == "__main__":
    main()
