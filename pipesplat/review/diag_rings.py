"""docs/13: where do the models lose detail? Per ring of image radius (centre = far pipe, edge = near wall), the fine and
mid band correlation of the video frame with its next frame aligned onto it (the ceiling the video's own noise allows,
as imgmetrics.ruler2, split by ring) and each model's fidelity there = corr(render, frame) / sqrt(ceiling).
Renders from the frame dumps of dump_frames.py (GS_ROOT/review/video), frames and next frames from scene.npz.

usage: python -I pipesplat/review/diag_rings.py DATA RUN [RUN ...]     e.g. sewer_pipe1 run_vhf run_3dgs
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "lampmap"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import imgmetrics as im  # noqa: E402
import paths  # noqa: E402

ds, runs = sys.argv[1], sys.argv[2:]
G = os.path.join(paths.GS_ROOT, "g0", ds)
V = os.path.join(paths.GS_ROOT, "review", "video", ds)
RINGS = np.linspace(0, 1, 6)
BANDS = (("fine", 0.7, 1.4), ("mid", 1.4, 2.8))
rd = lambda p: cv2.imread(p)[:, :, ::-1].astype(np.float32)  # noqa: E731


def corr_rings(a_img, b_img, m, ring, s1, s2):
    a, b = im._band(im.grey(a_img), s1, s2), im._band(im.grey(b_img), s1, s2)
    out = []
    for k in range(5):
        sel = m & (ring == k)
        x, y = a[sel].astype(np.float64), b[sel].astype(np.float64)
        x, y = (x - x.mean(), y - y.mean()) if sel.any() else (x, y)
        out.append(np.array([x @ y, x @ x, y @ y]))
    return out


sc = np.load(os.path.join(G, "scene.npz"))
c = lambda s: s[0] / np.sqrt(max(s[1] * s[2], 1e-12))  # noqa: E731
print(f"{ds}: per ring, centre (far) -> edge (near wall)")
for tag in ("held", "block"):
    imgs = sc["held_images"] if tag == "held" else sc["train_images"][sc["block_idx"]]
    nxt, msk, ids = sc[f"{tag}_next"], sc[f"{tag}_mask"], sc[f"{tag}_ids"]
    h, w = msk.shape[1:]
    yy, xx = np.mgrid[0:h, 0:w]
    ring = np.digitize(np.hypot(xx - w / 2, yy - h / 2) / np.hypot(w / 2, h / 2), RINGS[1:-1])
    share = [float((ring == k).mean()) for k in range(5)]
    acc = {}
    for i in range(len(ids)):
        real = imgs[i].astype(np.float32)
        nal = im.align_to(nxt[i].astype(np.float32), real)
        m = im.erode(msk[i].astype(bool), 8)
        for bname, s1, s2 in BANDS:
            for k, s in enumerate(corr_rings(nal, real, m, ring, s1, s2)):
                acc.setdefault(("ceil", bname, k), np.zeros(3)).__iadd__(s)
            for run in runs:
                p = os.path.join(V, run, f"{tag}_{int(ids[i]):06d}_G.jpg")
                if os.path.exists(p):
                    for k, s in enumerate(corr_rings(rd(p), real, m, ring, s1, s2)):
                        acc.setdefault((run, bname, k), np.zeros(3)).__iadd__(s)
    print(f" {tag}: pixel share per ring " + " ".join(f"{v:.2f}" for v in share))
    for bname, *_ in BANDS:
        ceil = [c(acc[("ceil", bname, k)]) for k in range(5)]
        line = f"  {bname:4s} ceiling " + " ".join(f"{v:.2f}" for v in ceil)
        for run in runs:
            if (run, bname, 0) in acc:
                line += f" | fid {run} " + " ".join(f"{c(acc[(run, bname, k)]) / np.sqrt(max(ceil[k], 1e-3)):.2f}" for k in range(5))
        print(line)
