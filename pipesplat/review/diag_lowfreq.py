"""docs/13: where is a model's low-frequency (shading) error? Low-pass (sigma 6 px, as imgmetrics.band_split) of
render - frame (grey; the dumped renders are already brightness-matched as scored), averaged over the held-out and over
the block frames: the signed mean (red = render too bright, blue = too dark: a pattern the model gets wrong the same
way in every frame) and the RMS; then the block frames' low-frequency error along the pipe (it grows toward the middle
of each 10 cm block, the point farthest from any training camera, when shading does not carry to new positions).
Renders from the frame dumps of dump_frames.py; mask = B's coverage.

usage: python -I pipesplat/review/diag_lowfreq.py DATA RUN OUT.jpg
"""
import csv
import glob
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

ds, run, out = sys.argv[1:4]
V = os.path.join(paths.GS_ROOT, "review", "video", ds, run)
G = os.path.join(paths.GS_ROOT, "g0", ds, run)
rd = lambda p: cv2.imread(p)[:, :, ::-1].astype(np.float32)  # noqa: E731
LIM = 12.0                                              # grey levels at full colour


def lowpass(z, m, s=6.0):
    wgt = m.astype(np.float32)
    return cv2.GaussianBlur(z * wgt, (0, 0), s) / (cv2.GaussianBlur(wgt, (0, 0), s) + 1e-6)


tiles = []
for tag in ("held", "block"):
    files = sorted(glob.glob(os.path.join(V, f"{tag}_*_G.jpg")))
    S = Q = N = None
    for f in files:
        g, r = im.grey(rd(f)), im.grey(rd(f.replace("_G.jpg", "_real.jpg")))
        m = im.erode(rd(f.replace("_G.jpg", "_B.jpg")).max(2) > 8, 8)
        d = lowpass(g - r, m) * m
        if S is None:
            S, Q, N = np.zeros_like(d), np.zeros_like(d), np.zeros_like(d)
        S += d
        Q += d * d
        N += m
    mean, rms = S / np.maximum(N, 1), np.sqrt(Q / np.maximum(N, 1))
    for img, signed in ((mean, True), (rms, False)):
        v = np.clip(img / LIM, -1, 1) if signed else np.clip(img / LIM, 0, 1)
        rgb = np.stack([np.clip(v, 0, 1), np.zeros_like(v), np.clip(-v, 0, 1)], 2) if signed else np.stack([v] * 3, 2)
        t = (rgb * 255).astype(np.uint8)[:, :, ::-1].copy()
        cv2.putText(t, f"{tag} {'mean' if signed else 'rms'}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles.append(cv2.resize(t, (360, int(360 * t.shape[0] / t.shape[1]))))
    print(f"{ds} {run} {tag:5s}: {len(files)} frames, mean |signed| {np.abs(mean[N > 0]).mean():.1f}, "
          f"rms {rms[N > 0].mean():.1f} grey levels")
cv2.imwrite(out, np.hstack(tiles))
pf = [r for r in csv.DictReader(open(os.path.join(G, "per_frame_r2.csv"), encoding="utf-8")) if r["tag"] == "block"]
blk = sorted((float(r["pos"]), float(r["G_mse_low"]), float(r["G_psnr"])) for r in pf)
print("block frames along the pipe (pos m: mse_low / PSNR):", "  ".join(f"{p:.3f}: {l:.0f} / {q:.1f}" for p, l, q in blk))
