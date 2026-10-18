"""Check a camera calibration on the inspection video itself: K = f * R from wall features moving outward.

usage: python -I focal_check_rings.py VIDEO COUNTER_CSV --first A --last B --centre CX CY
                                      [--crop X0 Y0 X1 Y1] [--stretch-x S] [--rmin 50] [--rmax 300] [--kappa=K]
       VIDEO, --crop, --stretch-x  as given to run_pipeline.py (frames in the pixels the intrinsics refer to)
       COUNTER_CSV  ground-truth travel per source frame (source_frame, counter_m), e.g. WORK/counter.csv
       --centre     the pipe centre in the image (vanishing point), in those pixels
       --kappa      also report K for this fixed pixel-unit distortion, e.g. --kappa=-1.033e-6 (the calibration's k1 / f^2;
                    write it with "=": a negative number would be read as an option)

A camera driving along a round pipe sees a wall point at along-pipe position Z, tether distance s, at the undistorted
image radius r_u = f R / (Z - s) = K / (Z - s); the lens maps it to r_d = r_u (1 + kappa r_u^2). For every frame the
mean radial edge profile over a sector is mapped back to Z; with the right (K, kappa) the same wall features (joints,
stains) from different frames land at the same Z, so the agreement between frames peaks there.
The left and right walls are measured separately: a camera off the pipe axis by e sees them at f (R +- e_x) / z, so
their mean is f R to first order (the upper wall gives f (R + e_y) for a camera below the axis; the invert is often
under water). Only the product f R is observable this way: with the calibrated f it gives the bore R, or with a known
bore the focal length. See focal_check_rotation.py for f on its own. Pipe 1 example: docs/03, section 4.
"""
import argparse
import csv

import cv2
import numpy as np

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("video"), ap.add_argument("counter")
ap.add_argument("--first", type=int, required=True), ap.add_argument("--last", type=int, required=True)
ap.add_argument("--centre", nargs=2, type=float, required=True)
ap.add_argument("--crop", nargs=4, type=int), ap.add_argument("--stretch-x", type=float, default=1.0)
ap.add_argument("--rmin", type=float, default=50.0), ap.add_argument("--rmax", type=float, default=300.0)
ap.add_argument("--kappa", type=float, action="append", default=[])
a = ap.parse_args()
cx, cy = a.centre

# ground truth: readings come in steps (e.g. 0.1 m); put each new reading at the frame where it appears, interpolate
gt = {int(r["source_frame"]): float(r["counter_m"]) for r in csv.DictReader(open(a.counter, encoding="utf-8"))}
fr_all = np.array(sorted(gt))
v_all = np.array([gt[i] for i in fr_all])
chg = np.r_[0, np.where(np.diff(v_all) != 0)[0] + 1]

cap = cv2.VideoCapture(a.video)
cap.set(cv2.CAP_PROP_POS_FRAMES, a.first)
frames = []
for i in range(a.first, a.last + 1):
    ok, fr = cap.read()
    if not ok:
        break
    if a.crop:
        fr = fr[a.crop[1]:a.crop[3], a.crop[0]:a.crop[2]]
    if a.stretch_x != 1.0:
        fr = cv2.resize(fr, (int(round(fr.shape[1] * a.stretch_x)), fr.shape[0]), interpolation=cv2.INTER_CUBIC)
    frames.append(cv2.GaussianBlur(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.2))
idx_frames = np.arange(a.first, a.first + len(frames))
s_frames = np.interp(idx_frames, fr_all[chg], v_all[chg])
H, W = frames[0].shape
yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
dx, dy = xx - cx, yy - cy
r_pix = np.hypot(dx, dy)
ang = np.degrees(np.arctan2(-dy, dx))
edges = np.arange(12, 420, 1.0)
rbin = np.digitize(r_pix, edges) - 1
r_mid = 0.5 * (edges[:-1] + edges[1:])
sectors = {"left": np.abs(ang) > 155, "right": np.abs(ang) < 25, "upper": (ang > 15) & (ang < 165)}
profiles = {k: [] for k in sectors}
for g in frames:
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    gr = np.abs((gx * dx + gy * dy) / (r_pix + 1e-9))
    for name, m in sectors.items():
        ok = m & (rbin >= 0) & (rbin < len(r_mid))
        cnt = np.bincount(rbin[ok], minlength=len(r_mid))
        profiles[name].append(np.bincount(rbin[ok], weights=gr[ok], minlength=len(r_mid)) / np.maximum(cnt, 1))

BIN = 0.01
kern = np.exp(-0.5 * (np.arange(-4, 5) / 2.0) ** 2)
kern /= kern.sum()


def prepare(prof):
    keep = (r_mid >= a.rmin) & (r_mid <= a.rmax)
    h = np.array(prof, np.float64)[:, keep]
    h = h - cv2.GaussianBlur(h, (0, 0), sigmaX=6, sigmaY=0.01)
    h /= np.linalg.norm(h, axis=1, keepdims=True) + 1e-12
    sel, last = [], -1e9                                      # one frame per 2 cm of travel (pauses repeat frames)
    for i, si in enumerate(s_frames):
        if si - last >= 0.02:
            sel.append(i)
            last = si
    return r_mid[keep], h[sel], s_frames[sel]


def agreement(r, h, s, K, kappa):
    """sum over Z of (sum_t a_t)^2 - sum_t a_t^2: agreement between different frames only"""
    ru = r.copy()
    for _ in range(30):
        ru = r / (1 + kappa * ru * ru)
    Zbins = np.arange(-1.0, s.max() + 12.0, BIN)
    k = np.clip(((s[:, None] + K / ru[None, :] - Zbins[0]) / BIN).astype(int), 0, len(Zbins) - 1)
    A = np.zeros(len(Zbins))
    self_sum = 0.0
    for t in range(len(s)):
        at = np.convolve(np.bincount(k[t], weights=h[t], minlength=len(Zbins)), kern, mode="same")
        A += at
        self_sum += float(at @ at)
    return (float(A @ A) - self_sum) / len(s)


print(f"frames {a.first}..{idx_frames[-1]}, travel {s_frames[0]:.2f}..{s_frames[-1]:.2f} m, radii {a.rmin:.0f}..{a.rmax:.0f} px, "
      f"centre ({cx:.0f}, {cy:.0f})")
Ks = np.arange(40.0, 400.1, 2.0)
kappas = np.arange(-3.0e-6, 1.01e-6, 1e-7)
best = {}
for name in sectors:
    r, h, s = prepare(profiles[name])
    # K first with kappa = 0 (coarse), then (K, kappa) around it
    k0 = Ks[int(np.argmax([agreement(r, h, s, K, 0.0) for K in Ks]))]
    Kf = np.arange(k0 - 16, k0 + 16.1, 1.0)
    C = np.array([[agreement(r, h, s, K, kp) for kp in kappas] for K in Kf])
    i, j = np.unravel_index(np.argmax(C), C.shape)
    fixed = {kp: Kf[int(np.argmax([agreement(r, h, s, K, kp) for K in Kf]))] for kp in [0.0] + a.kappa}
    best[name] = (Kf[i], kappas[j], fixed)
    print(f"{name:5s} wall: K = f*R {Kf[i]:.0f} px*m with fitted kappa {kappas[j]:.2e} /px^2; "
          + "; ".join(f"with kappa {kp:.2e}: K {K:.0f}" for kp, K in fixed.items()))
K_lr = 0.5 * (best["left"][0] + best["right"][0])
print(f"left/right mean K = f*R = {K_lr:.0f} px*m (camera offset cancels to first order); "
      f"upper - left/right = {best['upper'][0] - K_lr:+.0f} px*m = f * (vertical camera offset)")
print("bore for a few focal lengths: " + ", ".join(f"f {f:.0f} px -> D {2000 * K_lr / f:.0f} mm" for f in (380, 450, 490, 540, 600)))
