"""Focal length (and radial distortion) of a camera from a stretch of video where it only rotates (pan/tilt head,
robot standing still - e.g. the operator looking around a manhole): two views of a rotating camera obey
x2 ~ K R K^-1 x1, independent of the scene, so f is observable (it is not under forward motion in a pipe).

usage: python -I focal_check_rotation.py VIDEO --first A --last B --pp CX CY [--crop X0 Y0 X1 Y1] [--stretch-x S]
                                         [--step 2] [--gaps 4 8 12] [--compare F ...]
       VIDEO, --crop, --stretch-x  as given to run_pipeline.py; --pp principal point in those pixels
       Choose A..B where the robot does not move (on-screen distance constant): translation biases the result.
       --compare  report the residual at these focal lengths (e.g. the calibrated one)

SIFT matches and a RANSAC homography per frame pair; then on a grid of (f, kappa) the best rotation per pair
(Kabsch on bearing vectors) and the median reprojection residual; f, kappa = argmin. kappa is the pixel-unit radial
distortion, r_d = r_u (1 + kappa r_u^2) (OpenMVG's k1 = kappa f^2). Caveat: a pan/tilt pivot a few cm off the
optical centre adds parallax at close range; expect about +-10 %. Pipe 1 example: docs/03, section 4.
"""
import argparse

import cv2
import numpy as np

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("video")
ap.add_argument("--first", type=int, required=True), ap.add_argument("--last", type=int, required=True)
ap.add_argument("--pp", nargs=2, type=float, required=True)
ap.add_argument("--crop", nargs=4, type=int), ap.add_argument("--stretch-x", type=float, default=1.0)
ap.add_argument("--step", type=int, default=2), ap.add_argument("--gaps", nargs="+", type=int, default=[4, 8, 12])
ap.add_argument("--compare", nargs="+", type=float, default=[])
a = ap.parse_args()
CX, CY = a.pp

cap = cv2.VideoCapture(a.video)
sift = cv2.SIFT_create(nfeatures=3000)
clahe = cv2.createCLAHE(2.0, (8, 8))
feats = {}
cap.set(cv2.CAP_PROP_POS_FRAMES, a.first)
for i in range(a.first, a.last + 1):
    ok, fr = cap.read()
    if not ok:
        break
    if (i - a.first) % a.step:
        continue
    if a.crop:
        fr = fr[a.crop[1]:a.crop[3], a.crop[0]:a.crop[2]]
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    if a.stretch_x != 1.0:
        g = cv2.resize(g, (int(round(g.shape[1] * a.stretch_x)), g.shape[0]), interpolation=cv2.INTER_CUBIC)
    if g.mean() < 25:                      # too dark (lamp pointing into the void)
        continue
    kp, de = sift.detectAndCompute(clahe.apply(g), None)
    if de is not None and len(kp) >= 50:
        feats[i] = (np.array([k.pt for k in kp], np.float64), de)

bf = cv2.BFMatcher(cv2.NORM_L2)
pairs = []
for i in sorted(feats):
    for gap in a.gaps:
        j = i + gap
        if j not in feats:
            continue
        (p1, d1), (p2, d2) = feats[i], feats[j]
        good = [m for m, n in (x for x in bf.knnMatch(d1, d2, k=2) if len(x) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < 40:
            continue
        x1, x2 = p1[[m.queryIdx for m in good]], p2[[m.trainIdx for m in good]]
        Hm, inl = cv2.findHomography(x1, x2, cv2.RANSAC, 3.0)
        if Hm is None or inl.sum() < 40:
            continue
        inl = inl.ravel().astype(bool)
        if np.median(np.linalg.norm(x1[inl] - x2[inl], axis=1)) >= 12:     # enough rotation to tell f
            pairs.append((x1[inl], x2[inl]))
print(f"{len(feats)} usable frames in {a.first}..{a.last}; {len(pairs)} pairs with >= 40 homography inliers and "
      f">= 12 px median motion")
if not pairs:
    raise SystemExit("no usable pairs: choose a stretch where the camera pans or tilts")


def undist(p, kappa):
    d = p - [CX, CY]
    rd = np.linalg.norm(d, axis=1)
    ru = rd.copy()
    for _ in range(30):
        ru = rd / (1 + kappa * ru * ru)
    return d * (ru / np.maximum(rd, 1e-9))[:, None]


def residuals(f, kappa):
    out = []
    for x1, x2 in pairs:
        u1, u2 = undist(x1, kappa), undist(x2, kappa)
        b1 = np.c_[u1 / f, np.ones(len(u1))]
        b2 = np.c_[u2 / f, np.ones(len(u2))]
        b1 /= np.linalg.norm(b1, axis=1, keepdims=True)
        b2 /= np.linalg.norm(b2, axis=1, keepdims=True)
        U, _, Vt = np.linalg.svd(b2.T @ b1)
        R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
        q = b1 @ R.T
        out.append(np.median(np.linalg.norm(f * q[:, :2] / q[:, 2:3] - u2, axis=1)))
    return np.array(out)


fs = np.arange(250, 901, 10.0)
ks = np.arange(-3.0e-6, 0.51e-6, 2.5e-7)
G = np.array([[np.median(residuals(f, k)) for k in ks] for f in fs])
i, j = np.unravel_index(np.argmin(G), G.shape)
ff = np.arange(fs[i] - 20, fs[i] + 20.1, 2.0)
kk = np.arange(ks[j] - 3e-7, ks[j] + 3.01e-7, 5e-8)
F = np.array([[np.median(residuals(f, k)) for k in kk] for f in ff])
c, d = np.unravel_index(np.argmin(F), F.shape)
f_best, k_best = ff[c], kk[d]
print(f"best: f {f_best:.0f} px, kappa {k_best:.2e} /px^2 (k1 {k_best * f_best ** 2:.3f}), median residual {F[c, d]:.2f} px")
per = [fs[int(np.argmin([np.median(residuals(f, k_best)[[n]]) for f in fs]))] for n in range(len(pairs))]
print(f"per-pair best f at that kappa: median {np.median(per):.0f}, p25-p75 {np.percentile(per, 25):.0f}-{np.percentile(per, 75):.0f} px")
print("residual vs f at that kappa: " + " ".join(f"{f:.0f}:{r:.2f}" for f, r in zip(fs[::3], G[::3, j])))
for f in a.compare:
    print(f"f {f:.1f} px: residual {np.median(residuals(f, k_best)):.2f} px at the best kappa")
