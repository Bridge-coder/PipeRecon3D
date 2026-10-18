"""Evaluate the vision-only SfM of an InPipe-SLAM sequence against wheel encoders and the AprilTag references.

Vision alone has no scale. As in the two papers (one known dimension), one global scale is set from the
nominal pipe diameter (--diameter, default 200 mm: the pipe of the dataset's simulation assets is
"pipe_200mm"); no LiDAR value is used, so this stays a vision-only baseline.

  1. sfm_data.bin -> json (OpenMVG converter, paths.py)
  2. camera centres in frame order; pipe axis = principal direction of the camera path
  3. sparse points within RMAX model units of that axis -> cylinder fit -> pipe radius in model units,
     globally and per stretch of SEG cameras (shows scale drift along the run)
  4. scale = nominal radius / model radius; travel along the axis of every frame compared with the
     wheel encoders ((v6 - v7)/2) and with the two tag tracks (tag side 100 mm)

usage: python -I eval_sfm.py EXTRACT_DIR WORK_DIR OUT_DIR [--diameter MM]
"""
import csv
import json
import os
import subprocess
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipe_geometry import fit_cylinder  # noqa: E402

args = sys.argv[1:]
diam = 200.0
if "--diameter" in args:
    i = args.index("--diameter")
    diam = float(args[i + 1])
    del args[i:i + 2]
EX, WORK, OUT = args
os.makedirs(OUT, exist_ok=True)
from paths import CONVERTER as CONV  # noqa: E402
recon = os.path.join(WORK, "sfm", "recon")
js = os.path.join(recon, "sfm_data_eval.json")
subprocess.run([CONV, "-i", os.path.join(recon, "sfm_data.bin"), "-o", js, "-V", "-I", "-E", "-S"], check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
d = json.load(open(js, encoding="utf-8"))

frames = {r["file"]: r for r in csv.DictReader(open(os.path.join(WORK, "frames.csv"), encoding="utf-8"))}
view = {v["value"]["ptr_wrapper"]["data"]["id_view"]: v["value"]["ptr_wrapper"]["data"] for v in d["views"]}
pose = {e["key"]: (np.array(e["value"]["rotation"], float), np.array(e["value"]["center"], float))
        for e in d["extrinsics"]}
cams = sorted(((int(frames[v["filename"]]["color_idx"]), int(frames[v["filename"]]["log_time_ns"]), pose[v["id_pose"]])
               for v in view.values() if v["id_pose"] in pose), key=lambda c: c[0])
C = np.array([c[2][1] for c in cams])
t_cam = np.array([c[1] for c in cams], np.int64)
X = np.array([s["value"]["X"] for s in d["structure"]], float)
print(f"{len(cams)} cameras, {len(X)} sparse points")

# axis of the camera path; orient it along the direction of travel
# The pipe of 172011 is not straight (the camera path has a ~15-20 deg bend near the middle), so:
#  - travel = arc length of the camera path (smoothed over 5 cameras), not a projection on one line
#  - the pipe radius is fitted per stretch of SEG cameras along that stretch's own direction; the global
#    model radius is the median over the stretches inside the pipe
k5 = np.ones(5) / 5
Cs = np.column_stack([np.convolve(np.pad(C[:, j], 2, mode="edge"), k5, mode="valid") for j in range(3)])
s_cam = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(Cs, axis=0), axis=1))])
path_len = s_cam[-1]
SEG = 40
seg_rows = []
for k0 in range(0, len(cams) - SEG // 2, SEG // 2):
    k1 = min(len(cams), k0 + SEG) - 1
    d = Cs[k1] - Cs[k0]
    if np.linalg.norm(d) < 1e-9:
        continue
    d /= np.linalg.norm(d)
    q = X - Cs[k0]
    sx = q @ d
    rx = np.linalg.norm(q - np.outer(sx, d), axis=1)
    L = (Cs[k1] - Cs[k0]) @ d
    m = (sx > 0) & (sx < L + 0.3 * max(L, 1e-9) + 0.1 * path_len)
    if m.sum() < 200:
        continue
    m &= rx < 3 * np.median(rx[m])
    try:
        f2 = fit_cylinder(X[m], d0=d)
    except ValueError:
        continue
    seg_rows.append((k0, k1, s_cam[k0], s_cam[k1], f2["radius"], int(m.sum()), f2["rms"] / f2["radius"]))
inside = [r for r in seg_rows if r[3] < 0.85 * path_len and r[6] < 0.15]
R_model = float(np.median([r[4] for r in inside]))
scale = (diam / 2000.0) / R_model
print(f"pipe radius in model units {R_model:.4f} (median of {len(inside)} stretches; spread "
      f"{min(r[4] for r in inside) / R_model:.3f}-{max(r[4] for r in inside) / R_model:.3f}); scale from D = {diam:.0f} mm: "
      f"{scale:.4f} m per unit; camera path {path_len:.4f} units = {path_len * scale:.3f} m")
lateral = np.zeros(len(C))

# references
enc = list(csv.DictReader(open(os.path.join(EX, "encoders.csv"))))
te = np.array([int(r["log_time_ns"]) for r in enc], np.int64)
se = np.array([(float(r["v6"]) - float(r["v7"])) / 2 for r in enc])
s_enc = np.interp(t_cam, te, se)
ref = {"encoders": (s_enc - s_enc[0])}
vis = s_cam * scale
res_rows = []
for name, sref in ref.items():
    err = vis - sref
    res_rows.append((name, sref[-1], vis[-1], 100 * (vis[-1] - sref[-1]) / sref[-1]))
    print(f"travel {name}: {sref[-1]:.3f} m, vision {vis[-1]:.3f} m ({100 * (vis[-1] - sref[-1]) / sref[-1]:+.1f}%); "
          f"max |vision - {name}| along the run {1000 * np.abs(err).max():.0f} mm")
tags = {}
for cam_name, sign in (("usbcam", 1), ("color", -1)):
    p = os.path.join(os.path.dirname(OUT), "tag", f"tag_track_{cam_name}.csv")
    if os.path.exists(p):
        rows = list(csv.DictReader(open(p)))
        tt = np.array([int(r["log_time_ns"]) for r in rows], np.int64)
        dist = 0.1 * np.array([float(r["dist_unit"]) for r in rows])   # tag side 100 mm
        tags[cam_name] = (tt, sign * dist)

fig, ax = plt.subplots(3, 1, figsize=(13, 11))
tsec = (t_cam - te[0]) / 1e9
ax[0].plot((te - te[0]) / 1e9, se - s_enc[0], label="wheel encoders")
ax[0].plot(tsec, vis, ".", ms=3, label=f"vision SfM, scale from D = {diam:.0f} mm")
for nm, (tt, dd) in tags.items():
    # tag distance known up to a constant: align it with the encoders over its own span
    off = np.median(np.interp(tt, te, se) - s_enc[0] - dd)
    ax[0].plot((tt - te[0]) / 1e9, dd + off, ".", ms=1, alpha=.5, label=f"{nm} tag (100 mm), offset matched")
ax[0].set_ylabel("travel along the pipe (m)"); ax[0].set_xlabel("time (s)"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(s_enc - s_enc[0], 1000 * (vis - (s_enc - s_enc[0])), ".", ms=3)
ax[1].set_ylabel("vision - encoder (mm)"); ax[1].set_xlabel("encoder travel (m)"); ax[1].grid(alpha=.3)
if seg_rows:
    mid = [np.interp((r[2] + r[3]) / 2, s_cam, s_enc - s_enc[0]) for r in seg_rows]
    ax[2].plot(mid, [r[4] / R_model for r in seg_rows], "o-")
    ax[2].axhline(1, color="k", lw=.5)
ax[2].set_ylabel("local / global pipe radius (model)"); ax[2].set_xlabel("encoder travel at stretch centre (m)"); ax[2].grid(alpha=.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "vision_vs_references.png"), dpi=85)
with open(os.path.join(OUT, "vision_travel.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["color_idx", "log_time_ns", "s_vision_m", "s_encoder_m", "lateral_mm"])
    for c, sv, sr, la in zip(cams, vis, s_enc - s_enc[0], lateral * scale * 1000):
        w.writerow([c[0], c[1], f"{sv:.5f}", f"{sr:.5f}", f"{la:.2f}"])
print("local radius / global, per stretch (rms/R):", " ".join(f"{r[4] / R_model:.3f}({r[6]:.2f})" for r in seg_rows))
os.remove(js)
