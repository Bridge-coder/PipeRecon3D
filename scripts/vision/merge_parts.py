"""Put separately reconstructed stretches of one video into common pipe coordinates.

Usage: merge_parts.py CONVERTER_EXE WORK_DIR OUT_PREFIX MERGED_PLY PART [PART ...]

Incremental SfM sometimes cannot carry on across a jolt of the robot, and the video is then reconstructed
in several parts (run_pipeline.py --part), each in its own arbitrary frame and scale. Parts have no images
in common, so they cannot be joined by a rigid fit. Instead every part is described in pipe coordinates:
  axial position  along the fitted pipe axis, in metres, scale re-set slice by slice from the 300 mm bore
                  (slices that are not a 300 mm circle - manholes - take the scale of their neighbours)
  angle           around the axis, 0 = up as seen by the camera (the robot is assumed to stay upright)
  radius          distance from the centre of the local cross-section, in metres
The only link between parts is the axial offset: each part is shifted so that its camera positions agree
with the on-screen distance counter (median difference over its frames).
Outputs: OUT_PREFIX.json / .png and MERGED_PLY, the dense clouds of all parts as one straightened model
(x along the pipe, y to the left, z up, metres: a right-handed frame, so viewers do not show it mirrored).
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as ev  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
converter, work, out_prefix, merged_ply = sys.argv[1:5]
parts = sys.argv[5:]
HALF = ev.PIPE_DIAMETER_M / 2
frames = {r["file"]: int(r["source_frame"]) for r in csv.DictReader(open(os.path.join(work, "frames.csv"), encoding="utf-8"))}
counter = np.array([float(r["counter_m"]) for r in csv.DictReader(open(os.path.join(work, "counter.csv"), encoding="utf-8"))])


summary, clouds = [], []
for part in parts:
    f = ev.pipe_frame(converter, work, part, frames, counter)
    vtx = ev.read_ply_vertices(os.path.join(work, f"mvs_{part}", "scene_dense.ply"))
    x, th, r = f["to_pipe"](np.c_[vtx["x"], vtx["y"], vtx["z"]].astype(float))
    rgb = np.c_[vtx["red"], vtx["green"], vtx["blue"]]
    keep = r < 1.0                                          # drop stray points more than 1 m from the axis
    clouds.append((x[keep], th[keep], r[keep], rgb[keep], np.full(keep.sum(), len(clouds))))
    cnt = counter[f["src"]]
    run = cnt >= 0.25                                       # the counter lags when the robot starts
    i1 = int(np.argmax(run)) if run.any() else 0
    travel, gt = float(f["cam_x"][-1] - f["cam_x"][i1]), float(cnt[-1] - cnt[i1])
    resid = cnt - f["cam_x"]
    summary.append(dict(part=part, frames_registered=int(len(f["src"])), source_frames=[int(f["src"][0]), int(f["src"][-1])],
                        counter_m=[float(cnt[0]), float(cnt[-1])], travel_m=travel, counter_travel_m=gt,
                        travel_error=travel / gt - 1 if gt > 0.5 else None,
                        counter_minus_model_rms_m=float(np.sqrt(np.mean((resid[run] - np.median(resid[run])) ** 2))) if run.sum() > 10 else None,
                        slices=int(len(f["slice_ok"])), slices_300mm_circle=int(f["slice_ok"].sum()),
                        radius_min_max_over_median=[float(np.nanmin(f["slice_rad"][f["slice_ok"]])), float(np.nanmax(f["slice_rad"][f["slice_ok"]]))],
                        dense_points=int(keep.sum()), model_x_m=[float(np.percentile(x[keep], 0.5)), float(np.percentile(x[keep], 99.5))]))
    summary[-1]["_frame"] = f
    e = summary[-1]
    print(f"part {part}: {e['frames_registered']} frames (source {e['source_frames'][0]}..{e['source_frames'][1]}), counter "
          f"{e['counter_m'][0]:.2f}..{e['counter_m'][1]:.2f} m; travel {travel:.2f} m vs counter {gt:.2f} m"
          + (f" ({e['travel_error']:+.1%})" if e["travel_error"] is not None else "")
          + f"; {e['slices_300mm_circle']}/{e['slices']} slices are a 300 mm circle, radius {e['radius_min_max_over_median'][0]:.2f}.."
          f"{e['radius_min_max_over_median'][1]:.2f} of the median; dense points {e['dense_points']:,}")

X, TH, R, RGB, PART = (np.concatenate([c[k] for c in clouds]) for k in range(5))
total_frames = summary[0]["_frame"]["n_views"]
registered = sum(e["frames_registered"] for e in summary)
print(f"all parts: {registered} of {total_frames} frames registered ({registered / total_frames:.0%}); {len(X):,} dense points; "
      f"model spans {np.percentile(X, 0.5):.2f} .. {np.percentile(X, 99.5):.2f} m along the pipe")

# cross-section along the pipe from the dense cloud: where is it a 300 mm pipe, where is it wider?
bins = np.arange(np.floor(np.percentile(X, 0.5) * 10) / 10, np.percentile(X, 99.5), 0.1)
prof = []
for a in bins:
    m = (X >= a) & (X < a + 0.1)
    if m.sum() < 300:
        prof.append((a + 0.05, np.nan, np.nan, 0.0, 0.0, int(m.sum())))
        continue
    wall = (R[m] > 0.85 * HALF) & (R[m] < 1.15 * HALF)
    th = TH[m][wall]
    lower = len(np.unique((th[np.abs(th) >= 120] // 10).astype(int))) / 12.0     # the 120 degrees around the invert
    upper = len(np.unique((th[np.abs(th) < 120] // 10).astype(int))) / 24.0      # the rest of the circumference
    prof.append((a + 0.05, float(np.median(R[m])), float(np.percentile(R[m], 90)), lower, upper, int(m.sum())))
prof = np.array(prof)


def stretches(flag, min_bins):
    runs, start = [], None
    for k, w in enumerate(np.r_[flag, False]):
        if w and start is None:
            start = k
        if not w and start is not None:
            if k - start >= min_bins:
                runs.append((float(prof[start, 0] - 0.05), float(prof[k - 1, 0] + 0.05)))
            start = None
    return runs


empty = prof[:, 5] < 300
gaps = stretches(empty, 2)
full = ~empty & (prof[:, 3] >= 0.5) & (prof[:, 4] >= 0.7)
print("no dense points:", ", ".join(f"{a:.1f}-{b:.1f} m" for a, b in gaps) or "none")
print(f"10 cm bins: {len(prof)} in total, {int(empty.sum())} without dense points, {int(full.sum())} with the wall covered "
      f"all round (at least 70% of the upper 240 degrees and half of the invert)")

out = dict(parts=[{k: v for k, v in e.items() if k != "_frame"} for e in summary], frames_total=total_frames,
           frames_registered=registered, dense_points=int(len(X)), gaps_m=gaps,
           bins_total=int(len(prof)), bins_empty=int(empty.sum()), bins_covered_all_round=int(full.sum()),
           profile=[dict(x_m=float(p[0]), median_radius_m=None if np.isnan(p[1]) else float(p[1]),
                         p90_radius_m=None if np.isnan(p[2]) else float(p[2]), invert_coverage=float(p[3]),
                         upper_coverage=float(p[4]), points=int(p[5])) for p in prof])
json.dump(out, open(out_prefix + ".json", "w"), indent=1)

# straightened model: x along the pipe, y left, z up (right-handed)
os.makedirs(os.path.dirname(os.path.abspath(merged_ply)), exist_ok=True)
rec = np.empty(len(X), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
rec["x"], rec["y"], rec["z"] = X, -R * np.sin(np.radians(TH)), R * np.cos(np.radians(TH))
rec["red"], rec["green"], rec["blue"] = RGB[:, 0], RGB[:, 1], RGB[:, 2]
with open(merged_ply, "wb") as fh:
    fh.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(rec)}\nproperty float x\nproperty float y\n"
              "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode("ascii"))
    rec.tofile(fh)
print("saved", merged_ply)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
rng = np.random.default_rng(0)
k = rng.choice(len(X), min(len(X), 1500000), replace=False)
x0, x1 = np.percentile(X, 0.5) - 0.2, np.percentile(X, 99.5) + 0.2
fig, ax = plt.subplots(5, 1, figsize=(20, 19), sharex=True)
near = R[k] < 2.2 * HALF
ax[0].scatter(X[k][near], TH[k][near], s=0.15, c=RGB[k][near] / 255.0, linewidths=0)
ax[0].set(title="unrolled wall: dense points of all parts, their colours (angle 0 = up)", ylabel="angle [deg]", ylim=(-180, 180))
sc = ax[1].scatter(X[k], TH[k], s=0.15, c=np.clip(R[k] * 1000, 100, 400), cmap="viridis", linewidths=0)
ax[1].set(title="distance from the axis [mm] (150 = wall of a 300 mm pipe; larger = manhole)", ylabel="angle [deg]", ylim=(-180, 180))
fig.colorbar(sc, ax=ax[1], pad=0.01, fraction=0.012)
ax[2].scatter(X[k], rec["z"][k], s=0.15, c=RGB[k] / 255.0, linewidths=0)
ax[2].set(title="side view of the straightened model (red: camera positions)", ylabel="up [m]", ylim=(-0.6, 0.6))
for e in summary:
    f = e["_frame"]
    ax[2].plot(f["cam_x"], f["cam_r"] * np.cos(np.radians(f["cam_theta"])), "r-", lw=0.8)
ax[3].plot(prof[:, 0], prof[:, 1] * 1000, "k-", lw=1, label="median distance from the axis [mm]")
ax[3].axhline(150, color="b", ls="--", lw=0.8)
ax[3].plot(prof[:, 0], prof[:, 4] * 100, "-", color="tab:green", lw=1, label="upper 240 degrees of the wall covered [%]")
ax[3].plot(prof[:, 0], prof[:, 3] * 100, "-", color="tab:brown", lw=1, label="invert (lower 120 degrees) covered [%]")
ax[3].set(title="along the pipe, from the dense cloud (10 cm bins)", ylabel="[mm] / [%]", ylim=(0, 250))
ax[3].legend(loc="upper left", ncol=3)
cols = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
for j, e in enumerate(summary):
    f = e["_frame"]
    ax[4].plot(f["cam_x"], (counter[f["src"]] - f["cam_x"]) * 100, "-", color=cols[j % 5], lw=1,
               label=f"part {e['part']}: counter - model position")
    ax[4].axvspan(f["cam_x"][0], f["cam_x"][-1], color=cols[j % 5], alpha=0.07)
ax[4].axhline(0, color="k", lw=0.6)
ax[4].set(title="camera position along the pipe: on-screen counter minus model (each part shifted to a median of 0)",
          ylabel="[cm]", xlabel="along the pipe [m] (counter scale)", xlim=(x0, x1), ylim=(-30, 30))
ax[4].legend(loc="upper left", ncol=len(summary))
fig.tight_layout()
fig.savefig(out_prefix + ".png", dpi=100)
print("saved", out_prefix + ".json", "and", out_prefix + ".png")
