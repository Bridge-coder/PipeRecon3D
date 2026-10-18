"""F5 evaluation: vision-only vs LiDAR + encoders vs fused, against the AprilTag references.

usage: python -I f5_eval.py EXTRACT_DIR VISION_WORK RESULTS_DIR
writes RESULTS_DIR/fusion/f5_eval.json and prints the tables used in the write-up
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from timebase import load_offsets, stream_times  # noqa: E402
from vision_model import load_cameras, pose_at, _expmap, _rotvec  # noqa: E402

EX, VW, RES = sys.argv[1:4]
OUT = os.path.join(RES, "fusion")
off = load_offsets(os.path.join(OUT, "time_offsets.json"))
cam = load_cameras(EX, VW, off)
legs = json.load(open(os.path.join(OUT, "vision_legs.json")))
tagj = json.load(open(os.path.join(OUT, "f4_tag_scale.json")))
lam_tag = tagj["mm_per_unit_tag"] / 1000
rot = json.load(open(os.path.join(OUT, "extrinsic_rotation.json")))
R_cl = np.array(rot["R_cl"])
reg = json.load(open(os.path.join(OUT, "f3_register_leg2.json")))
_tf = os.path.join(OUT, "extrinsic_translation_final.json")
t_cl = np.array(json.load(open(_tf if os.path.exists(_tf) else os.path.join(OUT, "extrinsic_translation.json")))["t_cl_m"])
tE, rE = stream_times(EX, off, "encoders")
sE = np.array([(float(r["v6"]) - float(r["v7"])) / 2 for r in rE])
report = {}

# ---------- 1. travel against the two tags
# vision-only: arc length of the camera path, scale from the nominal 200 mm diameter (median stretch radius,
# as in the single-modal evaluation) and, for reference, with the tag scale
vt = list(csv.DictReader(open(os.path.join(RES, "vision", "vision_travel.csv"))))
s_vis200 = np.array([float(r["s_vision_m"]) for r in vt])          # D = 200 mm scale (eval_sfm.py)
s_vis_units = cam["s"]
tags = {}
col_rows = {int(r["idx"]): r for r in csv.DictReader(open(os.path.join(EX, "images", "color_index.csv")))}
tcol, rcol = stream_times(EX, off, "color")
tcol_by = {int(r["idx"]): t for r, t in zip(rcol, tcol)}
tU, rU = stream_times(EX, off, "usbcam")
tU_by = {int(r["idx"]): t for r, t in zip(rU, tU)}
for camname, sign, tby in (("usbcam", 1, tU_by), ("color", -1, tcol_by)):
    rows = list(csv.DictReader(open(os.path.join(RES, "tag", f"tag_track_{camname}.csv"))))
    t = np.array([tby[int(r["idx"])] for r in rows], np.int64)
    d = sign * 0.1 * np.array([float(r["dist_unit"]) for r in rows])
    tags[camname] = (t, d)
travel = []
for camname, (tt, dd) in tags.items():
    # compare the change in travel between the first and last 2 s of the tag's visibility (medians -> robust)
    a0, a1 = tt[0], tt[0] + 2e9
    b0, b1 = tt[-1] - 2e9, tt[-1]
    ma, mb = (tt >= a0) & (tt <= a1), (tt >= b0) & (tt <= b1)
    ref = np.median(dd[mb]) - np.median(dd[ma])
    ta, tb = np.median(tt[ma]), np.median(tt[mb])
    enc = np.interp(tb, tE, sE) - np.interp(ta, tE, sE)
    v200 = np.interp(tb, cam["t"], s_vis200) - np.interp(ta, cam["t"], s_vis200)
    vtag = (np.interp(tb, cam["t"], s_vis_units) - np.interp(ta, cam["t"], s_vis_units)) * lam_tag
    row = {"reference": f"{camname} tag", "ref_travel_m": ref,
           "vision_D200_m": v200, "vision_tagscale_m": vtag, "lidar_encoder_m": enc, "fused_m": enc}
    travel.append(row)
report["travel"] = travel

# ---------- 2. diameters per pipe section
from paths import FUSED_BIG  # noqa: E402
d = np.load(os.path.join(FUSED_BIG, "lidar_coloured_pipe_coords.npz"))
s_l, r_l = d["s"], d["r"]
# encoder travel at which the camera passes the bend
u_cam = (cam["C"] - np.array(legs["bend_point"])) @ np.array(legs["legs"][1]["dir"])
t_bend = cam["t"][int(np.argmin(np.abs(u_cam)))]
s_bend_cam = np.interp(t_bend, tE, sE) - np.interp(cam["t"][0], tE, sE)
lead = float(t_cl[2])          # LiDAR origin ahead of the camera along the optical axis (m)
# map s of a wall point = its distance from where the LiDAR origin was at the start; the LiDAR started
# `lead` ahead of the camera, and the camera reaches the bend after s_bend_cam of travel
s_bend = s_bend_cam - lead
sec = {"leg 1 (black)": (s_l > 0.25) & (s_l < s_bend - 0.1), "leg 2 (grey)": (s_l > s_bend + 0.1) & (s_l < s_l.max() - 0.15)}
diam = []
for i, (name, m) in enumerate(sec.items()):
    diam.append({"section": name, "lidar_D_mm": 2000 * float(np.median(r_l[m])), "lidar_points": int(m.sum()),
                 "vision_D_tagscale_mm": 2000 * legs["legs"][i]["radius_units"] * lam_tag,
                 "vision_D_encoderscale_mm": 2000 * legs["legs"][i]["radius_units"] * 1.730 / cam["s"][-1]})
report["diameter"] = diam
report["bend_s_lidar_map_m"] = s_bend

# ---------- 3. bend angle: vision legs; LiDAR axis + LiDAR gyro (LiDAR-only); camera gyro heading change
fits = {int(r["idx"]): r for r in csv.DictReader(open(os.path.join(EX, "lidar_fit_short.csv"))) if r.get("diameter_mm")}
tS, rS = stream_times(EX, off, "lidar")
tL, rL = stream_times(EX, off, "imu_lidar")
GL = np.array([[float(r[k]) for k in ("gx", "gy", "gz")] for r in rL]) - np.array(rot["gyro_bias_lidar"])


def lidar_axis_world(t, dL):
    R, _ = pose_at(cam, t)
    return R.T @ (R_cl @ dL)


def gyro_rotation(t0, t1):
    """LiDAR body rotation from t0 to t1 by integrating the gyro (R_t0<-t1)."""
    m = (tL >= t0) & (tL <= t1)
    R = np.eye(3)
    tt = tL[m] / 1e9
    for k in range(1, m.sum()):
        R = R @ _expmap(0.5 * (GL[m][k] + GL[m][k - 1]) * (tt[k] - tt[k - 1]))
    return R


t_split = cam["t"][legs["split_camera"]]
cands1 = [(t, fits[int(r["idx"])]) for t, r in zip(tS, rS) if int(r["idx"]) in fits and t_split - 40e9 < t < t_split - 25e9]
cands2 = [(t, fits[int(r["idx"])]) for t, r in zip(tS, rS) if int(r["idx"]) in fits and t_split + 5e9 < t < t_split + 20e9]
vec = lambda f: np.array([float(f["dir_x"]), float(f["dir_y"]), float(f["dir_z"])])  # noqa: E731
A1 = np.mean([lidar_axis_world(t, vec(f)) for t, f in cands1], axis=0)
A2 = np.mean([lidar_axis_world(t, vec(f)) for t, f in cands2], axis=0)
bend_lidar_vis = np.degrees(np.arccos(np.clip(A1 @ A2 / np.linalg.norm(A1) / np.linalg.norm(A2), -1, 1)))
# LiDAR-only: axis at t1 (leg 1) vs axis at t2 (leg 2) brought into the t1 frame by the LiDAR gyro
t1, f1 = cands1[len(cands1) // 2]
t2, f2 = cands2[len(cands2) // 2]
Rg = gyro_rotation(t1, t2)
d1, d2 = vec(f1), Rg @ vec(f2)
bend_lidar_only = np.degrees(np.arccos(np.clip(d1 @ d2 / np.linalg.norm(d1) / np.linalg.norm(d2), -1, 1)))
a1, a2 = np.array(legs["legs"][0]["dir"]), np.array(legs["legs"][1]["dir"])
report["bend_deg"] = {"vision_dense_legs": float(np.degrees(np.arccos(np.clip(a1 @ a2, -1, 1)))),
                      "lidar_axes_with_vision_rotation": float(bend_lidar_vis),
                      "lidar_only_axes_plus_lidar_gyro": float(bend_lidar_only),
                      "camera_heading_change_sfm": float(np.degrees(np.linalg.norm(_rotvec(cam["R"][0] @ cam["R"][-1].T))))}

# ---------- 4. extrinsics and clocks
report["extrinsic"] = {"R_cl_rotvec_deg": np.degrees(_rotvec(R_cl)).tolist(), "t_cl_mm": (t_cl * 1000).tolist(),
                       "registration_variants_leg2": {k: {kk: v[kk] for kk in ("t_cl_m", "scale_vs_encoder", "range_bias_mm",
                                                                                "rot_refine_deg", "rms_mm", "median_abs_mm")}
                                                      for k, v in reg["variants"].items()}}
report["clocks"] = {k: (v.get("offset_ns"), v.get("lag_ns")) for k, v in off.items()}
json.dump(report, open(os.path.join(OUT, "f5_eval.json"), "w"), indent=1)

print("TRAVEL between the first and last 2 s in which each tag is seen (tag side 100 mm):")
print(f"{'reference':12s} {'tag':>8s} {'vision D200':>12s} {'vision tag-scale':>17s} {'LiDAR+enc':>10s} {'fused':>8s}")
for r in travel:
    ref = r["ref_travel_m"]
    fmt = lambda x: f"{x:.3f} ({100 * (x - ref) / ref:+.1f}%)"  # noqa: E731
    print(f"{r['reference']:12s} {ref:8.3f} {fmt(r['vision_D200_m']):>16s} {fmt(r['vision_tagscale_m']):>17s} "
          f"{fmt(r['lidar_encoder_m']):>16s} {fmt(r['fused_m']):>16s}")
print("DIAMETER per section (mm):")
for r in diam:
    print(f"  {r['section']:14s} LiDAR (0.2-0.6 m ranges) {r['lidar_D_mm']:.1f} | vision tag scale {r['vision_D_tagscale_mm']:.1f} | "
          f"vision encoder scale {r['vision_D_encoderscale_mm']:.1f}")
print("BEND angle (deg):", {k: round(v, 1) for k, v in report["bend_deg"].items()})
print(f"bend in LiDAR map coordinates at s = {s_bend:.3f} m (camera passes it at encoder {s_bend_cam:.3f} m)")
