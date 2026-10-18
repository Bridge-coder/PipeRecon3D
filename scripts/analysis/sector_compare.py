"""Vision dense (x AprilTag scale) vs fused LiDAR wall, radius per 30-degree sector about the same axis (the vision
cylinder fit of each leg), middle 0.15-0.75 m of each leg. Then diff(theta) = c + a cos + b sin separates a radius
difference (c) from a lateral offset between the two clouds (a, b).  usage: python -I sector_compare.py"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ply_io import read_ply_vertices
import paths
RES = os.path.join(paths.INPIPE_RESULTS, "fusion")
lam = json.load(open(f"{RES}/f4_tag_scale.json"))["mm_per_unit_tag"] / 1000
legs = json.load(open(f"{RES}/vision_legs.json"))
V = read_ply_vertices(os.path.join(paths.INPIPE_VISION, "mvs", "scene_dense.ply"))
XV = np.column_stack([V["x"], V["y"], V["z"]]).astype(float) * lam
with open(os.path.join(paths.FUSED_BIG, "lidar_coloured_world.ply"), "rb") as f:
    while f.readline().strip() != b"end_header":
        pass
    rec = np.frombuffer(f.read(), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
XL = np.column_stack([rec["x"], rec["y"], rec["z"]]).astype(float)
B = np.array(legs["bend_point"]) * lam
for L in legs["legs"]:
    d, p, R = np.array(L["dir"]), np.array(L["point"]) * lam, L["radius_units"] * lam
    up = np.array([0, -1.0, 0]); up -= (up @ d) * d; up /= np.linalg.norm(up); side = np.cross(d, up)
    sb = (B - p) @ d
    out = []
    for X in (XV, XL):
        q = X - p; s = q @ d; rr = q - np.outer(s, d); r = np.linalg.norm(rr, axis=1)
        th = np.degrees(np.arctan2(rr @ side, rr @ up))
        # middle of the leg, at least 0.15 m from the bend, only the wall
        m = (np.abs(r - R) < 0.03) & ((s < sb - 0.15) & (s > sb - 0.75) if L["name"] == "leg1" else (s > sb + 0.15) & (s < sb + 0.75))
        out.append((th[m], r[m]))
    print(f"{L['name']}: axis = vision cylinder fit (R {R * 1000:.1f} mm); per 30-deg sector: median radius vision / LiDAR [mm] (points)")
    diffs, sec_mid = [], []
    for a in range(-180, 180, 30):
        mv = (out[0][0] >= a) & (out[0][0] < a + 30); ml = (out[1][0] >= a) & (out[1][0] < a + 30)
        if mv.sum() > 300 and ml.sum() > 300:
            rv, rl = np.median(out[0][1][mv]) * 1000, np.median(out[1][1][ml]) * 1000
            diffs.append(rv - rl)
            sec_mid.append(a + 15)
            print(f"   {a:4d}..{a + 30:4d}: {rv:6.1f} / {rl:6.1f}  diff {rv - rl:+5.1f}   ({mv.sum():,} / {ml.sum():,})")
        else:
            print(f"   {a:4d}..{a + 30:4d}: too few points ({mv.sum()} / {ml.sum()})")
    print(f"   sectors with both: {len(diffs)}, vision - LiDAR median {np.median(diffs):+.1f} mm, range {min(diffs):+.1f} .. {max(diffs):+.1f}")
    t = np.radians(np.array(sec_mid)); dd = np.array(diffs)
    A = np.c_[np.ones_like(t), np.cos(t), np.sin(t)]
    x = np.linalg.lstsq(A, dd, rcond=None)[0]
    print(f"   fit: radius difference vision - LiDAR {x[0]:+.1f} mm; vision wall shifted up {x[1]:+.1f} / right {x[2]:+.1f} mm "
          f"relative to the LiDAR wall (|{np.hypot(x[1], x[2]):.1f}| mm); rms residual {np.sqrt(np.mean((A @ x - dd) ** 2)):.1f} mm")
