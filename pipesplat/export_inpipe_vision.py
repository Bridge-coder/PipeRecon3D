"""InPipe 172011 with vision-consistent poses and geometry (docs/10 X7), for the same evaluation as before.

docs/10 X6: the fused poses (axial = encoder, lateral = vision x AprilTag scale, orientation = SfM) and the LiDAR
height field (radius 13-15 % too small against the RealSense depth) do not agree with the images at the mm level,
and every model anchored on them breaks 10 cm away from the training views. concrete_2, where poses and geometry
both come from the images (SfM + OpenMVS), does not break. This export gives InPipe the concrete_2 treatment:
  poses    = the vision SfM cameras of INPIPE_VISION (OpenMVG, fixed intrinsics), metric by the AprilTag scale
             (results/inpipe_172011/fusion/f4_tag_scale.json); held-out frames slerp/lerp between their SfM
             neighbours by time (vision_model.pose_at), block frames their own SfM pose
  geometry = that work folder's OpenMVS mesh (scene_textured.obj) and dense cloud, same scale
  frames   = exactly the frames, images, next frames, B renders and masks of the existing InPipe scene
             (BASE_DIR/scene.npz), so every number stays comparable with docs/08-10
The LiDAR is not used. The pose frame is the SfM frame x scale, i.e. not the fused frame - only the images,
B and the masks are shared, which do not depend on the frame.

usage: python -I pipesplat/export_inpipe_vision.py BASE_DIR OUT_DIR
writes OUT_DIR/scene.npz (vision kind: no depth_scan) and OUT_DIR/surface.npz
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPTS = os.path.join(ROOT, "scripts")
for p in (os.path.join(SCRIPTS, "vision"), os.path.join(SCRIPTS, "recon"), os.path.join(SCRIPTS, "fusion"), SCRIPTS, HERE):
    sys.path.insert(0, p)
import paths  # noqa: E402
import render_views as rv  # noqa: E402
from ply_io import read_ply_vertices  # noqa: E402
from timebase import load_offsets  # noqa: E402
from vision_model import load_cameras, pose_at  # noqa: E402
from export_surface import faces  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)


def viewmat(R, C):
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = R, -R @ C
    return m


def main():
    base, out = sys.argv[1], sys.argv[2]
    t0 = time.time()
    os.makedirs(out, exist_ok=True)
    d = dict(np.load(os.path.join(base, "scene.npz")))
    off = load_offsets(os.path.join(paths.INPIPE_RESULTS, "fusion", "time_offsets.json"))
    cam = load_cameras(paths.INPIPE_EXTRACT, paths.INPIPE_VISION, off)
    lam = json.load(open(os.path.join(paths.INPIPE_RESULTS, "fusion", "f4_tag_scale.json")))["mm_per_unit_tag"] / 1000
    by_idx = {int(i): k for k, i in enumerate(cam["idx"])}
    missing = [int(i) for i in d["train_ids"] if int(i) not in by_idx]
    if missing:
        sys.exit(f"training frames without an SfM pose: {missing[:10]}")
    tr = [by_idx[int(i)] for i in d["train_ids"]]
    d["train_viewmats"] = np.stack([viewmat(cam["R"][k], cam["C"][k] * lam) for k in tr]).astype(np.float32)
    d["train_eyes"] = (cam["C"][tr] * lam).astype(np.float32)
    held = [pose_at(cam, int(t)) for t in d["held_t"]]
    d["held_viewmats"] = np.stack([viewmat(R, C * lam) for R, C in held]).astype(np.float32)
    d["block_viewmats"] = d["train_viewmats"][d["block_idx"]]
    # camera: the SfM's single focal length at the exported width (area-averaging pixel-centre convention)
    s = float(d["size"][0]) / 1280.0
    f, (cx, cy) = 927.9388 * s, ((656.8645 + 0.5) * s - 0.5, (373.5485 + 0.5) * s - 0.5)
    d["K"] = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], np.float32)
    # geometry: OpenMVS dense cloud (Gaussian starts and depth) and mesh (measured surface), metric
    v = read_ply_vertices(os.path.join(paths.INPIPE_VISION, "mvs", "scene_dense.ply"))
    P = np.c_[v["x"], v["y"], v["z"]].astype(np.float64) * lam
    C = cam["C"] * lam
    near = np.concatenate([np.min(np.linalg.norm(P[i:i + 100000, None, :] - C[None, ::10, :], axis=2), axis=1)
                           for i in range(0, len(P), 100000)])
    keep = near < 1.5
    d["points"] = d["depth_points"] = P[keep].astype(np.float32)
    for k in ("depth_scan", "scan_t"):
        d.pop(k, None)
    xyz, tri, _ = rv.read_obj(os.path.join(paths.INPIPE_VISION, "mvs", "scene_textured.obj"))
    c, n, sz = faces(xyz * lam, tri)
    np.savez(os.path.join(out, "surface.npz"), centres=c, normals=n, size=sz)
    np.savez(os.path.join(out, "scene.npz"), **d)
    meta = dict(kind="inpipe-vision", base=base, scale_m_per_unit=lam, train=int(len(tr)), held=int(len(held)),
                block_frames=int(len(d["block_idx"])), points=int(len(d["points"])), surface_triangles=int(len(c)),
                K=d["K"].tolist(), size=d["size"].tolist())
    json.dump(meta, open(os.path.join(out, "scene.json"), "w"), indent=1)
    print(f"{len(tr)} training frames, {len(held)} held-out, {len(d['block_idx'])} block; dense {keep.sum():,} of {len(P):,} "
          f"points within 1.5 m of the cameras; mesh {len(c):,} triangles (size median {np.median(sz) * 1000:.2f} mm); "
          f"scale {lam * 1000:.2f} mm/unit -> {out} [{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
