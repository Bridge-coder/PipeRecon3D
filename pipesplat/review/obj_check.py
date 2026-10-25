"""docs/10 X4c / X8: textured OBJs in the export_tex_obj.py layout (wall material + optional end-cap material) and the
OpenMVS textured mesh drawn by the repository's rasteriser (render_views + lampmap.Renderer) from held-out poses, next
to the raw video frame. Texture only - no lamp map, no brightness matching: what a plain viewer shows.

usage: python -I pipesplat/review/obj_check.py WORK_DIR VIDEO|SCENE.npz OUT.jpg --frames F1 F2 .. --obj LABEL=DIR/model.obj ..
           [--openmvs] [--ahead 3.0]
VIDEO: the original video (OpenCV cannot open paths with non-ASCII characters on Windows) or the scene.npz of
export_g0.py, whose held-out raw frames (held_images, held_ids) are the same frames
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SCRIPTS = os.path.join(ROOT, "scripts")
for p in (os.path.join(SCRIPTS, "vision"), SCRIPTS, os.path.join(ROOT, "lampmap")):
    sys.path.insert(0, p)
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import paths  # noqa: E402
import render_views as rv  # noqa: E402
from video_compare import source_frame_reader  # noqa: E402
import lampmap as lm  # noqa: E402


def read_tex_obj(path):
    """export_tex_obj.py OBJ: vertices, triangles, per-corner UVs (m, 3, 2) in one atlas (wall on top, cap below
    it at the left) and the atlas image (BGR)"""
    data = open(path, "rb").read()                   # layout of export_tex_obj.py: v.., vt.., usemtl wall f.., [usemtl cap f..]
    i_v, i_vt, i_w = data.find(b"\nv "), data.find(b"\nvt "), data.find(b"\nusemtl wall")
    i_c = data.find(b"\nusemtl cap")
    v = np.fromstring(data[i_v:i_vt].replace(b"\nv ", b" "), dtype=np.float64, sep=" ").reshape(-1, 3)
    vt = np.fromstring(data[i_vt:i_w].replace(b"\nvt ", b" "), dtype=np.float64, sep=" ").reshape(-1, 2)

    def faces(block):
        block = block[block.find(b"\nf "):]
        return np.fromstring(block.replace(b"\nf ", b" ").replace(b"/", b" "), dtype=np.int64, sep=" ").reshape(-1, 6) - 1

    fw = faces(data[i_w:i_c if i_c > 0 else len(data)])
    fc = faces(data[i_c:]) if i_c > 0 else np.zeros((0, 6), np.int64)
    f = np.concatenate([fw, fc])
    tri, tuv = f[:, 0::2], f[:, 1::2]
    mat = np.r_[np.zeros(len(fw), bool), np.ones(len(fc), bool)]
    d = os.path.dirname(path)
    rd = lambda n: cv2.imdecode(np.fromfile(os.path.join(d, n), np.uint8), cv2.IMREAD_COLOR)  # noqa: E731
    wall = rd("texture.png")
    Hw, Ww = wall.shape[:2]
    cap = rd("cap.png") if mat.any() else np.zeros((0, 0, 3), np.uint8)
    Hc, Wc = cap.shape[:2]
    pad = 16                                         # seam faces reach just past the last row: wrap, as a viewer does
    atlas = np.zeros((Hw + pad + Hc, max(Ww, Wc), 3), np.uint8)
    atlas[:Hw, :Ww] = wall
    atlas[Hw:Hw + pad, :Ww] = wall[:pad]
    atlas[Hw + pad:, :Wc] = cap
    H, W = atlas.shape[:2]
    uv = vt[tuv].astype(np.float64)                      # (m, 3, 2), OBJ convention (v = 0 at the bottom)
    s, t = uv[..., 0], uv[..., 1]
    s_w, t_w = s * Ww / W, 1 - (1 - t) * Hw / H
    s_c, t_c = s * Wc / W, 1 - (Hw + pad + (1 - t) * Hc) / H
    uv = np.stack([np.where(mat[:, None], s_c, s_w), np.where(mat[:, None], t_c, t_w)], -1).astype(np.float32)
    return v, tri.astype(np.int32), uv, atlas


def label(img, text):
    img = img.copy()
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("work"), ap.add_argument("video"), ap.add_argument("out")
    ap.add_argument("--frames", type=int, nargs="+", required=True)
    ap.add_argument("--obj", nargs="+", default=[])
    ap.add_argument("--openmvs", action="store_true")
    ap.add_argument("--ahead", type=float, default=3.0)
    a = ap.parse_args()
    model = rv.load_model(paths.CONVERTER, a.work, [None], straighten=False)
    frames = {r["file"]: int(r["source_frame"]) for r in csv.DictReader(open(os.path.join(a.work, "frames.csv"), encoding="utf-8"))}
    src = np.array([frames[n] for n in model["cam_file"]])
    order = np.argsort(src)
    if a.video.endswith(".npz"):                 # the held-out raw frames of export_g0.py (RGB)
        sc = np.load(a.video)
        hid, him = sc["held_ids"].tolist(), sc["held_images"]
        read = lambda s: np.ascontiguousarray(him[hid.index(s)][:, :, ::-1])  # noqa: E731
    else:
        read = source_frame_reader(a.video)
    cols = []
    if a.openmvs:
        cols.append(("OpenMVS (CLAHE)", lm.Renderer(model, a.ahead)))
    for spec in a.obj:
        name, path = spec.split("=", 1)
        v, tri, uv, atlas = read_tex_obj(path)
        m = dict(xyz=v, tri=tri, uv=uv, tex=[atlas], blank=np.zeros(len(tri), bool), centroid=v[tri].mean(1),
                 intrinsics=model["intrinsics"])
        cols.append((name, lm.Renderer(m, a.ahead)))
        print(f"{name}: {len(v):,} vertices, {len(tri):,} faces, atlas {atlas.shape[1]}x{atlas.shape[0]}")
    rows = []
    for s in a.frames:
        j = int(np.searchsorted(src[order], s))
        k0, k1 = order[max(j - 1, 0)], order[min(j, len(order) - 1)]
        wgt = 0.0 if src[k1] == src[k0] else (s - src[k0]) / (src[k1] - src[k0])
        pose = lm.lerp_pose(model, k0, k1, min(max(wgt, 0.0), 1.0))
        panels = [label(read(s), f"video {s}")]
        for name, ren in cols:
            rgb, msk = ren(pose)
            panels.append(label(np.clip(rgb, 0, 255).astype(np.uint8)[:, :, ::-1], name))
        rows.append(np.hstack(panels))
        print(f"frame {s} (x {pose[0][0]:.2f} m) done")
    cv2.imencode(".jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
