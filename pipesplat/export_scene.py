"""Export a short stretch of a vision work folder for the gsplat experiments (runs on Windows, repo Python).

usage: python -I pipesplat/export_scene.py WORK_DIR VIDEO OUT_DIR [--frames 30] [--held 5] [--at 0.5] [--ahead 1.5]
       WORK_DIR  scripts/vision/run_pipeline.py work folder after step 13 (mvs/scene_textured.obj, frames.csv, sfm/recon)
       VIDEO     the original video (frames are read as recorded, no CLAHE - the same frames lampmap compares with)
       OUT_DIR   scene.npz + scene.json

Takes --frames consecutive registered model frames around --at (fraction of the run), their poses in pipe
coordinates (metres; x along the pipe, y left, z up - render_views.load_model) converted to OpenCV world-to-camera
matrices (x right, y down, z forward), the registered intrinsics (focal, principal point, radial k1), the raw video
frames, --held held-out frames halfway between model frames (pose interpolated as in lampmap), and the textured
OpenMVS mesh faces from just behind the first camera to --ahead metres past the last one as coloured points (face
centroid, texture colour at the centroid UV, face normal) to initialise Gaussians.
"""
import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(os.path.dirname(HERE), "scripts")
sys.path.insert(0, os.path.join(SCRIPTS, "vision"))
sys.path.insert(0, SCRIPTS)
import paths  # noqa: E402
import render_views as rv  # noqa: E402
from video_compare import source_frame_reader  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)


def cv_viewmat(eye, ahead, above):
    """world-to-camera 4x4 in the OpenCV convention for a camera at eye looking at ahead, with above-eye as up
    (the same right/up/forward axes as render_views.camera)"""
    f = (ahead - eye) / np.linalg.norm(ahead - eye)
    r = np.cross(f, above - eye)
    r /= np.linalg.norm(r)
    up = np.cross(r, f)
    R = np.stack([r, -up, f])
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = R, -R @ eye
    return m


def bilinear(img, px, py):
    """img sampled at (px, py) pixel positions (pixel centres at integers); cv2.remap cannot take maps wider
    than 32767, and there are hundreds of thousands of points"""
    x0 = np.clip(np.floor(px).astype(np.int64), 0, img.shape[1] - 2)
    y0 = np.clip(np.floor(py).astype(np.int64), 0, img.shape[0] - 2)
    wx, wy = (px - x0)[:, None], (py - y0)[:, None]
    im = img.astype(np.float32)
    return ((1 - wy) * ((1 - wx) * im[y0, x0] + wx * im[y0, x0 + 1])
            + wy * ((1 - wx) * im[y0 + 1, x0] + wx * im[y0 + 1, x0 + 1])).round()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("work"), ap.add_argument("video"), ap.add_argument("out")
    ap.add_argument("--frames", type=int, default=30), ap.add_argument("--held", type=int, default=5)
    ap.add_argument("--at", type=float, default=0.5), ap.add_argument("--ahead", type=float, default=1.5)
    ap.add_argument("--plain", action="store_true"), ap.add_argument("--crop", nargs=4, type=int)
    ap.add_argument("--stretch-x", type=float, default=1.0)
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)

    model = rv.load_model(paths.CONVERTER, a.work, [None], straighten=False)
    frames = {r["file"]: int(r["source_frame"]) for r in csv.DictReader(open(os.path.join(a.work, "frames.csv"), encoding="utf-8"))}
    src = np.array([frames[n] for n in model["cam_file"]])
    order = np.argsort(src)
    step = int(np.median(np.diff(src[order])))
    i0 = int(np.clip(round(a.at * len(order)) - a.frames // 2, 0, len(order) - a.frames))
    win = order[i0:i0 + a.frames]
    gaps = np.diff(src[win])
    print(f"{len(order)} registered frames, step {step}; window = source frames {src[win[0]]}..{src[win[-1]]} "
          f"({a.frames} frames, largest gap {gaps.max()}) [{time.time() - t0:.0f} s]")

    intr = model["intrinsics"]
    f, (cx, cy), k1 = intr["focal_length"], intr["principal_point"], intr.get("disto_k1", [0.0])[0]
    w, h = int(intr["width"]), int(intr["height"])
    read = source_frame_reader(a.video, a.plain, a.crop, a.stretch_x)

    def grab(key):
        fr = read(key)
        if fr is None or fr.shape[:2] != (h, w):
            sys.exit(f"video frame {key}: {None if fr is None else fr.shape} - expected {h}x{w}")
        return fr[:, :, ::-1].copy()                     # RGB

    eyes, views, imgs = [], [], []
    for k in win:
        eye, ahead, above = model["cam_xyz"][k], model["cam_ahead"][k], model["cam_above"][k]
        eyes.append(eye), views.append(cv_viewmat(eye, ahead, above)), imgs.append(grab(src[k]))

    held_keys, held_views, held_imgs, held_eyes = [], [], [], []
    pairs = [(win[j], win[j + 1]) for j in range(len(win) - 1) if src[win[j + 1]] - src[win[j]] == step]
    for k0, k1_ in (pairs[j] for j in np.linspace(0, len(pairs) - 1, a.held).round().astype(int)):
        s = int((src[k0] + src[k1_]) // 2)
        wt = (s - src[k0]) / (src[k1_] - src[k0])
        g = lambda q: (1 - wt) * model[q][k0] + wt * model[q][k1_]  # noqa: E731
        held_keys.append(s), held_eyes.append(g("cam_xyz"))
        held_views.append(cv_viewmat(g("cam_xyz"), g("cam_ahead"), g("cam_above"))), held_imgs.append(grab(s))
    print(f"frames read: {len(imgs)} model + {len(held_imgs)} held-out {held_keys} [{time.time() - t0:.0f} s]")

    # coloured points = textured mesh faces in the stretch the cameras look at
    x0, x1 = min(e[0] for e in eyes) - 0.05, max(e[0] for e in eyes) + a.ahead
    cen = model["centroid"]
    sel = np.flatnonzero((cen[:, 0] > x0) & (cen[:, 0] < x1) & ~model["blank"])
    tri = model["tri"][sel]
    uv = model["uv"][sel].mean(axis=1)
    tex = model["tex"][0]
    px = np.clip(uv[:, 0] * tex.shape[1] - 0.5, 0, tex.shape[1] - 1)
    py = np.clip((1 - uv[:, 1]) * tex.shape[0] - 0.5, 0, tex.shape[0] - 1)
    col = bilinear(tex, px, py)[:, ::-1]                     # BGR texture -> RGB
    xyz = model["xyz"]
    nrm = np.cross(xyz[tri[:, 1]] - xyz[tri[:, 0]], xyz[tri[:, 2]] - xyz[tri[:, 0]])
    edge = np.sqrt(np.linalg.norm(nrm, axis=1))           # sqrt(2 x face area) ~ edge length of the face
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-30
    print(f"points: {len(sel):,} textured faces with centroid x in {x0:.2f}..{x1:.2f} m, face size median "
          f"{np.median(edge) * 1000:.1f} mm [{time.time() - t0:.0f} s]")

    out_npz = os.path.join(a.out, "scene.npz")
    np.savez_compressed(
        out_npz, images=np.stack(imgs), viewmats=np.stack(views).astype(np.float32), eyes=np.stack(eyes).astype(np.float32),
        frame_keys=src[win].astype(np.int32),
        held_images=np.stack(held_imgs), held_viewmats=np.stack(held_views).astype(np.float32),
        held_eyes=np.stack(held_eyes).astype(np.float32), held_keys=np.array(held_keys, np.int32),
        points=cen[sel].astype(np.float32), colors=col.astype(np.uint8), normals=nrm.astype(np.float32),
        face_size=edge.astype(np.float32),
        # pixel-centre convention: OpenMVG/OpenCV put pixel centres at integer coordinates, gsplat at +0.5
        K=np.array([[f, 0, cx + 0.5], [0, f, cy + 0.5], [0, 0, 1]], np.float32), k1=np.float32(k1),
        size=np.array([w, h], np.int32))
    meta = dict(work=a.work, video=a.video, frames=a.frames, held=a.held, at=a.at, ahead_m=a.ahead, step=step,
                source_frames=[int(s) for s in src[win]], held_out_frames=held_keys, x_range_m=[float(x0), float(x1)],
                points=int(len(sel)), focal=float(f), principal_point=[float(cx), float(cy)], k1=float(k1), size=[w, h])
    json.dump(meta, open(os.path.join(a.out, "scene.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"wrote {out_npz} ({os.path.getsize(out_npz) / 1e6:.0f} MB) [{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
