"""Side-by-side video of a reconstructed inspection video: the input frame next to the model seen from the same
camera, so you can judge frame by frame what was reconstructed and how well.

usage: python -I video_compare.py WORK_DIR OUT_MP4 [--video VIDEO] [--parts A B ...] [--every N] [--fps F]
                                  [--plain [--crop X0 Y0 X1 Y1] [--stretch-x S]] [--straighten]
       WORK_DIR  a run_pipeline.py work folder after step 13 (mvs/scene_textured.obj, or mvs_<part>/ per part)
       --video   the original video: the left panel then shows the untouched frame (PAL stretch and crop as in
                 extract_frames.py when it is a 720x576 Sheffield video); without it the extracted (CLAHE) frame
       --plain/--crop/--stretch-x  the frame geometry given to run_pipeline.py for a non-Sheffield video (it must be
                 the same, or the left panel does not line up with the model; a 720x576 video is otherwise taken
                 for a Sheffield one)
       --parts   separately reconstructed stretches (run_pipeline.py --part), joined in pipe coordinates
       --every   use every Nth registered camera (default 1)
       --straighten  as in render_views.py

Panels (all model views use the registered camera's position, orientation, focal length and lens distortion):
  (1) input frame                       (2) textured mesh (OpenMVS) from the same camera
  (3) mesh coloured by its distance from the fitted bore (blue = inside, red = outside, +-20 mm)
  (4) side view of the whole model, cut open, with the camera's position; on-screen counter vs model position
Scale and axis come from render_views.load_model (pipe fit of the sparse model; bore 300 mm or
PIPE3D_PIPE_DIAMETER_M, see evaluate.py).
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402
import render_views as rv  # noqa: E402

PW, PH = 640, 457                     # panel size (768x548 frames scaled by 5/6)
DEV_MM = 20.0


def font(size):
    for name in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_BIG, F_SMALL = font(20), font(15)


def camera_k(model, k, scale):
    """pinhole camera of registered camera k, oversized to cover the distorted frame, plus the remap grid that
    applies the lens distortion (as render_views.inside_view, but for a given camera index)"""
    eye, ahead, above = model["cam_xyz"][k], model["cam_ahead"][k], model["cam_above"][k]
    intr = model["intrinsics"]
    f, (cx, cy), k1 = intr["focal_length"], intr["principal_point"], intr.get("disto_k1", [0.0])[0]
    w, h = int(round(intr["width"] * scale)), int(round(intr["height"] * scale))
    xd = ((np.arange(w) + 0.5) / scale - 0.5 - cx) / f
    yd = ((np.arange(h) + 0.5) / scale - 0.5 - cy) / f
    xu, yu = rv.undistort_grid(*np.meshgrid(xd, yd), k1)       # NaN where the lens model does not reach (corners)
    focal = f * scale
    cw, ch = 2 * int(np.ceil(np.nanmax(np.abs(xu)) * focal)) + 4, 2 * int(np.ceil(np.nanmax(np.abs(yu)) * focal)) + 4
    cam = rv.camera(eye, ahead, (cw, ch), focal=focal, up=above - eye)
    mx = np.nan_to_num(cw / 2 + xu * focal - 0.5, nan=-10).astype(np.float32)
    my = np.nan_to_num(ch / 2 + yu * focal - 0.5, nan=-10).astype(np.float32)
    return cam, mx, my


def deviation_image(model, cam, r_dev):
    """picture of the mesh coloured by the radial deviation of the surface (diverging blue-white-red)"""
    _, fbuf, b1, b2 = rv.rasterize(cam, model["xyz"], model["tri"], 0.02)
    hit = fbuf >= 0
    f, b1, b2 = fbuf[hit], b1[hit], b2[hit]
    v = model["tri"][f]
    d = (1 - b1 - b2) * r_dev[v[:, 0]] + b1 * r_dev[v[:, 1]] + b2 * r_dev[v[:, 2]]
    t = np.clip(d / DEV_MM, -1, 1)
    rgb = np.empty((len(t), 3), np.float32)
    pos = t > 0
    rgb[pos] = np.c_[np.full(pos.sum(), 255.0), 255 * (1 - t[pos]), 255 * (1 - t[pos])]
    rgb[~pos] = np.c_[255 * (1 + t[~pos]), 255 * (1 + t[~pos]), np.full((~pos).sum(), 255.0)]
    rgb *= 0.55 + 0.45 * np.clip(1.0 - _depth_fade(cam, model, f, b1, b2), 0, 1)[:, None]
    out = np.full(hit.shape + (3,), 255, np.uint8)
    out[hit] = np.clip(rgb, 0, 255)
    return cv2.resize(out, (hit.shape[1] // rv.SS, hit.shape[0] // rv.SS), interpolation=cv2.INTER_AREA)


def _depth_fade(cam, model, f, b1, b2):
    """0 near the camera .. 1 at 1.5 m, so the far wall is drawn darker (depth cue)"""
    p = model["xyz"][model["tri"][f]]
    q = (1 - b1 - b2)[:, None] * p[:, 0] + b1[:, None] * p[:, 1] + b2[:, None] * p[:, 2]
    return np.linalg.norm(q - cam["eye"], axis=1) / 1.5


def source_frame_reader(video, plain=False, crop=None, stretch_x=1.0):
    cap = cv2.VideoCapture(video)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sheffield = not plain and (w, h) == (720, 576)

    def read(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            return None
        # same geometry as extract_frames.py (no CLAHE: the frame as recorded)
        if sheffield:
            fr = cv2.resize(fr, (787, 576), interpolation=cv2.INTER_CUBIC)[:548, 16:784]
        elif plain:
            if crop:
                fr = fr[crop[1]:crop[3], crop[0]:crop[2]]
            if stretch_x != 1.0:
                fr = cv2.resize(fr, (int(round(fr.shape[1] * stretch_x)), fr.shape[0]), interpolation=cv2.INTER_CUBIC)
        return fr
    return read


def label(img, text, xy=(10, 8), fnt=F_BIG, fill=(255, 255, 0), outline=True):
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    x, y = xy
    for dx, dy in ((-1, -1), (1, 1), (-1, 1), (1, -1)) if outline else ():
        dr.text((x + dx, y + dy), text, font=fnt, fill=(0, 0, 0))
    dr.text(xy, text, font=fnt, fill=fill)
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("work"), ap.add_argument("out")
    ap.add_argument("--video"), ap.add_argument("--parts", nargs="+")
    ap.add_argument("--every", type=int, default=1), ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--plain", action="store_true"), ap.add_argument("--crop", nargs=4, type=int)
    ap.add_argument("--stretch-x", type=float, default=1.0), ap.add_argument("--straighten", action="store_true")
    a = ap.parse_args()
    straighten = bool(a.parts) or a.straighten
    model = rv.load_model(paths.CONVERTER, a.work, a.parts or [None], straighten=straighten)
    xyz = model["xyz"]
    half = rv.ev.PIPE_DIAMETER_M / 2
    r_dev = (np.hypot(xyz[:, 1], xyz[:, 2]) - half) * 1000.0           # mm; pipe coordinates are scaled to the bore
    frames = {r["file"]: int(r["source_frame"]) for r in csv.DictReader(open(os.path.join(a.work, "frames.csv"), encoding="utf-8"))}
    cpath = os.path.join(a.work, "counter.csv")
    counter = ({int(r["source_frame"]): float(r["counter_m"]) for r in csv.DictReader(open(cpath, encoding="utf-8"))}
               if os.path.exists(cpath) else {})
    read_src = source_frame_reader(a.video, a.plain, a.crop, a.stretch_x) if a.video else None
    order = np.argsort(model["cam_xyz"][:, 0])
    order = order[::a.every]
    scale = PW / model["intrinsics"]["width"]

    # side view, cut open (far half only), orthographic, whole model across the width
    p = xyz[np.hypot(xyz[:, 1], xyz[:, 2]) < 0.6]
    x0, x1 = np.percentile(p[:, 0], 0.2) - 0.05, np.percentile(p[:, 0], 99.8) + 0.05
    x0 = min(x0, model["cam_xyz"][:, 0].min() - 0.05)          # the whole camera track stays on the strip,
    x1 = max(x1, model["cam_xyz"][:, 0].max() + 0.05)          # also where nothing was reconstructed
    z0, z1 = -(half + 0.07), half + 0.07              # +-0.22 m for the 300 mm bore
    sw = 2 * PW
    ppm = sw / (x1 - x0)
    sh = int(round((z1 - z0) * ppm))
    if sh > 260:                        # short models: limit the strip height
        ppm = 260 / (z1 - z0)
        sh = 260
    centre = np.array([0.5 * (x0 + x1), 0.0, 0.0])
    side_cam = rv.camera(centre - [0, 50.0, 0], centre, (sw, sh), scale=ppm)
    side = rv.draw(model, side_cam, ~rv.toward_viewer(model, side_cam), light=0.25, gain=rv.GAIN)

    H = 2 * PH + sh + 40
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (2 * PW, H))
    print(f"{len(order)} frames -> {a.out} ({2 * PW}x{H}, {a.fps} fps)", flush=True)
    for n, k in enumerate(order):
        name = model["cam_file"][k]
        src = frames[name]
        cam, mx, my = camera_k(model, k, scale)
        tex = cv2.remap(rv.draw(model, cam, light=0.0, near=0.02), mx, my, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        dev = cv2.remap(deviation_image(model, cam, r_dev), mx, my, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        fr = read_src(src) if read_src else None
        if fr is None:
            fr = cv2.imdecode(np.fromfile(os.path.join(a.work, "images", name), np.uint8), cv2.IMREAD_COLOR)
        fr = cv2.resize(fr, (PW, PH), interpolation=cv2.INTER_AREA)[:, :, ::-1]
        tex = cv2.resize(tex, (PW, PH), interpolation=cv2.INTER_AREA)
        dev = cv2.resize(dev, (PW, PH), interpolation=cv2.INTER_AREA)
        x_cam = float(model["cam_xyz"][k, 0])
        cnt = counter.get(src)
        p1 = label(fr, f"(1) 原始视频帧  #{src}")
        p2 = label(tex, "(2) 重建模型（纹理网格），同一相机位姿")
        p3 = label(dev, f"(3) 管壁偏差：蓝=向内 红=向外（±{DEV_MM:.0f} mm）")
        info = np.full((PH, PW, 3), 255, np.uint8)
        lines = [f"相机位置（模型，沿管轴）：{x_cam:6.2f} m"]
        if cnt is not None:
            lines += [f"画面计数器读数（真值）：  {cnt:6.2f} m", f"差值：{(x_cam - cnt) * 100:+.1f} cm"]
        lines += ["", f"注册相机 {n * a.every + 1} / {len(model['cam_xyz'])}",
                  f"尺度：{'逐段' if straighten else ''}按 {rv.ev.PIPE_DIAMETER_M * 1000:.0f} mm 管径拟合（稀疏模型）",
                  "灰色 = 没有任何图像覆盖的面"]
        im = Image.fromarray(info)
        dr = ImageDraw.Draw(im)
        dr.text((14, 10), "(4) 当前位置", font=F_BIG, fill=(0, 0, 0))
        for i, t in enumerate(lines):
            dr.text((14, 50 + 28 * i), t, font=F_SMALL if i >= 3 else F_BIG, fill=(30, 30, 30))
        p4 = np.asarray(im)
        strip = side.copy()
        if strip.shape[1] != sw:
            strip = cv2.resize(strip, (sw, sh))
        xs = int(round((x_cam - x0) * ppm))
        cv2.line(strip, (xs, 0), (xs, sh - 1), (0, 160, 0), 2)
        if cnt is not None:
            xc = int(round((cnt - x0) * ppm))
            cv2.line(strip, (xc, 0), (xc, sh - 1), (220, 0, 0), 1)
        bar = np.full((40, sw, 3), 255, np.uint8)
        bar = label(bar, f"侧视剖面（远侧半管内壁）  {x0:.2f}–{x1:.2f} m   绿线=模型中的相机位置  红线=计数器读数",
                    (10, 8), F_SMALL, (0, 0, 0), outline=False)
        frame = np.vstack([np.hstack([p1, p2]), np.hstack([p3, p4]), bar, strip])
        vw.write(np.ascontiguousarray(frame[:, :, ::-1]))
        if n % 20 == 0:
            print(f"  frame {n + 1}/{len(order)}: source {src}, x {x_cam:.2f} m", flush=True)
            if n == 0:
                cv2.imencode(".jpg", frame[:, :, ::-1])[1].tofile(os.path.splitext(a.out)[0] + "_first.jpg")
        if n == len(order) // 2:
            cv2.imencode(".jpg", frame[:, :, ::-1])[1].tofile(os.path.splitext(a.out)[0] + "_middle.jpg")
    vw.release()
    print("saved", a.out)


if __name__ == "__main__":
    main()
