"""Paper-style 3-D pictures of a textured reconstruction (what Fang 2022 Fig. 20-23 and Liu 2024 Fig. 21-28 show).

Usage: render_views.py CONVERTER_EXE WORK_DIR OUT_DIR [--parts A B ...] [--inside X ...] [--marks X=LABEL ...]
                       [--gaps A-B ...] [--title TEXT] [--straighten]
       --parts   separately reconstructed stretches (run_pipeline.py --part); without it: WORK_DIR/mvs
       --straighten  straighten a single reconstruction slice by slice too (centre and scale from the bore, as for
                 parts): for long real sewers that bend, where one straight cylinder does not fit
       --inside  positions along the pipe [m] for the views from inside (default: three, spread over the run)
       --marks   positions or ranges along the pipe [m] to label on the side elevation, e.g. 2.11=joint 4.63-5.69=manhole
       --gaps    ranges with no model, drawn grey on the side elevation, e.g. 2.7-3.0

Reads the OpenMVS textured mesh (scene_textured.obj and its texture atlas), puts it into pipe coordinates
(evaluate.pipe_frame) and draws it with a small software rasteriser - numpy and OpenCV only, nothing to install:
  view_whole.png      the whole model seen from outside, perspective
  view_cutaway.png    the same with the half towards the viewer removed, so that the inner wall shows
  view_side.png       side elevation of the outside and of the cut-open model, orthographic, to scale
  view_inside_N.png   looking along the pipe from the position of a registered camera
  model_views.png     all of them on one sheet, the inside views next to their video frames
A single reconstruction is only moved, turned and scaled (one scale from the 300 mm bore): the pictures show
the geometry as reconstructed. Parts have no common frame, so each is straightened and placed by the counter,
exactly as merge_parts.py does for the dense clouds.
Frame of the pictures: x along the pipe in the direction of travel, y to the left, z up, metres.
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as ev  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
SS = 2                    # the pictures are drawn at SS times their final size and averaged down
LIGHT = 0.45              # share of the brightness that depends on the direction of the surface (outside views)
BLANK_RGB = (178, 178, 178)  # faces that no image covers (TextureMesh paints them orange)
GAIN = 1.3                # brightening of the views from outside; the views from inside keep the video's brightness


# ------------------------------------------------------------------------------- model ----------
def read_obj(path):
    """An OpenMVS textured OBJ (all "v", then all "vt", then all "f a/ta b/tb c/tc", one material):
    vertices (n,3), triangles (m,3) and the texture coordinates of the three corners of each triangle (m,3,2)."""
    data = open(path, "rb").read()
    i_v, i_vt, i_f = data.find(b"\nv "), data.find(b"\nvt "), data.find(b"\nf ")
    if not 0 <= i_v < i_vt < i_f or data.count(b"\nusemtl ") != 1:
        raise ValueError("unexpected OBJ layout in " + path)
    v = np.fromstring(data[i_v:i_vt].replace(b"\nv ", b" "), dtype=np.float64, sep=" ").reshape(-1, 3)
    vt = np.fromstring(data[i_vt:data.find(b"\nusemtl ", i_vt)].replace(b"\nvt ", b" "), dtype=np.float64, sep=" ")
    f = np.fromstring(data[i_f:].replace(b"\nf ", b" ").replace(b"/", b" "), dtype=np.int64, sep=" ").reshape(-1, 6)
    return v, (f[:, 0::2] - 1).astype(np.int32), vt.reshape(-1, 2)[f[:, 1::2] - 1].astype(np.float32)


def load_model(converter, work, parts, straighten):
    """Textured mesh(es) of a work directory in pipe coordinates, plus the registered cameras."""
    frames = {r["file"]: int(r["source_frame"]) for r in csv.DictReader(open(os.path.join(work, "frames.csv"), encoding="utf-8"))}
    counter_csv = os.path.join(work, "counter.csv")
    if os.path.exists(counter_csv):
        counter = np.array([float(r["counter_m"]) for r in csv.DictReader(open(counter_csv, encoding="utf-8"))])
    else:   # a video without on-screen counter: positions are then measured from the first camera
        counter = np.zeros(max(frames.values()) + 1)
    file_of = {v: k for k, v in frames.items()}
    xyz, tri, uv, tid, tex, cams, blank = [], [], [], [], [], [], []
    n = 0
    for part in parts:
        f = ev.pipe_frame(converter, work, part, frames, counter, straighten=straighten)
        mvs = os.path.join(work, f"mvs_{part}" if part else "mvs")
        v, t, c = read_obj(os.path.join(mvs, "scene_textured.obj"))
        x, th, r = f["to_pipe"](v)
        keep = (r[t] < 1.0).all(axis=1)                       # drop stray pieces more than 1 m from the axis
        xyz.append(np.c_[pipe_xyz(x, th, r)])
        tri.append(t[keep] + n), uv.append(c[keep]), tid.append(np.full(keep.sum(), len(tex), np.int8))
        n += len(v)
        tex.append(cv2.imdecode(np.fromfile(os.path.join(mvs, "scene_textured_material_00_map_Kd.jpg"), np.uint8), cv2.IMREAD_COLOR))
        blank.append(np.ptp(c[keep], axis=1).max(axis=1) < 1e-7)   # no image: all three corners on one texel
        cams += [(*pipe_xyz(*(q[k] for q in (f["cam_x"], f["cam_theta"], f["cam_r"]))), *pipe_xyz(*(q[k] for q in f["cam_ahead"])),
                  *pipe_xyz(*(q[k] for q in f["cam_above"])), file_of[int(f["src"][k])]) for k in range(len(f["src"]))]
        print(f"part {part or '-'}: {len(v):,} vertices, {int(keep.sum()):,} of {len(t):,} triangles kept, texture "
              f"{tex[-1].shape[1]}x{tex[-1].shape[0]}, {blank[-1].mean():.1%} of the triangles not covered by any image, "
              f"{len(f['src'])} cameras, x {f['cam_x'][0]:.2f}..{f['cam_x'][-1]:.2f} m")
    xyz, tri, uv, tid = np.concatenate(xyz), np.concatenate(tri), np.concatenate(uv), np.concatenate(tid)
    fn = np.cross(xyz[tri[:, 1]] - xyz[tri[:, 0]], xyz[tri[:, 2]] - xyz[tri[:, 0]])     # area-weighted face normals
    vn = np.stack([np.bincount(tri.ravel(), np.repeat(fn[:, k], 3), len(xyz)) for k in range(3)], axis=1)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-30
    return dict(xyz=xyz, tri=tri, uv=uv, tid=tid, tex=tex, blank=np.concatenate(blank), vn=vn.astype(np.float32),
                centroid=xyz[tri].mean(axis=1), cam_xyz=np.array([c[0:3] for c in cams]), cam_ahead=np.array([c[3:6] for c in cams]),
                cam_above=np.array([c[6:9] for c in cams]), cam_file=[c[9] for c in cams], intrinsics=f["intrinsics"])


def pipe_xyz(x, theta, r):
    """Pipe coordinates (axial position, angle from up towards the right in degrees, radius) to x, y (left), z (up)."""
    t = np.radians(theta)
    return x, -r * np.sin(t), r * np.cos(t)


# ---------------------------------------------------------------------------- rasteriser ----------
def camera(eye, target, size, fov=None, scale=None, up=(0.0, 0.0, 1.0), focal=None):
    """Camera looking from eye to target. fov: horizontal field of view in degrees, or focal: focal length in
    pixels of the final picture (perspective); scale: pixels per metre (orthographic). size: (width, height)
    of the final picture; the viewing direction goes through its centre."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    f = (target - eye) / np.linalg.norm(target - eye)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    cam = dict(eye=eye, R=np.array([r, np.cross(r, f), f]), size=(size[0] * SS, size[1] * SS))
    if focal is not None:
        cam["focal"] = focal * SS
    elif fov is not None:
        cam["focal"] = 0.5 * cam["size"][0] / np.tan(np.radians(fov) / 2)
    else:
        cam["scale"] = scale * SS
    return cam


def project(cam, p):
    """Picture coordinates (in pixels of the oversized drawing) and depth of world points."""
    c = (np.asarray(p, float) - cam["eye"]) @ cam["R"].T
    k = cam["focal"] / np.maximum(c[:, 2], 1e-9) if "focal" in cam else cam["scale"]
    return cam["size"][0] / 2 + k * c[:, 0], cam["size"][1] / 2 - k * c[:, 1], c[:, 2]


def rasterize(cam, xyz, tri, near=0.03):
    """Z-buffer drawing of triangles. Returns, per pixel: depth, index into tri (-1 = empty) and the weights of
    the second and third corner (perspective-correct). Triangles are handled together in groups of similar
    size on the picture: every pixel of their bounding box is tested against the three edges."""
    w_pic, h_pic = cam["size"]
    sx, sy, depth = project(cam, xyz)
    persp = "focal" in cam
    t_ok = (depth[tri] > near).all(axis=1) if persp else np.ones(len(tri), bool)
    x, y = sx[tri], sy[tri]
    xlo = np.maximum(np.ceil(x.min(axis=1) - 0.5), 0).astype(np.int64)        # pixel centres are at +0.5
    xhi = np.minimum(np.floor(x.max(axis=1) - 0.5), w_pic - 1).astype(np.int64)
    ylo = np.maximum(np.ceil(y.min(axis=1) - 0.5), 0).astype(np.int64)
    yhi = np.minimum(np.floor(y.max(axis=1) - 0.5), h_pic - 1).astype(np.int64)
    e1x, e1y, e2x, e2y = x[:, 1] - x[:, 0], y[:, 1] - y[:, 0], x[:, 2] - x[:, 0], y[:, 2] - y[:, 0]
    area2 = e1x * e2y - e1y * e2x
    t_ok &= (xlo <= xhi) & (ylo <= yhi) & (np.abs(area2) > 1e-12)
    size = np.maximum(xhi - xlo, yhi - ylo) + 1
    wgt = 1.0 / np.maximum(depth, 1e-9) if persp else depth                    # what is linear across the picture
    zbuf = np.full(w_pic * h_pic, np.inf, np.float32)
    fbuf = np.full(w_pic * h_pic, -1, np.int32)
    b1buf, b2buf = np.zeros(w_pic * h_pic, np.float32), np.zeros(w_pic * h_pic, np.float32)
    lo = 0
    for box in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 1024, 2048, 4096, 1 << 30):
        ids = np.flatnonzero(t_ok & (size > lo) & (size <= box))
        lo = box
        if not len(ids):
            continue
        box = int(size[ids].max())
        gy, gx = (g.ravel() for g in np.mgrid[0:box, 0:box])
        step = max(1, 2000000 // (box * box))
        for a in range(0, len(ids), step):
            i = ids[a:a + step]
            px, py = xlo[i, None] + gx, ylo[i, None] + gy
            fx, fy = px + 0.5 - x[i, 0, None], py + 0.5 - y[i, 0, None]
            l1 = (fx * e2y[i, None] - e2x[i, None] * fy) / area2[i, None]
            l2 = (e1x[i, None] * fy - fx * e1y[i, None]) / area2[i, None]
            m = (px <= xhi[i, None]) & (py <= yhi[i, None]) & (l1 >= -1e-6) & (l2 >= -1e-6) & (l1 + l2 <= 1 + 1e-6)
            row = np.nonzero(m)[0]
            l1, l2, pix, face = l1[m], l2[m], (py[m] * w_pic + px[m]), i[row]
            w0, w1, w2 = wgt[tri[face, 0]], wgt[tri[face, 1]], wgt[tri[face, 2]]
            if persp:
                iw = (1 - l1 - l2) * w0 + l1 * w1 + l2 * w2
                z, l1, l2 = (1.0 / iw).astype(np.float32), l1 * w1 / iw, l2 * w2 / iw
            else:
                z = ((1 - l1 - l2) * w0 + l1 * w1 + l2 * w2).astype(np.float32)
            np.minimum.at(zbuf, pix, z)
            hit = z <= zbuf[pix]
            fbuf[pix[hit]], b1buf[pix[hit]], b2buf[pix[hit]] = face[hit], l1[hit], l2[hit]
    return tuple(b.reshape(h_pic, w_pic) for b in (zbuf, fbuf, b1buf, b2buf))


def draw(model, cam, faces=None, light=LIGHT, near=0.03, gain=1.0):
    """Picture (RGB, white background) of the model, or of the triangles selected by the mask `faces`;
    gain brightens the texture (the pipe wall is dark grey in the video)."""
    sel = np.arange(len(model["tri"])) if faces is None else np.flatnonzero(faces)
    tri = model["tri"][sel]
    _, fbuf, b1, b2 = rasterize(cam, model["xyz"], tri, near)
    hit = fbuf >= 0
    f, b1, b2 = sel[fbuf[hit]], b1[hit][:, None], b2[hit][:, None]
    uv = (1 - b1 - b2) * model["uv"][f, 0] + b1 * model["uv"][f, 1] + b2 * model["uv"][f, 2]
    rgb = np.zeros((len(f), 3), np.float32)
    for t, tex in enumerate(model["tex"]):                    # colour from the texture atlas of each part
        of_t = model["tid"][f] == t
        mx, my = np.full(hit.shape, -1, np.float32), np.full(hit.shape, -1, np.float32)
        where = np.flatnonzero(hit)[of_t]
        mx.flat[where] = uv[of_t, 0] * tex.shape[1] - 0.5
        my.flat[where] = (1 - uv[of_t, 1]) * tex.shape[0] - 0.5
        rgb[of_t] = cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).reshape(-1, 3)[where][:, ::-1]
    rgb[model["blank"][f]] = BLANK_RGB
    if light > 0:                                             # light from behind the viewer, upper left; both sides lit
        v = model["tri"][f]
        n = (1 - b1 - b2) * model["vn"][v[:, 0]] + b1 * model["vn"][v[:, 1]] + b2 * model["vn"][v[:, 2]]
        to_light = cam["R"].T @ np.array([-0.45, 0.55, -0.70])
        lam = np.abs(n @ to_light) / (np.linalg.norm(n, axis=1) * np.linalg.norm(to_light) + 1e-30)
        rgb *= ((1 - light) + light * (0.25 + 1.1 * lam))[:, None]
    rgb *= gain
    out = np.full(hit.shape + (3,), 255, np.uint8)
    out[hit] = np.clip(rgb, 0, 255)
    return cv2.resize(out, (hit.shape[1] // SS, hit.shape[0] // SS), interpolation=cv2.INTER_AREA)


def crop(img, pad=30):
    """Cut away the white margin, leaving pad pixels."""
    ys, xs = np.nonzero((img < 250).any(axis=2))
    if not len(ys):
        return img
    return img[max(ys.min() - pad, 0):ys.max() + pad + 1, max(xs.min() - pad, 0):xs.max() + pad + 1]


def save(path, img):
    cv2.imencode(".png", img[:, :, ::-1])[1].tofile(path)     # cv2.imwrite cannot write to non-ASCII paths
    print("saved", path, f"{img.shape[1]}x{img.shape[0]}")


# -------------------------------------------------------------------------------- views ----------
def outside_camera(model, azimuth, elevation, size, fov=30.0):
    """Perspective camera beside and above the start of the pipe (on its right), far enough to hold the model.
    azimuth: angle between the viewing direction and the pipe, elevation: above the horizontal, degrees."""
    p = model["xyz"][::97]
    p = p[np.hypot(p[:, 1], p[:, 2]) < 0.6]
    target = np.array([0.5 * (p[:, 0].min() + p[:, 0].max()), 0.0, 0.0])
    az, el = np.radians(azimuth), np.radians(elevation)
    back = np.array([-np.cos(el) * np.cos(az), -np.cos(el) * np.sin(az), np.sin(el)])
    lo, hi = 0.2, 200.0
    for _ in range(40):                                       # nearest distance at which everything is in the picture
        cam = camera(target + 0.5 * (lo + hi) * back, target, size, fov=fov)
        sx, sy, z = project(cam, p)
        inside = (z > 0.05).all() and (np.abs(sx / cam["size"][0] - 0.5).max() < 0.47) and (np.abs(sy / cam["size"][1] - 0.5).max() < 0.47)
        lo, hi = (lo, 0.5 * (lo + hi)) if inside else (0.5 * (lo + hi), hi)
    return camera(target + hi * back, target, size, fov=fov)


def toward_viewer(model, cam):
    """Triangles of the half of the pipe that faces the camera (split by the plane through the pipe axis)."""
    side = cam["eye"] - np.array([cam["eye"][0], 0.0, 0.0])
    return model["centroid"] @ (side / np.linalg.norm(side)) > 0


def side_rows(model, row_m, px_per_m):
    """Side elevations seen from the right of the pipe, x increasing to the right, cut into rows of row_m metres:
    list of (x0, x1, z0, z1, picture of the outside, picture with the near half removed)."""
    p = model["xyz"][np.hypot(model["xyz"][:, 1], model["xyz"][:, 2]) < 0.6]
    x0, x1 = np.percentile(p[:, 0], 0.05) - 0.05, np.percentile(p[:, 0], 99.95) + 0.05
    z0, z1 = np.percentile(p[:, 2], 0.05) - 0.05, np.percentile(p[:, 2], 99.95) + 0.05
    n = int(np.ceil((x1 - x0) / row_m - 1e-9))
    rows = []
    for k in range(n):
        a, b = x0 + k * (x1 - x0) / n, x0 + (k + 1) * (x1 - x0) / n
        centre = np.array([0.5 * (a + b), 0.0, 0.5 * (z0 + z1)])
        cam = camera(centre - [0, 50.0, 0], centre, (int(round((b - a) * px_per_m)), int(round((z1 - z0) * px_per_m))), scale=px_per_m)
        rows.append((a, b, z0, z1, draw(model, cam, gain=GAIN), draw(model, cam, ~toward_viewer(model, cam), light=0.25, gain=GAIN)))
    return rows


def undistort_grid(xd, yd, k1):
    """Undistorted normalised coordinates of distorted ones under OpenMVG's radial-k1 model x_d = x_u (1 + k1 r_u^2),
    inverted by iteration. With k1 < 0 the model reaches distorted radii only up to 2 / (3 sqrt(-3 k1)) (0.67 for the
    Sheffield camera, whose frame corners are at 0.60); pixels beyond 95 % of that - the corners of a wide-angle
    frame, e.g. the Pipe1 sewer camera - have no undistorted position: NaN, left blank in the pictures."""
    out = (xd ** 2 + yd ** 2 >= (0.95 * 2 / (3 * np.sqrt(-3 * k1))) ** 2) if k1 < 0 else np.zeros(xd.shape, bool)
    xd, yd = np.where(out, 0.0, xd), np.where(out, 0.0, yd)
    xu, yu = xd.copy(), yd.copy()
    for _ in range(20):
        g = 1 + k1 * (xu ** 2 + yu ** 2)
        xu, yu = xd / g, yd / g
    xu[out], yu[out] = np.nan, np.nan
    return xu, yu


def inside_view(model, x, scale=1.5):
    """The model seen by the registered camera nearest to position x: same position, viewing direction, focal
    length and lens distortion as the video frame, so the two can be laid side by side.
    Returns (picture, camera position, frame file); the picture is the frame size times scale."""
    k = int(np.argmin(np.abs(model["cam_xyz"][:, 0] - x)))
    eye, ahead, above = model["cam_xyz"][k], model["cam_ahead"][k], model["cam_above"][k]
    intr = model["intrinsics"]
    f, (cx, cy), k1 = intr["focal_length"], intr["principal_point"], intr.get("disto_k1", [0.0])[0]
    w, h = int(round(intr["width"] * scale)), int(round(intr["height"] * scale))
    # normalised undistorted coordinates of every output pixel (OpenMVG radial-k1 model, inverted by iteration)
    xd = ((np.arange(w) + 0.5) / scale - 0.5 - cx) / f
    yd = ((np.arange(h) + 0.5) / scale - 0.5 - cy) / f
    xu, yu = undistort_grid(*np.meshgrid(xd, yd), k1)
    focal = f * scale
    cw, ch = 2 * int(np.ceil(np.nanmax(np.abs(xu)) * focal)) + 4, 2 * int(np.ceil(np.nanmax(np.abs(yu)) * focal)) + 4
    pinhole = draw(model, camera(eye, ahead, (cw, ch), focal=focal, up=above - eye), light=0.0, near=0.02)
    img = cv2.remap(pinhole, np.nan_to_num(cw / 2 + xu * focal - 0.5, nan=-10).astype(np.float32),
                    np.nan_to_num(ch / 2 + yu * focal - 0.5, nan=-10).astype(np.float32),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return img, eye, model["cam_file"][k]


def parse_mark(text):
    """'2.11=joint' or '4.63-5.69=manhole' -> (start, end, label)."""
    where, label = text.split("=", 1)
    a, _, b = where.partition("-")
    return float(a), float(b or a), label


def mark_strip(ax, x0, x1, z0, z1, marks, gaps):
    """Labels on a side elevation: a dashed red line for one position, dashed lines joined by a bracket for a
    range, a grey band for a stretch with no model."""
    for a, b in gaps:
        if b >= x0 and a <= x1:
            ax.axvspan(a, b, color="0.85", lw=0, zorder=0)
            ax.text(0.5 * (a + b), 0.5 * (z0 + z1), "未重建", color="0.35", fontsize=9, ha="center", va="center")
    top = z1 - 0.06 * (z1 - z0)
    for a, b, label in marks:
        if b < x0 or a > x1:
            continue
        for x in {a, b}:
            ax.axvline(x, color="r", lw=0.9, ls="--")
        if b > a:
            ax.plot([max(a, x0), min(b, x1)], [top, top], "r-", lw=1.2)
        ax.text(min(max(0.5 * (a + b), x0), x1), top, label, color="r", fontsize=9, ha="center", va="center",
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.set_xlim(x0, x1)
    ax.set_ylim(z0, z1)


def sheet(path, title, whole, cut, rows, inside, frames, marks, gaps):
    """Everything on one page, laid out in inches so that every picture keeps its proportions."""
    import matplotlib.pyplot as plt
    fig_w, left, right, cap, tick = 18.0, 0.7, 0.3, 0.38, 0.45
    width = fig_w - left - right
    items, y = [], 0.25 + (0.45 if title else 0)              # (kind, payload, top, height), inches from the top
    for img, name in ((whole, "(a) 带纹理三维模型，从管外看（透视）"), (cut, "(b) 去掉靠近观察者的一半，露出内壁（透视）")):
        h = min(width * img.shape[0] / img.shape[1], 6.0)
        items.append(("pic", (img, name), y + cap, h))
        y += cap + h + 0.15
    for k, (x0, x1, z0, z1, img_out, img_cut) in enumerate(rows):
        h = width * (z1 - z0) / (x1 - x0)
        for img, name in ((img_out, "外观"), (img_cut, "剖开，看到的是远侧一半的内壁")):
            label = f"(c) 侧视图，正投影、按比例，{x0:.1f}–{x1:.1f} m：{name}"
            if k == 0 and name == "外观":
                label += "（横轴：沿管道位置 [m]，计数器刻度）"
            items.append(("strip", (img, label, x0, x1, z0, z1), y + cap, h))
            y += cap + h + tick
    n, gap = len(inside), 0.25
    col_w = (width - gap * (n - 1)) / n
    h = col_w * inside[0][0].shape[0] / inside[0][0].shape[1]
    for r, label in enumerate(("(d) 模型内部：与下方视频帧相同的相机位置、朝向、焦距和镜头畸变",
                               "(e) 同一位置的视频帧（抽帧时做过 CLAHE 对比度增强）")):
        items.append(("row", (r, label), y + cap, h))
        y += cap + h + 0.3
    fig_h = y
    fig = plt.figure(figsize=(fig_w, fig_h))
    if title:
        fig.text(0.5, 1 - 0.35 / fig_h, title, ha="center", va="center", fontsize=15)
    for kind, payload, top, h in items:
        if kind == "pic":
            img, name = payload
            w = h * img.shape[1] / img.shape[0]
            ax = fig.add_axes([(left + (width - w) / 2) / fig_w, 1 - (top + h) / fig_h, w / fig_w, h / fig_h])
            ax.imshow(img)
            ax.axis("off")
            fig.text(left / fig_w, 1 - (top - 0.08) / fig_h, name, fontsize=12, va="bottom")
        elif kind == "strip":
            img, label, x0, x1, z0, z1 = payload
            ax = fig.add_axes([left / fig_w, 1 - (top + h) / fig_h, width / fig_w, h / fig_h])
            ax.imshow(img, extent=(x0, x1, z0, z1), aspect="auto")
            ax.set_ylabel("[m]", fontsize=9)
            ax.tick_params(labelsize=9)
            mark_strip(ax, x0, x1, z0, z1, marks, gaps)
            fig.text(left / fig_w, 1 - (top - 0.08) / fig_h, label, fontsize=11, va="bottom")
        else:
            r, label = payload
            fig.text(left / fig_w, 1 - (top - 0.08) / fig_h, label, fontsize=12, va="bottom")
            for k, (img, eye, name) in enumerate(inside):
                ax = fig.add_axes([(left + k * (col_w + gap)) / fig_w, 1 - (top + h) / fig_h, col_w / fig_w, h / fig_h])
                ax.imshow(img if r == 0 else frames[k])
                ax.set_xticks([]), ax.set_yticks([])
                ax.set_xlabel(f"相机在 {eye[0]:.2f} m" if r == 0 else name, fontsize=10, labelpad=2)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("saved", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("converter"), ap.add_argument("work"), ap.add_argument("out")
    ap.add_argument("--parts", nargs="+"), ap.add_argument("--inside", nargs="+", type=float)
    ap.add_argument("--marks", nargs="+", default=[]), ap.add_argument("--gaps", nargs="+", default=[])
    ap.add_argument("--title", default=""), ap.add_argument("--straighten", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model = load_model(a.converter, a.work, a.parts or [None], straighten=bool(a.parts) or a.straighten)
    length = float(np.ptp(model["cam_xyz"][:, 0]))
    marks = [parse_mark(m) for m in a.marks]
    gaps = [parse_mark(g + "=")[:2] for g in a.gaps]

    cam = outside_camera(model, 36, 13, (3400, 1500))
    whole = crop(draw(model, cam, gain=GAIN))
    save(os.path.join(a.out, "view_whole.png"), whole)
    cam = outside_camera(model, 55, 32, (3400, 1500))
    cut = crop(draw(model, cam, ~toward_viewer(model, cam), light=0.25, gain=GAIN))
    save(os.path.join(a.out, "view_cutaway.png"), cut)
    rows = side_rows(model, 5.2, 640)
    inside_x = a.inside or list(model["cam_xyz"][0, 0] + length * np.array([0.15, 0.45, 0.75]))
    inside = [inside_view(model, x) for x in inside_x]
    frames = []
    for k, (img, _, name) in enumerate(inside):
        save(os.path.join(a.out, f"view_inside_{k + 1}.png"), img)
        fr = cv2.imdecode(np.fromfile(os.path.join(a.work, "images", name), np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
        frames.append(cv2.resize(fr, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2 * len(rows), 1, figsize=(18, 2.9 * len(rows) + 0.9), squeeze=False)
    for k, (x0, x1, z0, z1, img_out, img_cut) in enumerate(rows):
        for ax, img, name in ((axes[2 * k, 0], img_out, "外观"), (axes[2 * k + 1, 0], img_cut, "剖开（去掉靠近观察者的一半，看到的是内壁）")):
            ax.imshow(img, extent=(x0, x1, z0, z1))
            ax.set_ylabel("高度 [m]")
            ax.set_title(f"侧视图 {x0:.1f}–{x1:.1f} m：{name}", fontsize=10, loc="left")
            mark_strip(ax, x0, x1, z0, z1, marks, gaps)
        axes[2 * k + 1, 0].set_xlabel("沿管道位置 [m]（计数器刻度）")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "view_side.png"), dpi=130)
    plt.close(fig)
    print("saved", os.path.join(a.out, "view_side.png"))
    sheet(os.path.join(a.out, "model_views.png"), a.title, whole, cut, rows, inside, frames, marks, gaps)


if __name__ == "__main__":
    main()
