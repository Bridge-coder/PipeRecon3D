"""Side-by-side check of the reconstructions against the video: for every SfM frame, the original image next to
the same view rendered from (a) the OpenMVS textured mesh and (b) the fused LiDAR wall, plus a cross-section
where the two geometries can be compared in millimetres. Also one sheet of outside / cut-away 3-D views.

Frame of everything: the fused metric world frame of f5_fuse.py (vision axes, AprilTag scale; the LiDAR PLY is
already in it). Vision mesh and vision dense cloud are scaled by the tag scale; each is drawn from its own camera
centre (vision: SfM centre x scale, LiDAR: fused centre), so neither is shifted to fit the other.

usage: python -I compare_views.py EXTRACT_DIR VISION_WORK RESULTS_DIR OUT_DIR [--every N] [--sheet-only]
"""
import csv
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "fusion"))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "vision"))
import render_views as rv  # noqa: E402
from paths import FUSED_BIG  # noqa: E402
from ply_io import read_ply_vertices  # noqa: E402
from timebase import load_offsets  # noqa: E402
from vision_model import load_cameras  # noqa: E402

args = sys.argv[1:]
EVERY = 1
if "--every" in args:
    i = args.index("--every")
    EVERY = int(args[i + 1])
    del args[i:i + 2]
SHEET_ONLY = "--sheet-only" in args
if SHEET_ONLY:
    args.remove("--sheet-only")
EX, VW, RES, OUT = args
os.makedirs(OUT, exist_ok=True)
FONT = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 19)
FONT_S = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 15)
PW, PH = 640, 360                                    # one panel
AHEAD = 0.30                                         # cross-section this far ahead of the camera [m]
SLAB = 0.006                                         # half thickness of the cross-section slab [m]
DEV = 0.010                                          # colour range of the LiDAR radius residual [m]

FU = os.path.join(RES, "fusion")
off = load_offsets(os.path.join(FU, "time_offsets.json"))
lam = json.load(open(os.path.join(FU, "f4_tag_scale.json")))["mm_per_unit_tag"] / 1000
legs = json.load(open(os.path.join(FU, "vision_legs.json")))
a1, a2 = np.array(legs["legs"][0]["dir"]), np.array(legs["legs"][1]["dir"])
B = np.array(legs["bend_point"]) * lam
ci = json.load(open(os.path.join(EX, "bag_info.json")))["camera_info"]["color"]
K = np.array(ci["k"]).reshape(3, 3)
W, H = ci["width"], ci["height"]
cam = load_cameras(EX, VW, off)
fused = {int(r["color_idx"]): np.array([float(r["x_m"]), float(r["y_m"]), float(r["z_m"])])
         for r in csv.DictReader(open(os.path.join(FU, "fused_trajectory.csv")))}


def centerline(P):
    """arc position u [m] (0 at the bend), nearest centerline point, local axis direction."""
    u1, u2 = (P - B) @ a1, (P - B) @ a2
    on2 = u2 > 0
    Q = np.where(on2[:, None], B + np.outer(u2, a2), B + np.outer(np.minimum(u1, 0), a1))
    d = np.where(on2[:, None], a2, a1)
    return np.where(on2, u2, np.minimum(u1, 0)), Q, d


# ------------------------------------------------------------------ data ----------
print("loading the textured mesh ...", flush=True)
mvs = os.path.join(VW, "mvs")
v, tri, uv = rv.read_obj(os.path.join(mvs, "scene_textured.obj"))
xyz = v * lam
tex = cv2.imdecode(np.fromfile(os.path.join(mvs, "scene_textured_material_00_map_Kd.jpg"), np.uint8), cv2.IMREAD_COLOR)
fn = np.cross(xyz[tri[:, 1]] - xyz[tri[:, 0]], xyz[tri[:, 2]] - xyz[tri[:, 0]])
vn = np.stack([np.bincount(tri.ravel(), np.repeat(fn[:, k], 3), len(xyz)) for k in range(3)], axis=1)
vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-30
mesh = dict(xyz=xyz, tri=tri, uv=uv, tid=np.zeros(len(tri), np.int8), tex=[tex],
            blank=np.ptp(uv, axis=1).max(axis=1) < 1e-7, vn=vn.astype(np.float32), centroid=xyz[tri].mean(axis=1))
print(f"mesh: {len(xyz):,} vertices, {len(tri):,} triangles, {mesh['blank'].mean():.1%} not covered by any image")

print("loading the LiDAR wall and the vision dense cloud ...", flush=True)
with open(os.path.join(FUSED_BIG, "lidar_coloured_world.ply"), "rb") as f:
    while f.readline().strip() != b"end_header":
        pass
    rec = np.frombuffer(f.read(), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
L = np.column_stack([rec["x"], rec["y"], rec["z"]]).astype(np.float64)
Lrgb = np.column_stack([rec["r"], rec["g"], rec["b"]])
uL, QL, dL = centerline(L)
qL = L - QL
rL = np.linalg.norm(qL - (qL * dL).sum(1, keepdims=True) * dL, axis=1)
legL = uL > 0
# residual after removing each leg's mean cross-section shape (ellipse + offset from the vision centerline):
# median radius per 4-degree angle bin and leg, angle measured from the cameras' mean "up"
UPW = -np.mean(cam["R"][:, 1], axis=0)
thL = np.zeros(len(L))
for leg, ax_ in ((~legL, a1), (legL, a2)):
    up_ = UPW - (UPW @ ax_) * ax_
    up_ /= np.linalg.norm(up_)
    side_ = np.cross(ax_, up_)
    thL[leg] = np.degrees(np.arctan2(qL[leg] @ side_, qL[leg] @ up_))
binL = ((thL + 180) // 4).astype(int) + 90 * legL
med = np.array([np.median(rL[binL == b]) if (binL == b).any() else 0.0 for b in range(180)])
devL = rL - med[binL]
# single LiDAR points scatter by about +-1 cm, and a z-buffer always shows the extreme ones (the outermost from
# outside, the innermost from inside); so every point is drawn with the mean residual of its 1 cm x 4 deg cell
cell = np.unique(((uL + 3.0) / 0.01).astype(np.int64) * 200 + binL, return_inverse=True)[1].ravel()
ok_ = np.abs(devL) < 0.05
devL_cell = (np.bincount(cell, np.where(ok_, devL, 0)) / np.maximum(np.bincount(cell, ok_), 1))[cell]
print(f"single-point scatter about the cell mean: rms {np.sqrt(np.mean((devL - devL_cell)[ok_] ** 2)) * 1000:.1f} mm")
devL = devL_cell
print(f"LiDAR residual after removing the mean cross-section of each leg: rms {np.sqrt(np.mean(devL[np.abs(devL) < 0.05] ** 2)) * 1000:.1f} mm, "
      f"cross-section radius range leg 1 {(med[:90].max() - med[:90].min()) * 1000:.0f} mm, leg 2 {(med[90:].max() - med[90:].min()) * 1000:.0f} mm")
print(f"LiDAR: {len(L):,} points, median radius leg 1 {np.median(rL[~legL]) * 1000:.1f} mm, leg 2 {np.median(rL[legL]) * 1000:.1f} mm "
      f"(about the vision centerline)")
Vd = read_ply_vertices(os.path.join(mvs, "scene_dense.ply"))
V = np.column_stack([Vd["x"], Vd["y"], Vd["z"]]).astype(np.float64) * lam
uV, QV, dV = centerline(V)
qV = V - QV
rV = np.linalg.norm(qV - (qV * dV).sum(1, keepdims=True) * dV, axis=1)
wall = rV < 0.16
print(f"vision dense: {len(V):,} points ({wall.mean():.0%} within 160 mm of the centerline), median radius "
      f"leg 1 {np.median(rV[wall & (uV < 0)]) * 1000:.1f} mm, leg 2 {np.median(rV[wall & (uV > 0)]) * 1000:.1f} mm")


def dev_colour(dev):
    """blue (inside) - white - red (outside), +-DEV"""
    x = np.clip(dev / DEV, -1, 1)
    c = np.empty((len(x), 3))
    neg = x < 0
    c[neg] = np.column_stack([1 + x[neg], 1 + x[neg], np.ones(neg.sum())]) * [255, 255, 255]
    c[~neg] = np.column_stack([np.ones((~neg).sum()), 1 - x[~neg], 1 - x[~neg]]) * [255, 255, 255]
    c[neg] = c[neg] * [0.75, 0.85, 1.0]
    c[~neg] = c[~neg] * [1.0, 0.8, 0.75]
    return c.astype(np.uint8)


devRGB = dev_colour(devL)


# ------------------------------------------------------------- rendering ----------
def pinhole_cam(R, C, w, h):
    """render_views camera dict for a world->camera rotation R (OpenCV axes) and centre C. The rasteriser puts
    the principal point at the picture centre, so the canvas is made big enough and cropped afterwards."""
    s = w / W
    fx, cx, cy = K[0, 0] * s, K[0, 2] * s, K[1, 2] * s
    cw, ch = 2 * int(np.ceil(max(cx, w - cx))), 2 * int(np.ceil(max(cy, h - cy)))
    c = dict(eye=np.asarray(C, float), R=np.array([R[0], -R[1], R[2]]), size=(cw * rv.SS, ch * rv.SS), focal=fx * rv.SS)
    crop = (int(round(cw / 2 - cx)), int(round(ch / 2 - cy)))
    return c, crop


def render_mesh(R, C, w=PW, h=PH):
    c, (x0, y0) = pinhole_cam(R, C, w, h)
    img = rv.draw(mesh, c, light=0.0, near=0.02)
    return img[y0:y0 + h, x0:x0 + w]


def splat(P, rgb, R, C, w=PW, h=PH, size=2, near=0.02, bg=255):
    """z-buffered point drawing from a pinhole camera (recorded K scaled to w x h)."""
    s = w / W
    x = (P - C) @ R.T
    ok = x[:, 2] > near
    x, col = x[ok], rgb[ok]
    u = K[0, 0] * s * x[:, 0] / x[:, 2] + K[0, 2] * s
    vv = K[1, 1] * s * x[:, 1] / x[:, 2] + K[1, 2] * s
    order = np.argsort(-x[:, 2])                       # far first: nearer points overwrite them
    u, vv, col = u[order], vv[order], col[order]
    img = np.full((h, w, 3), bg, np.uint8)
    for dy in range(size):
        for dx in range(size):
            ui, vi = (u + dx - size / 2).astype(int), (vv + dy - size / 2).astype(int)
            m = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
            img[vi[m], ui[m]] = col[m]
    return img


def label(img, text, sub=None):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, d.textlength(text, font=FONT) + 14, 28], fill=(0, 0, 0))
    d.text((7, 2), text, font=FONT, fill=(255, 255, 0))
    if sub:
        y = img.shape[0] - 24 * len(sub.split("\n")) - 4
        d.rectangle([0, y - 2, max(d.textlength(t, font=FONT_S) for t in sub.split("\n")) + 12, img.shape[0]], fill=(0, 0, 0))
        d.multiline_text((6, y), sub, font=FONT_S, fill=(230, 230, 230), spacing=6)
    return np.asarray(im)


def cross_section(R, C, w=PW, h=PH):
    """slab AHEAD in front of the camera, perpendicular to the local pipe axis, seen as the camera sees it:
    vision dense (black), LiDAR (red), a 200 mm reference circle (grey), centred on the vision centerline."""
    u, Q, d = centerline(C[None])
    u_x = u[0] + AHEAD
    Qx = (B + u_x * (a2 if u_x > 0 else a1))
    d = a2 if u_x > 0 else a1
    right = R[0] - (R[0] @ d) * d
    right /= np.linalg.norm(right)
    down = np.cross(d, right)
    img = np.full((h, w, 3), 255, np.uint8)
    sc = (h - 60) / 0.26                                  # pixels per metre
    ox, oy = w // 2, h // 2 + 10

    def to_px(P):
        q = P - Qx
        return (ox + sc * (q @ right)).astype(int), (oy + sc * (q @ down)).astype(int)

    for rr, col in ((0.100, (200, 200, 200)),):
        cv2.circle(img, (ox, oy), int(sc * rr), col, 1, cv2.LINE_AA)
    cv2.line(img, (ox - 6, oy), (ox + 6, oy), (150, 150, 150), 1)
    cv2.line(img, (ox, oy - 6), (ox, oy + 6), (150, 150, 150), 1)
    mV = (np.abs((V - Qx) @ d) < SLAB) & wall
    mL = np.abs((L - Qx) @ d) < SLAB
    for P, col, dots in ((L[mL], (240, 120, 120), 1), (V[mV], (0, 0, 0), 2)):    # sparse vision points on top
        x, y = to_px(P)
        ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        for dy in range(dots):
            for dx in range(dots):
                img[np.clip(y[ok] + dy, 0, h - 1), np.clip(x[ok] + dx, 0, w - 1)] = col
    rv_, rl_ = np.median(np.linalg.norm(((V[mV] - Qx) - np.outer((V[mV] - Qx) @ d, d)), axis=1)) if mV.sum() > 50 else np.nan, \
        np.median(np.linalg.norm(((L[mL] - Qx) - np.outer((L[mL] - Qx) @ d, d)), axis=1)) if mL.sum() > 50 else np.nan
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    dr.text((8, 34), f"视觉稠密（黑点）：中位半径 {rv_ * 1000:.0f} mm  ({mV.sum()} 点)", font=FONT_S, fill=(40, 40, 40))
    dr.text((8, 56), f"激光雷达（粉红）：中位半径 {rl_ * 1000:.0f} mm  ({mL.sum()} 点)", font=FONT_S, fill=(200, 30, 30))
    dr.text((8, 78), "灰圈：直径 200 mm 参考", font=FONT_S, fill=(140, 140, 140))
    return label(np.asarray(im), f"(5) 横截面：相机前方 {AHEAD * 100:.0f} cm，厚 {2 * SLAB * 1000:.0f} mm",
                 f"管段：{'第二节（灰）' if u_x > 0 else '第一节（黑）'}；离弯头 {u_x:+.2f} m"), (rv_, rl_, u_x)


# overview map (top view onto the bend plane) as a static background, the camera drawn on it per frame
nrm = np.cross(a1, a2)
nrm /= np.linalg.norm(nrm)
e2 = np.cross(nrm, a1)
traj = np.array([fused[i] for i in cam["idx"]])
o = traj[0]
pick = np.random.default_rng(0).choice(len(L), 600000, replace=False)
qq = L[pick] - o
mx_lo, mx_hi = (qq @ a1).min() - 0.05, (qq @ a1).max() + 0.05
my_lo, my_hi = (qq @ e2).min() - 0.05, (qq @ e2).max() + 0.05
msc = min((PW - 40) / (mx_hi - mx_lo), (PH - 70) / (my_hi - my_lo))


def map_px(P):
    q = np.atleast_2d(P) - o
    return (20 + msc * (q @ a1 - mx_lo)).astype(int), (PH - 20 - msc * (q @ e2 - my_lo)).astype(int)


base = np.full((PH, PW, 3), 255, np.uint8)
order = np.argsort(qq @ nrm)
x, y = map_px(L[pick][order])
ok = (x >= 0) & (x < PW) & (y >= 0) & (y < PH)
base[y[ok], x[ok]] = Lrgb[pick][order][ok]
tx, ty = map_px(traj)
for k in range(len(tx) - 1):
    cv2.line(base, (tx[k], ty[k]), (tx[k + 1], ty[k + 1]), (255, 0, 0), 1, cv2.LINE_AA)


def overview(k, C, R):
    img = base.copy()
    x, y = map_px(C)
    ahead = C + 0.12 * R[2]
    x2, y2 = map_px(ahead)
    cv2.circle(img, (int(x[0]), int(y[0])), 7, (0, 160, 0), -1)
    cv2.arrowedLine(img, (int(x[0]), int(y[0])), (int(x2[0]), int(y2[0])), (0, 160, 0), 2, tipLength=0.35)
    u, _, _ = centerline((C + AHEAD * R[2])[None])
    xs, ys = map_px(C + AHEAD * R[2])
    cv2.circle(img, (int(xs[0]), int(ys[0])), 4, (220, 30, 30), -1)
    return label(img, "(6) 俯视全图：彩色雷达管壁 + 融合轨迹（蓝）",
                 "绿：当前相机位置/朝向   红点：横截面位置")


# ----------------------------------------------------------- outside sheet ----------
def outside_views():
    """the two models from outside and cut open, same virtual cameras, metric, side by side"""
    up = -np.mean(cam["R"][:, 1], axis=0)
    up -= (up @ a1) * a1
    up /= np.linalg.norm(up)
    sel_m = np.linalg.norm(mesh["centroid"] - centerline(mesh["centroid"])[1], axis=1) < 0.16
    uu = centerline(mesh["centroid"])[0]
    sel_m &= (uu > -1.1) & (uu < 1.0)
    target = B + 0.5 * ((-0.85) * a1 + 0.8 * a2)
    side = np.cross(up, a1)
    pics = []
    for name, az, el, cut in (("外观", 0.0, 22.0, False), ("剖开（去掉靠近观察者的一半）", 0.0, 35.0, True)):
        back = np.cos(np.radians(el)) * side + np.sin(np.radians(el)) * up
        eye = target + 3.0 * back
        c = rv.camera(eye, target, (1300, 560), fov=40, up=up)
        keep = sel_m.copy()
        keepL = (rL < 0.16) & (uL > -1.1) & (uL < 1.0)
        if cut:
            keep &= (mesh["centroid"] - centerline(mesh["centroid"])[1]) @ back < 0
            keepL &= qL @ back < 0
        a = rv.draw(mesh, c, keep, light=0.35 if not cut else 0.2, gain=1.25)
        Rw = c["R"] * [[1], [-1], [1]]                    # back to OpenCV axes for splat()
        s = 1300 / W
        fx = c["focal"] / rv.SS
        Kc = np.array([[fx, 0, 650], [0, fx, 280], [0, 0, 1]])
        xx = (L[keepL] - eye) @ Rw.T
        okk = xx[:, 2] > 0.05
        uu_ = Kc[0, 0] * xx[okk, 0] / xx[okk, 2] + 650
        vv_ = Kc[1, 1] * xx[okk, 1] / xx[okk, 2] + 280
        order = np.argsort(-xx[okk, 2])
        b = np.full((560, 1300, 3), 255, np.uint8)
        col = Lrgb[keepL][okk][order]
        col = np.clip(col.astype(float) * 1.25, 0, 255).astype(np.uint8)
        for dy in (0, 1):
            for dx in (0, 1):
                ui, vi = (uu_[order] + dx).astype(int), (vv_[order] + dy).astype(int)
                m = (ui >= 0) & (ui < 1300) & (vi >= 0) & (vi < 560)
                b[vi[m], ui[m]] = col[m]
        c2 = np.full((560, 1300, 3), 255, np.uint8)
        col2 = devRGB[keepL][okk][order]
        for dy in (0, 1):
            for dx in (0, 1):
                ui, vi = (uu_[order] + dx).astype(int), (vv_[order] + dy).astype(int)
                m = (ui >= 0) & (ui < 1300) & (vi >= 0) & (vi < 560)
                c2[vi[m], ui[m]] = col2[m]
        pics.append((name, rv.crop(a, 12), rv.crop(b, 12), rv.crop(c2, 12)))
        del s
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    hw = max(p[k].shape[0] / p[k].shape[1] for p in pics for k in (1, 2, 3))
    fig, ax = plt.subplots(3, 2, figsize=(20, 3 * (10 * hw + 0.45) + 0.6))
    for j, (name, a, b, c2) in enumerate(pics):
        ax[0, j].imshow(a)
        ax[0, j].set_title(f"视觉：OpenMVS 带纹理网格，{name}", fontsize=13, loc="left")
        ax[1, j].imshow(b)
        ax[1, j].set_title(f"融合：激光雷达管壁点 + 相机颜色，{name}", fontsize=13, loc="left")
        ax[2, j].imshow(c2)
        ax[2, j].set_title(f"激光雷达几何：去掉每节平均截面（椭圆）后的半径残差（蓝 = 偏小，红 = 偏大，±{DEV * 1000:.0f} mm），{name}", fontsize=13, loc="left")
        for i in range(3):
            ax[i, j].axis("off")
    fig.suptitle("InPipe-SLAM 172011：两种模态的三维模型，同一虚拟相机、同一公制坐标系（AprilTag 尺度）", fontsize=15)
    fig.tight_layout()
    path = os.path.join(OUT, "model_views.png")
    fig.savefig(path, dpi=80)
    plt.close(fig)
    print("saved", path, flush=True)


outside_views()
if SHEET_ONLY:
    sys.exit(0)

# ------------------------------------------------------------ per-frame video ----------
vw = cv2.VideoWriter(os.path.join(OUT, "compare.mp4"), cv2.VideoWriter_fourcc(*"avc1"), 4, (3 * PW, 2 * PH))
log = []
sheet_frames = []
ks = np.arange(0, len(cam["idx"]), EVERY)
picks = set(ks[(np.array([0.12, 0.4, 0.6, 0.85]) * (len(ks) - 1)).astype(int)].tolist())   # also saved as JPEG
for k in range(0, len(cam["idx"]), EVERY):
    idx = int(cam["idx"][k])
    R, Cv = cam["R"][k], cam["C"][k] * lam
    Cf = fused[idx]
    raw = cv2.imread(os.path.join(EX, "images", "color", f"{idx:06d}.png"))
    p1 = label(cv2.resize(raw, (PW, PH), interpolation=cv2.INTER_AREA)[:, :, ::-1], "(1) 原始视频帧（前视彩色相机）",
               f"彩色帧 {idx}   行程 s = {cam['s'][k] * lam:.2f} m（视觉）")
    p2 = label(render_mesh(R, Cv), "(2) 视觉重建：OpenMVS 带纹理网格", "同一相机位姿、同一内参渲染（纹理来自做过 CLAHE 的帧）")
    p3 = label(splat(L, Lrgb, R, Cf), "(3) 融合：雷达管壁点 + 相机颜色", "雷达几何（0.2–0.6 m 量程，全程累积），融合位姿")
    p4 = label(splat(L, devRGB, R, Cf), f"(4) 雷达几何：半径残差 ±{DEV * 1000:.0f} mm", "已去掉每节的平均截面（椭圆）   蓝：偏小   红：偏大")
    p5, (rv_, rl_, u_x) = cross_section(R, Cf)
    p6 = overview(k, Cf, R)
    frame = np.vstack([np.hstack([p1, p2, p3]), np.hstack([p4, p5, p6])])
    vw.write(frame[:, :, ::-1].copy())
    log.append([idx, round(float(u_x), 3), round(float(rv_) * 1000, 1), round(float(rl_) * 1000, 1)])
    if k in picks:
        sheet_frames.append(frame)
        cv2.imencode(".jpg", frame[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(os.path.join(OUT, f"compare_{idx:06d}.jpg"))
    if k % 20 == 0:
        print(f"frame {k}/{len(cam['idx'])}: colour {idx}, section at {u_x:+.2f} m from the bend, "
              f"radius vision {rv_ * 1000:.0f} / LiDAR {rl_ * 1000:.0f} mm", flush=True)
vw.release()
with open(os.path.join(OUT, "cross_sections.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["color_idx", "section_from_bend_m", "vision_radius_mm", "lidar_radius_mm"])
    w.writerows(log)
print("saved", os.path.join(OUT, "compare.mp4"))
