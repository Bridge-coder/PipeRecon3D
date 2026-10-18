"""R5: the new model (R1-R4) next to the earlier fused result (f5_fuse.py) and the video.

  default   OBJ export (textured, and coloured by the local deviation), outside / cut-away sheet, full-size
            render triplets of four frames, and a held-out check: colour frames between the SfM frames (not
            used for the texture) against renders from their poses - how far the render's edges lie from the
            real frame's edges (pixels)
  --video   compare_new.mp4, six panels per SfM frame:
            (1) original frame  (2) before: LiDAR points + camera colour  (3) now: textured surface
            (4) now: local deviation of the surface  (5) cross-section 30 cm ahead, before vs now
            (6) the new unrolled texture with the camera position
Poses: "before" = the fused pose (as compare_views.py drew it); "now" = the fused pose with the R2b correction
(as the texture was made). The mesh ends at U_MAX (the pipe exit; beyond it the LiDAR sees the end board).

output: RECON_BIG/model/ (OBJ, MTL, PNG), RECON_BIG/viz/;
        results/inpipe_172011/recon/ (sheet, frames, r5_heldout.json); compare_new.mp4 in both places
usage: python -I r5_compare.py [--video] [--every N]
"""
import json
import os
import shutil
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recon_common import BIG, EX, K, W, H, OUT, Centerline, FusedPoses, heightfield_mesh, off, stream_times  # noqa: E402
from vision_model import _expmap  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vision"))
import render_views as rv  # noqa: E402
from paths import FUSED_BIG  # noqa: E402

args = sys.argv[1:]
VIDEO = "--video" in args
EVERY = int(args[args.index("--every") + 1]) if "--every" in args else 1
U_MAX = 1.005
PW, PH = 640, 360
AHEAD, SLAB, DEV = 0.30, 0.006, 0.002
FONT = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 19)
FONT_S = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 15)
MODEL, VIZ = os.path.join(BIG, "model"), os.path.join(BIG, "viz")
os.makedirs(MODEL, exist_ok=True)
os.makedirs(VIZ, exist_ok=True)
t0 = time.time()

# ---------------------------------------------------------------- data ------------------------------------
cl = Centerline(os.path.join(BIG, "centerline.npz"))
hf = np.load(os.path.join(BIG, "heightfield.npz"))
m = heightfield_mesh(hf, cl)
xyz, tri, vuv = m["xyz"], m["tri"], m["vuv"]
u_lo, u_hi = m["uv_range"]
u_v = u_lo + vuv[:, 0] * (u_hi - u_lo)
tri = tri[(u_v[tri] <= U_MAX).all(1)]
tex = cv2.imdecode(np.fromfile(os.path.join(BIG, "texture_refined.png"), np.uint8), cv2.IMREAD_COLOR)
a_, b_ = m["rows"]
dl = hf["dev_local"][a_:b_]                                 # nu x nt, theta ascending


def dev_colour(dev, lim):
    """blue (inside) - white - red (outside), +-lim; grey where unknown (RGB)"""
    x = np.clip(np.nan_to_num(dev) / lim, -1, 1)
    c = np.empty(x.shape + (3,))
    neg = x < 0
    c[neg] = np.stack([1 + x[neg], 1 + x[neg], np.ones(neg.sum())], -1) * [0.75 * 255, 0.85 * 255, 255]
    c[~neg] = np.stack([np.ones((~neg).sum()), 1 - x[~neg], 1 - x[~neg]], -1) * [255, 0.8 * 255, 0.75 * 255]
    c[~np.isfinite(dev)] = (175, 175, 175)
    return c.astype(np.uint8)


dev_tex = dev_colour(dl.T[::-1], DEV)[:, :, ::-1].copy()      # rows: theta +180 at the top, BGR like tex
dev_tex = cv2.resize(dev_tex, (dev_tex.shape[1] * 4, dev_tex.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
fn = np.cross(xyz[tri[:, 1]] - xyz[tri[:, 0]], xyz[tri[:, 2]] - xyz[tri[:, 0]])
vn = np.stack([np.bincount(tri.ravel(), np.repeat(fn[:, k], 3), len(xyz)) for k in range(3)], axis=1)
vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-30


def mesh_dict(texture):
    return dict(xyz=xyz, tri=tri, uv=vuv[tri], tid=np.zeros(len(tri), np.int8), tex=[texture],
                blank=np.zeros(len(tri), bool), vn=vn.astype(np.float32), centroid=xyz[tri].mean(axis=1))


mesh_tex, mesh_dev = mesh_dict(tex), mesh_dict(dev_tex)
print(f"mesh: {len(xyz):,} vertices, {len(tri):,} triangles (u <= {U_MAX} m) [{time.time() - t0:.0f} s]", flush=True)

with open(os.path.join(FUSED_BIG, "lidar_coloured_world.ply"), "rb") as f:
    while f.readline().strip() != b"end_header":
        pass
    rec = np.frombuffer(f.read(), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
L = np.column_stack([rec["x"], rec["y"], rec["z"]]).astype(np.float64)
Lrgb = np.column_stack([rec["r"], rec["g"], rec["b"]])
uL, thL, rL = cl.coords(L)
zw = np.load(os.path.join(BIG, "wall_refined.npz"))
pick = np.random.default_rng(0).choice(len(zw["u"]), 3_000_000, replace=False)
uN, thN, rN = zw["u"][pick].astype(float), zw["th"][pick].astype(float), zw["r"][pick].astype(float)
oN, oL = np.argsort(uN), np.argsort(uL)
uN_s, uL_s = uN[oN], uL[oL]
print(f"before: {len(L):,} LiDAR points; now: {len(uN):,} refined wall points (subsample) [{time.time() - t0:.0f} s]", flush=True)

fp = FusedPoses()
cam = fp.cam
sc = np.load(os.path.join(BIG, "scan_corrections.npz"))
tsc = sc["t"].astype(float)


def poses_at(t):
    """(R, C) before (fused) and now (fused + R2b correction)"""
    R, C, _ = fp.pose(t)
    f = lambda A: np.array([np.interp(float(t), tsc, A[:, j]) for j in range(3)])  # noqa: E731
    d, w, O = f(sc["d"]), f(sc["w"]), f(sc["O"])
    return (R, C), (R @ _expmap(w).T, C + d + np.cross(w, C - O))


# ------------------------------------------------------------- OBJ export ----------------------------------
if not VIDEO:
    used = np.unique(tri)
    remap = np.full(len(xyz), -1, np.int64)
    remap[used] = np.arange(len(used)) + 1
    cv2.imencode(".png", tex)[1].tofile(os.path.join(MODEL, "pipe_texture.png"))
    cv2.imencode(".png", dev_tex)[1].tofile(os.path.join(MODEL, "pipe_deviation.png"))
    lines_v = "\n".join(f"v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}" for p in xyz[used])
    lines_vt = "\n".join(f"vt {q[0]:.6f} {q[1]:.6f}" for q in vuv[used])
    lines_vn = "\n".join(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}" for n in vn[used])
    ft = remap[tri]
    lines_f = "\n".join(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}" for a, b, c in ft)
    for name, png in (("pipe_textured", "pipe_texture.png"), ("pipe_deviation", "pipe_deviation.png")):
        with open(os.path.join(MODEL, f"{name}.mtl"), "w") as f:
            f.write(f"newmtl wall\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nillum 1\nmap_Kd {png}\n")
        with open(os.path.join(MODEL, f"{name}.obj"), "w") as f:
            f.write(f"# InPipe-SLAM 172011 pipe wall, metric, fused world frame (vision axes, AprilTag scale)\n"
                    f"mtllib {name}.mtl\n{lines_v}\n{lines_vt}\n{lines_vn}\nusemtl wall\n{lines_f}\n")
    print(f"OBJ -> {MODEL} ({len(used):,} vertices, {len(tri):,} triangles) [{time.time() - t0:.0f} s]", flush=True)


# ------------------------------------------------------------- rendering -----------------------------------
def pinhole_cam(R, C, w, h):
    """render_views camera for a world->camera rotation R (OpenCV axes) and centre C (as compare_views.py)"""
    s = w / W
    fx, cx, cy = K[0, 0] * s, K[0, 2] * s, K[1, 2] * s
    cw, ch = 2 * int(np.ceil(max(cx, w - cx))), 2 * int(np.ceil(max(cy, h - cy)))
    c = dict(eye=np.asarray(C, float), R=np.array([R[0], -R[1], R[2]]), size=(cw * rv.SS, ch * rv.SS), focal=fx * rv.SS)
    return c, (int(round(cw / 2 - cx)), int(round(ch / 2 - cy)))


def render(mesh, R, C, w=PW, h=PH, light=0.0):
    c, (x0, y0) = pinhole_cam(R, C, w, h)
    return rv.draw(mesh, c, light=light, near=0.02)[y0:y0 + h, x0:x0 + w]


def splat(P, rgb, R, C, w=PW, h=PH, size=2, near=0.02):
    s = w / W
    x = (P - C) @ R.T
    ok = x[:, 2] > near
    x, col = x[ok], rgb[ok]
    u = K[0, 0] * s * x[:, 0] / x[:, 2] + K[0, 2] * s
    v = K[1, 1] * s * x[:, 1] / x[:, 2] + K[1, 2] * s
    o = np.argsort(-x[:, 2])
    u, v, col = u[o], v[o], col[o]
    img = np.full((h, w, 3), 255, np.uint8)
    for dy in range(size):
        for dx in range(size):
            ui, vi = (u + dx - size / 2).astype(int), (v + dy - size / 2).astype(int)
            mm = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
            img[vi[mm], ui[mm]] = col[mm]
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


def slab(u_sorted, order, u_x):
    a, b = np.searchsorted(u_sorted, [u_x - SLAB, u_x + SLAB])
    return order[a:b]


def thickness(th, r):
    if len(r) < 200:
        return np.nan
    it = ((th + 180) // 4).astype(int) % 90
    med = np.array([np.median(r[it == q]) if (it == q).any() else np.nan for q in range(90)])
    e = r - med[it]
    e = e[np.isfinite(e)]
    p5, p95 = np.percentile(e, [5, 95])
    return 1000 * (p95 - p5)


def section_px(u_x, th, r, R, w, h, sc_):
    """pixel coordinates of pipe points at u_x seen along the pipe, oriented like the camera image"""
    P, _, T = cl.point(np.full(len(th), u_x), th, r)
    Cx, _, Tx = cl.point(np.array([u_x]), np.array([0.0]), np.array([0.0]))
    Tx = Tx[0]
    right = R[0] - (R[0] @ Tx) * Tx
    right /= np.linalg.norm(right)
    down = np.cross(Tx, right)
    q = P - Cx[0]
    return (w // 2 + sc_ * (q @ right)).astype(int), (h // 2 + 12 + sc_ * (q @ down)).astype(int)


def cross_section(u_x, R, w=PW, h=PH):
    img = np.full((h, w, 3), 255, np.uint8)
    sc_ = (h - 70) / 0.24
    cv2.circle(img, (w // 2, h // 2 + 12), int(sc_ * 0.100), (210, 210, 210), 1, cv2.LINE_AA)
    iL, iN = slab(uL_s, oL, u_x), slab(uN_s, oN, u_x)
    x, y = section_px(u_x, thL[iL], rL[iL], R, w, h, sc_)
    ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    img[y[ok], x[ok]] = (240, 130, 130)
    x, y = section_px(u_x, thN[iN], rN[iN], R, w, h, sc_)
    ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    img[y[ok], x[ok]] = (90, 90, 90)
    # the surface: height field row at u_x
    i = int(np.clip(round((u_x - hf["u"][0]) / float(hf["du"]) - 0.5), 0, len(hf["u"]) - 1))
    rr = hf["r_mesh"][i]
    good = np.isfinite(rr)
    x, y = section_px(u_x, hf["th"][good], rr[good], R, w, h, sc_)
    pts = np.column_stack([x, y])
    brk = np.flatnonzero(np.diff(np.flatnonzero(good)) > 1) + 1
    for seg in np.split(pts, brk):
        if len(seg) > 1:
            cv2.polylines(img, [seg.reshape(-1, 1, 2).astype(np.int32)], False, (0, 0, 0), 1, cv2.LINE_AA)
    tb, tn = thickness(thL[iL], rL[iL]), thickness(thN[iN], rN[iN])
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.text((8, 34), f"之前（粉红）：融合雷达点，管壁厚 {tb:.0f} mm（5–95%）", font=FONT_S, fill=(210, 60, 60))
    d.text((8, 56), f"现在（灰点）：校正 + 细化后的雷达点，管壁厚 {tn:.1f} mm", font=FONT_S, fill=(70, 70, 70))
    d.text((8, 78), "黑线：新网格的截面   浅灰圈：直径 200 mm 参考", font=FONT_S, fill=(110, 110, 110))
    return label(np.asarray(im), f"(5) 横截面：相机前方 {AHEAD * 100:.0f} cm，厚 {2 * SLAB * 1000:.0f} mm",
                 f"沿管位置 u = {u_x:+.2f} m（0 = 弯头）"), tb, tn


strip_h = int(PW * tex.shape[0] / tex.shape[1])
strip = cv2.resize(tex, (PW, strip_h), interpolation=cv2.INTER_AREA)[:, :, ::-1]


def unrolled(u_cam, u_x):
    img = np.full((PH, PW, 3), 255, np.uint8)
    y0 = 36
    img[y0:y0 + strip_h] = strip
    for uu, col in ((u_cam, (0, 160, 0)), (u_x, (220, 30, 30))):
        x = int((uu - u_lo) / (u_hi - u_lo) * PW)
        if 0 <= x < PW:
            cv2.line(img, (x, y0), (x, y0 + strip_h), col, 2)
    return label(img, "(6) 新展开图：按纹理像素从图像取色（0.5 mm）",
                 "横向：沿管 u   纵向：转角（上下边 = 底缝）\n绿：相机位置   红：横截面位置")


color_t, color_rows = stream_times(EX, off, "color")
t_of = {int(r["idx"]): t for r, t in zip(color_rows, color_t)}


def load_raw(idx):
    return cv2.imread(os.path.join(EX, "images", "color", f"{idx:06d}.png"))


# ------------------------------------------------------------- video ---------------------------------------
if VIDEO:
    out_mp4 = os.path.join(VIZ, "compare_new.mp4")
    vw = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"avc1"), 8, (3 * PW, 2 * PH))
    if not vw.isOpened():                              # no H.264 encoder in this OpenCV build: MPEG-4 part 2
        vw = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), 8, (3 * PW, 2 * PH))
    assert vw.isOpened(), "no video encoder"
    log = []
    for k in range(0, len(cam["idx"]), EVERY):
        idx = int(cam["idx"][k])
        t = cam["t"][k]
        (Rb, Cb), (Rn, Cn) = poses_at(t)
        u_cam = float(cl.coords(Cn[None])[0][0])
        u_x = u_cam + AHEAD
        p1 = label(cv2.resize(load_raw(idx), (PW, PH), interpolation=cv2.INTER_AREA)[:, :, ::-1],
                   "(1) 原始视频帧（前视彩色相机）", f"彩色帧 {idx}   沿管位置 u = {u_cam:+.2f} m（0 = 弯头）")
        p2 = label(splat(L, Lrgb, Rb, Cb), "(2) 之前：雷达管壁点 + 相机颜色", "每个雷达点取一个颜色（f5_fuse），融合位姿")
        p3 = label(render(mesh_tex, Rn, Cn), "(3) 现在：雷达高度场网格 + 逐像素贴图",
                   "偏差自标定 + 位姿细化；纹理 0.5 mm，平场与曝光校正")
        p4 = label(render(mesh_dev, Rn, Cn, light=0.3), f"(4) 现在：局部几何偏差 ±{DEV * 1000:.0f} mm",
                   "减去沿管 ±5 cm 滑动中位数   蓝：凹   红：凸")
        p5, tb, tn = cross_section(u_x, Rn)
        p6 = unrolled(u_cam, u_x)
        frame = np.vstack([np.hstack([p1, p2, p3]), np.hstack([p4, p5, p6])])
        vw.write(frame[:, :, ::-1].copy())
        log.append([idx, round(u_x, 3), round(float(tb), 1), round(float(tn), 2)])
        if idx in (840, 1640, 2200, 2920):
            cv2.imencode(".jpg", frame[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(
                os.path.join(OUT, f"compare_new_{idx:06d}.jpg"))
        if k % 20 == 0:
            print(f"frame {k}/{len(cam['idx'])}: colour {idx}, section u {u_x:+.2f} m, wall before {tb:.0f} mm / now {tn:.1f} mm "
                  f"[{time.time() - t0:.0f} s]", flush=True)
    vw.release()
    shutil.copy(out_mp4, os.path.join(RES_VIZ := os.path.join(os.path.dirname(OUT), "viz"), "compare_new.mp4"))
    lg = np.array(log, float)
    ok = np.isfinite(lg[:, 2]) & np.isfinite(lg[:, 3])
    json.dump({"frames": len(log), "wall_thickness_in_section_mm_before_median": float(np.median(lg[ok, 2])),
               "wall_thickness_in_section_mm_now_median": float(np.median(lg[ok, 3])), "per_frame": log},
              open(os.path.join(OUT, "r5_video_sections.json"), "w"), indent=0)
    print(f"video -> {out_mp4} and {RES_VIZ}; section wall thickness median before {np.median(lg[ok, 2]):.1f} mm, "
          f"now {np.median(lg[ok, 3]):.1f} mm [{time.time() - t0:.0f} s]")
    sys.exit(0)

# ------------------------------------------------------- full-size frames, held-out check -------------------
for idx in (840, 1640, 2200, 2920):
    (Rb, Cb), (Rn, Cn) = poses_at(t_of[idx])
    trip = [load_raw(idx)[:, :, ::-1], splat(L, Lrgb, Rb, Cb, W, H, size=3), render(mesh_tex, Rn, Cn, W, H)]
    cv2.imencode(".jpg", np.hstack(trip)[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(
        os.path.join(VIZ, f"triplet_{idx:06d}.jpg"))
print(f"full-size triplets -> {VIZ} [{time.time() - t0:.0f} s]", flush=True)


def edges(img):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (0, 0), 2.0)
    return cv2.Canny(g, 20, 50) > 0


rows_ = []
sfm = set(int(i) for i in cam["idx"])
for k in range(0, len(cam["idx"]) - 1, 8):
    idx = int(cam["idx"][k]) + 5                      # halfway to the next SfM frame: not used for the texture
    if idx in sfm or idx not in t_of:
        continue
    (Rb, Cb), (Rn, Cn) = poses_at(t_of[idx])
    real = load_raw(idx)[:, :, ::-1]
    er = edges(real)
    dt = cv2.distanceTransform((~er).astype(np.uint8), cv2.DIST_L2, 5)
    row = [idx]
    for img in (splat(L, Lrgb, Rb, Cb, W, H, size=3), render(mesh_tex, Rn, Cn, W, H)):
        cover = cv2.erode((img < 250).any(2).astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
        e = edges(img) & cover
        d = dt[e]
        row += [float(np.median(d)) if len(d) else np.nan, float((d <= 2).mean()) if len(d) else np.nan, int(e.sum())]
    rows_.append(row)
rows_ = np.array(rows_, float)
res = {"frames": int(len(rows_)),
       "what": "colour frames halfway between SfM frames (not used for the new texture). Canny edges (grey, blur 2 px) of the "
               "render vs of the real frame at 1280 x 720: distance of each render edge pixel to the nearest real edge, over "
               "the area the render covers. The earlier LiDAR colours were taken from the frames nearest in time, so for "
               "'before' these frames are not held out; its single coloured points also make many edges of their own",
       "before_median_px": float(np.nanmedian(rows_[:, 1])), "before_share_within_2px": float(np.nanmedian(rows_[:, 2])),
       "before_edge_pixels": float(np.nanmedian(rows_[:, 3])),
       "now_median_px": float(np.nanmedian(rows_[:, 4])), "now_share_within_2px": float(np.nanmedian(rows_[:, 5])),
       "now_edge_pixels": float(np.nanmedian(rows_[:, 6])),
       "per_frame [idx, before median px, before share <=2 px, before edge px, now ...]": rows_.round(3).tolist()}
json.dump(res, open(os.path.join(OUT, "r5_heldout.json"), "w"), indent=0)
print(f"held-out frames: {len(rows_)}; render edges to real edges, median: before {res['before_median_px']:.1f} px "
      f"({res['before_share_within_2px']:.0%} within 2 px), now {res['now_median_px']:.1f} px "
      f"({res['now_share_within_2px']:.0%} within 2 px) [{time.time() - t0:.0f} s]", flush=True)

# ------------------------------------------------------------- outside sheet ---------------------------------
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
legs = json.load(open(os.path.join(os.path.dirname(OUT), "fusion", "vision_legs.json")))
lam = json.load(open(os.path.join(os.path.dirname(OUT), "fusion", "f4_tag_scale.json")))["mm_per_unit_tag"] / 1000
a1, a2 = np.array(legs["legs"][0]["dir"]), np.array(legs["legs"][1]["dir"])
B = np.array(legs["bend_point"]) * lam
up = -np.mean(cam["R"][:, 1], axis=0)
up -= (up @ a1) * a1
up /= np.linalg.norm(up)
side = np.cross(up, a1)
target = B + 0.5 * ((-0.85) * a1 + 0.8 * a2)
i_c = np.clip(np.round((u_v - cl.u[0]) / cl.ds).astype(int), 0, len(cl.u) - 1)
q_v = xyz - cl.C[i_c]                                      # vertex offset from the centerline
cent_q = q_v[tri].mean(axis=1)
qL = L - cl.C[np.clip(np.round((uL - cl.u[0]) / cl.ds).astype(int), 0, len(cl.u) - 1)]
pics = []
for name, el, cut in (("外观", 22.0, False), ("剖开（去掉靠近观察者的一半）", 35.0, True)):
    back = np.cos(np.radians(el)) * side + np.sin(np.radians(el)) * up
    eye = target + 3.0 * back
    c = rv.camera(eye, target, (1300, 560), fov=40, up=up)
    keep = np.ones(len(tri), bool)
    keepL = (rL < 0.16) & (uL > -0.55) & (uL < U_MAX)
    if cut:
        keep &= cent_q @ back < 0
        keepL &= qL @ back < 0
    img_t = rv.crop(rv.draw(mesh_tex, c, keep, light=0.0 if not cut else 0.0, gain=1.0), 12)
    img_d = rv.crop(rv.draw(mesh_dev, c, keep, light=0.35), 12)
    Rw = c["R"] * [[1], [-1], [1]]
    fx = c["focal"] / rv.SS
    xx = (L[keepL] - eye) @ Rw.T
    okk = xx[:, 2] > 0.05
    uu_ = fx * xx[okk, 0] / xx[okk, 2] + 650
    vv_ = fx * xx[okk, 1] / xx[okk, 2] + 280
    o = np.argsort(-xx[okk, 2])
    bimg = np.full((560, 1300, 3), 255, np.uint8)
    col = np.clip(Lrgb[keepL][okk][o].astype(float) * 1.25, 0, 255).astype(np.uint8)
    for dy in (0, 1):
        for dx in (0, 1):
            ui, vi = (uu_[o] + dx).astype(int), (vv_[o] + dy).astype(int)
            mm = (ui >= 0) & (ui < 1300) & (vi >= 0) & (vi < 560)
            bimg[vi[mm], ui[mm]] = col[mm]
    pics.append((name, rv.crop(bimg, 12), img_t, img_d))
hw = max(p[k].shape[0] / p[k].shape[1] for p in pics for k in (1, 2, 3))
fig, ax = plt.subplots(3, 2, figsize=(20, 3 * (10 * hw + 0.45) + 0.6))
for j, (name, bimg, img_t, img_d) in enumerate(pics):
    ax[0, j].imshow(bimg)
    ax[0, j].set_title(f"之前：雷达管壁点 + 相机颜色（f5_fuse），{name}", fontsize=13, loc="left")
    ax[1, j].imshow(img_t)
    ax[1, j].set_title(f"现在：雷达高度场网格 + 逐像素贴图，{name}", fontsize=13, loc="left")
    ax[2, j].imshow(img_d)
    ax[2, j].set_title(f"现在：局部几何偏差（减去沿管 ±5 cm 滑动中位数，±{DEV * 1000:.0f} mm），{name}", fontsize=13, loc="left")
    for i in range(3):
        ax[i, j].axis("off")
fig.suptitle("InPipe-SLAM 172011：之前与现在的三维模型，同一虚拟相机、同一公制坐标系", fontsize=15)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "r5_model_views.png"), dpi=80)
print(f"sheet -> {os.path.join(OUT, 'r5_model_views.png')} [{time.time() - t0:.0f} s]")
