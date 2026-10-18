"""Publication-style flowchart of the reconstruction pipeline as it was actually run (run_pipeline.py and the
evaluation scripts), in Chinese and English.

Usage: draw_flowchart.py OUT_DIR
Writes OUT_DIR/pipeline_flowchart_zh.{pdf,svg,png} and pipeline_flowchart_en.{pdf,svg,png}
(PDF with embedded TrueType fonts, SVG with editable text, PNG at 600 dpi). Width 7.2 in = a double-column figure.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

plt.rcParams["pdf.fonttype"] = 42          # embed TrueType, keeps text editable
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["axes.unicode_minus"] = False

# box styles: (fill, edge, dashed)
STYLE = {"data": ("#E4ECF5", "#4A6688", False), "std": ("#FFFFFF", "#555555", False),
         "mod": ("#FBE3C8", "#C9731E", False), "eval": ("#E6F0E1", "#5B8A4A", False)}
HEAD = ["#5B6675", "#3E6A9E", "#2E776F", "#6A5A8C", "#7A6A4F"]
INK = "#333333"

TEXT = {
    "zh": dict(
        font=["Times New Roman", "SimSun"], head_font=["Times New Roman", "SimHei"],
        stages=[
            ("数据与预处理", "", True, [
                ("管内 CCTV 视频\n720×576，25 帧/s", "data"),
                ("抽帧：每 5 帧取 1 帧\n（约 1 cm 一帧）", "std"),
                ("像素比校正，裁去黑边\n与字幕条 → 768×548", "mod"),
                ("CLAHE 对比度增强", "mod"),
                ("字幕计数器识别 → 行走\n距离真值（只用于标定\n和检验，不参与重建）", "eval")],
             "图像序列"),
            ("稀疏重建", "OpenMVG 1.6", True, [
                ("图像列表：内参固定\n针孔 + 径向畸变 k1", "mod"),
                ("SIFT 特征（ULTRA）", "std"),
                ("顺序匹配（后 25 帧）\n+ 基础矩阵 RANSAC", "std"),
                ("初始像对排序（>2° 的\n三角化角，前向运动）", "mod"),
                ("增量式 SfM，不优化内参；\n注册不足 90% 换下一像对", "mod"),
                ("颠簸处中断 → 分段重建", "mod")],
             "相机位姿、稀疏点云"),
            ("稠密重建与纹理", "OpenMVS 2.4.0", True, [
                ("剔除相机后方或贴近\n相机中心的三维点", "mod"),
                ("格式转换、去畸变\n+ 主点格式修正", "mod"),
                ("深度图稠密匹配与融合\n视角阈值放宽 3°/0.5°", "mod"),
                ("网格重建", "std"),
                ("纹理映射\n（关闭接缝匀色）", "mod")],
             "稠密点云、网格、纹理"),
            ("公制尺度与管轴坐标", "", True, [
                ("圆柱拟合管轴\n（稀疏点云）", "mod"),
                ("逐切片圆拟合\n（每片约 0.3 m）", "mod"),
                ("以管径定比例：全程\n一个比例或逐切片修正", "mod"),
                ("管轴坐标 (s, θ, r)\nθ = 0 为相机上方", "mod"),
                ("多段：按计数器对齐\n轴向位置后拼接", "mod")],
             "公制三维模型"),
            ("评估与可视化", "", False, [
                ("行走距离 vs 计数器", "eval"),
                ("管壁偏差、离散、\n圆周覆盖率", "eval"),
                ("模型互比（管壁亮度\n曲线对齐求伸缩系数）", "eval"),
                ("管壁展开图、\n带纹理三维视图", "eval"),
                ("模型内部视图 vs\n同位置视频帧", "eval")],
             "精度指标与图件"),
        ],
        output="输出：", calib_title="相机内参标定（只在视频 1 上做一次，每次重跑增量式 SfM）",
        calib=("① 扫描 k1，使重建出的管道沿程等径 → k1 = −0.33（只用“管道等径”，不用真值）\n"
               "② 调焦距，使行走距离等于计数器读数 → f = 806 px；主点 (383, 288) 为假设值"),
        to_calib="距离真值", from_calib="固定内参\n（所有视频共用）",
        prior="先验：管道内径 300 mm", legend=[("data", "输入数据"), ("std", "OpenMVG / OpenMVS 标准步骤"),
                                              ("mod", "针对管内前视视频增加或修改的步骤"), ("eval", "评估（不参与重建）")]),
    "en": dict(
        font=["Times New Roman"], head_font=["Times New Roman"],
        stages=[
            ("Data & pre-processing", "", True, [
                ("In-pipe CCTV video\n720×576, 25 fps", "data"),
                ("Keep every 5th frame\n(≈1 cm of travel)", "std"),
                ("Pixel-aspect fix, crop\nborder & OSD → 768×548", "mod"),
                ("CLAHE contrast\nenhancement", "mod"),
                ("OSD counter reading →\nground-truth travel (for\ncalibration & checks only)", "eval")],
             "image sequence"),
            ("Sparse reconstruction", "OpenMVG 1.6", True, [
                ("Image listing, fixed\nintrinsics, pinhole + k1", "mod"),
                ("SIFT features (ULTRA)", "std"),
                ("Sequential matching\n(next 25) + F-RANSAC", "std"),
                ("Initial-pair ranking\n(>2° rays, forward motion)", "mod"),
                ("Incremental SfM, intrinsics\nfixed; next pair if <90%", "mod"),
                ("Break at a jolt →\nreconstruct in parts", "mod")],
             "camera poses,\nsparse cloud"),
            ("Dense reconstruction", "OpenMVS 2.4.0", True, [
                ("Drop 3-D points behind\nor at a camera centre", "mod"),
                ("Export, undistort +\nprincipal-point fix", "mod"),
                ("Depth-map densification,\nview angles relaxed 3°/0.5°", "mod"),
                ("Mesh reconstruction", "std"),
                ("Texturing\n(seam leveling off)", "mod")],
             "dense cloud,\nmesh, texture"),
            ("Metric scale &\npipe frame", "", True, [
                ("Cylinder fit of the\npipe axis (sparse cloud)", "mod"),
                ("Circle fit per slice\n(≈0.3 m)", "mod"),
                ("Scale from the bore:\none scale or per slice", "mod"),
                ("Pipe coordinates (s, θ, r)\nθ = 0 above the camera", "mod"),
                ("Parts: aligned along the\naxis by the counter", "mod")],
             "metric 3-D model"),
            ("Evaluation & figures", "", False, [
                ("Travel vs counter", "eval"),
                ("Wall deviation, scatter,\ncircumferential coverage", "eval"),
                ("Model-to-model stretch\n(wall-brightness profiles)", "eval"),
                ("Unrolled wall maps,\ntextured 3-D views", "eval"),
                ("Views inside the model\nvs the video frames", "eval")],
             "accuracy metrics,\nfigures"),
        ],
        output="Output: ", calib_title="Intrinsic calibration (once, on video 1; incremental SfM re-run for each trial)",
        calib=("(1) scan k1 until the reconstructed pipe keeps a constant bore → k1 = −0.33 (no ground truth used)\n"
               "(2) tune the focal length until travel equals the counter → f = 806 px; principal point (383, 288) assumed"),
        to_calib="ground-truth\ntravel", from_calib="fixed intrinsics\n(all videos)",
        prior="Prior: pipe bore 300 mm", legend=[("data", "input data"), ("std", "standard OpenMVG / OpenMVS step"),
                                                 ("mod", "step added or changed for forward-looking in-pipe video"),
                                                 ("eval", "evaluation (not part of the reconstruction)")]),
}

W = 7.2                    # inches
M, GAP = 0.08, 0.2         # outer margin, gap between stage columns
FS, FS_HEAD, FS_SMALL = 6.3, 7.3, 6.0
LINE = FS * 1.30 / 72      # line height in inches
PAD, STEP_GAP = 0.05, 0.11


def box(ax, x, y, w, h, kind, text, font, fs=FS, weight="normal"):
    fill, edge, dashed = STYLE[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.035", fc=fill, ec=edge,
                                lw=0.6, ls="--" if dashed else "-"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, family=font, color=INK,
            weight=weight, linespacing=1.15)


def arrow(ax, p, q, lw=0.7, head=4.0, color="#444444", style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=f"{style},head_length={head * 0.12},head_width={head * 0.07}",
                                 mutation_scale=10, lw=lw, color=color, shrinkA=0, shrinkB=0))


def draw(lang, out_dir):
    t = TEXT[lang]
    n = len(t["stages"])
    cw = (W - 2 * M - (n - 1) * GAP) / n
    bw = cw - 0.12
    heights = [[2 * PAD + (s.count("\n") + 1) * LINE for s, _ in st[3]] for st in t["stages"]]
    head_h = 0.34
    out_h = 2 * PAD + max(st[4].count("\n") + 1 for st in t["stages"]) * LINE
    col_h = max(head_h + 0.1 + sum(h) + STEP_GAP * (len(h) - 1) + 0.16 + out_h + 0.08 for h in heights)
    calib_h = 2 * PAD + 3 * LINE + 0.06
    legend_h = 0.2
    H = M + col_h + 0.42 + calib_h + 0.12 + legend_h + M
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W), ax.set_ylim(0, H), ax.axis("off")
    top = H - M
    bottom = top - col_h
    for k, (title, tool, seq, steps, output) in enumerate(t["stages"]):
        x = M + k * (cw + GAP)
        ax.add_patch(FancyBboxPatch((x, bottom), cw, col_h, boxstyle="round,pad=0,rounding_size=0.05",
                                    fc="#F4F6F8", ec="#B8C0C8", lw=0.6))
        ax.add_patch(FancyBboxPatch((x, top - head_h), cw, head_h, boxstyle="round,pad=0,rounding_size=0.05",
                                    fc=HEAD[k], ec=HEAD[k], lw=0.6))
        label = f"({chr(97 + k)}) {title}" + (f"\n{tool}" if tool else "")
        ax.text(x + cw / 2, top - head_h / 2, label, ha="center", va="center", fontsize=FS_HEAD if "\n" not in label else FS_HEAD - 0.4,
                family=t["head_font"], weight="bold", color="white", linespacing=1.1)
        y = top - head_h - 0.1
        bx = x + (cw - bw) / 2
        for j, ((text, kind), h) in enumerate(zip(steps, heights[k])):
            box(ax, bx, y - h, bw, h, kind, text, t["font"])
            if j + 1 < len(steps) and seq:
                arrow(ax, (x + cw / 2, y - h), (x + cw / 2, y - h - STEP_GAP), head=3.2)
            y -= h + STEP_GAP
        # output of the stage, pinned to the bottom of the column
        oy = bottom + 0.08
        ax.add_patch(FancyBboxPatch((bx, oy), bw, out_h, boxstyle="round,pad=0,rounding_size=0.035",
                                    fc="white", ec=HEAD[k], lw=1.0))
        ax.text(bx + bw / 2, oy + out_h / 2, t["output"] + output, ha="center", va="center", fontsize=FS - 0.5, linespacing=1.15,
                family=t["head_font"], weight="bold", color=HEAD[k])
        arrow(ax, (x + cw / 2, y + STEP_GAP), (x + cw / 2, oy + out_h), head=3.2, color="#888888", style="-|>")
        if k + 1 < n:
            ym = bottom + col_h * 0.52
            arrow(ax, (x + cw + 0.02, ym), (x + cw + GAP - 0.02, ym), lw=1.6, head=6)

    # calibration (under the first three columns) and the bore prior (under the scale column)
    cy = bottom - 0.42 - calib_h
    cx0, cx1 = M, M + 3 * cw + 2 * GAP
    ax.add_patch(FancyBboxPatch((cx0, cy), cx1 - cx0, calib_h, boxstyle="round,pad=0,rounding_size=0.04",
                                fc="#FBE3C8", ec="#C9731E", lw=0.7, ls="--"))
    ax.text(cx0 + 0.08, cy + calib_h - PAD - 0.01, t["calib_title"], ha="left", va="top", fontsize=FS,
            family=t["head_font"], weight="bold", color=INK)
    ax.text(cx0 + 0.08, cy + PAD + 0.02, t["calib"], ha="left", va="bottom", fontsize=FS_SMALL, family=t["font"],
            color=INK, linespacing=1.25)
    xa, xb = M + cw * 0.5, M + cw + GAP + cw * 0.5
    arrow(ax, (xa, bottom), (xa, cy + calib_h), head=3.5, color="#5B8A4A")
    ax.text(xa + 0.04, (bottom + cy + calib_h) / 2, t["to_calib"], ha="left", va="center", fontsize=FS_SMALL - 0.3,
            family=t["font"], color="#5B8A4A", linespacing=1.1)
    arrow(ax, (xb, cy + calib_h), (xb, bottom), head=3.5, color="#C9731E")
    ax.text(xb + 0.04, (bottom + cy + calib_h) / 2, t["from_calib"], ha="left", va="center", fontsize=FS_SMALL - 0.3,
            family=t["font"], color="#C9731E", linespacing=1.1)
    px = M + 3 * (cw + GAP)
    ph = 2 * PAD + LINE
    py = cy + calib_h - ph
    box(ax, px + (cw - bw) / 2, py, bw, ph, "data", t["prior"], t["font"])
    arrow(ax, (px + cw / 2, py + ph), (px + cw / 2, bottom), head=3.5, color="#4A6688")

    # legend
    lx, ly = M, M + 0.02
    for kind, label in t["legend"]:
        fill, edge, _ = STYLE[kind]
        ax.add_patch(FancyBboxPatch((lx, ly), 0.2, 0.12, boxstyle="round,pad=0,rounding_size=0.02", fc=fill, ec=edge, lw=0.6))
        txt = ax.text(lx + 0.26, ly + 0.06, label, ha="left", va="center", fontsize=FS_SMALL, family=t["font"], color=INK)
        fig.canvas.draw()
        bb = txt.get_window_extent().transformed(ax.transData.inverted())
        lx = bb.x1 + 0.22

    base = os.path.join(out_dir, f"pipeline_flowchart_{lang}")
    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 600})):
        fig.savefig(f"{base}.{ext}", facecolor="white", **kw)
    plt.close(fig)
    print("saved", base + ".pdf/.svg/.png", f"({W:.1f} x {H:.2f} in)")


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for lang in ("zh", "en"):
        draw(lang, out)
