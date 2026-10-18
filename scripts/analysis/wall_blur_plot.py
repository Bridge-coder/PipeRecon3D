"""Before / after figure for the LiDAR wall blur diagnosis (reads DIAG_BIG/wall_blur_moving.npz)."""
import os, sys, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import paths
OUTPNG = os.path.join(paths.INPIPE_RESULTS, "diag", "wall_blur_before_after.png")
z = np.load(os.path.join(paths.DIAG_BIG, "wall_blur_moving.npz"))
s, th, r, ring, rho, scan = (z[k] for k in ("s", "th", "r", "ring", "rho", "scan"))
b_ring, b_rng, rbins = z["b_ring"], z["b_rng"], z["rbins"]
odd = scan % 2 == 1  # bias was calibrated on even scans; show only the held-out odd scans
ib = np.clip(np.digitize(rho, rbins) - 1, 0, len(rbins) - 1)
r_corr = r - b_ring[ring] - b_rng[ib]
band = (rho > 0.25) & (rho < 0.45)
before = odd
after = odd & band

S0, S1 = 0.25, 0.65
def unrolled(m, rr):
    iS = ((s[m] - S0) / 0.01).astype(int)
    iT = ((th[m] + 180) / 4).astype(int)
    ns, nt = int((S1 - S0) / 0.01), 90
    ok = (iS >= 0) & (iS < ns) & (iT >= 0) & (iT < nt)
    img = np.full((nt, ns), np.nan)
    key = iT[ok] * ns + iS[ok]
    order = np.argsort(key, kind="stable")
    ks, vs = key[order], rr[m][ok][order]
    u, st, cnt = np.unique(ks, return_index=True, return_counts=True)
    for k, a, n in zip(u, st, cnt):
        if n >= 5:
            img.flat[k] = np.median(vs[a:a + n])
    with warnings.catch_warnings():  # angle rows without any cell -> all-NaN, fine
        warnings.simplefilter("ignore", RuntimeWarning)
        img -= np.nanmedian(img, axis=1, keepdims=True)  # remove the average cross-section (ellipse)
    return img

fig, ax = plt.subplots(2, 2, figsize=(13, 9.5), gridspec_kw={"width_ratios": [1, 2.2]})
SL = (0.45, 0.46)
for k, (m, rr, title) in enumerate([
        (before, r, "现在：0.2–0.6 m 全部点，未做激光线/距离偏差校正"),
        (after, r_corr, "改进后：每条激光线 + 距离偏差自标定，只用 0.25–0.45 m")]):
    sl = m & (s > SL[0]) & (s < SL[1])
    t = np.radians(th[sl])
    x, y = 1000 * rr[sl] * np.sin(t), 1000 * rr[sl] * np.cos(t)
    a = ax[k, 0]
    a.scatter(x, y, s=0.3, c="k", alpha=0.25, linewidths=0)
    res = rr[sl]
    # thickness: p5-p95 of r minus per-4deg median
    it = ((th[sl] + 180) / 4).astype(int)
    med = np.array([np.median(res[it == q]) if (it == q).sum() else np.nan for q in range(90)])
    e = 1000 * (res - med[it])
    p5, p95 = np.percentile(e, [5, 95])
    a.set_title(f"{title}\n横截面 s = {SL[0]:.2f}–{SL[1]:.2f} m（厚 1 cm），{sl.sum():,} 点\n管壁厚度（5–95%）{p95 - p5:.1f} mm", fontsize=9.5)
    a.set_aspect("equal"); a.set_xlim(-125, 125); a.set_ylim(-125, 125)
    a.set_xlabel("mm"); a.grid(alpha=0.3)
    img = unrolled(m, rr) * 1000
    b = ax[k, 1]
    im = b.imshow(img, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-5, vmax=5,
                  extent=[S0, S1, -180, 180])
    b.set_xlabel("沿管位置 s（m，编码器）"); b.set_ylabel("转角（°，0 = 顶部，±180 = 底缝）")
    b.set_title("展开图：每 1 cm × 4° 小格的中位半径，减去平均截面（±5 mm）", fontsize=9.5)
    plt.colorbar(im, ax=b, label="半径残差 mm", fraction=0.04)
fig.suptitle("雷达管壁为什么糊：系统偏差（每条激光线、近距离）远大于随机噪声 —— 第一节管，机器人行驶中 218 帧，只显示未参与标定的奇数帧", fontsize=11)
fig.tight_layout()
fig.savefig(OUTPNG, dpi=110)
print("->", OUTPNG)
