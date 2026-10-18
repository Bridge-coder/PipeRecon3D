"""R5b: close-ups from the full-size render triplets of r5_compare.py (original | before | now, 1280 x 720 each).

output: results/inpipe_172011/recon/r5_detail.png
usage: python -I r5b_detail.py
"""
import os
import sys

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recon_common import BIG, OUT  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
CROPS = {1640: (250, 230, 900, 690, "第二节管口（插在第一节里）的端面、壁上小孔、底缝"),
         840: (280, 120, 880, 560, "远处的两节管接口")}
fig, ax = plt.subplots(len(CROPS), 3, figsize=(19, 5.6 * len(CROPS)))
for i, (idx, (x0, y0, x1, y1, what)) in enumerate(CROPS.items()):
    trip = cv2.imdecode(np.fromfile(os.path.join(BIG, "viz", f"triplet_{idx:06d}.jpg"), np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
    for j, name in enumerate(("原始视频帧", "之前：雷达点 + 相机颜色", "现在：高度场网格 + 逐像素贴图")):
        ax[i, j].imshow(trip[y0:y1, j * 1280 + x0:j * 1280 + x1])
        ax[i, j].set_title(f"{name}（彩色帧 {idx}，局部放大）", fontsize=12, loc="left")
        ax[i, j].axis("off")
    ax[i, 0].text(0, 1.10, what, transform=ax[i, 0].transAxes, fontsize=13)
fig.suptitle("同一相机位姿下的局部放大：现在的贴图来自图像像素，不再受雷达点间距和测距误差限制", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "r5_detail.png"), dpi=90)
print("->", os.path.join(OUT, "r5_detail.png"))
