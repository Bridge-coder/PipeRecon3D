# 混凝土管第 1 段视频：OpenMVG + OpenMVS 复现结果小结

日期：2026-10-13（标定内参后的版本）
数据：`data\sheffield_concrete_5m\300mm_concrete_5meters_ON_2lm_2u_CLOSED_1.mp4`（720×576，25 帧/秒，95 秒）
工具：OpenMVG 1.6、OpenMVS 2.4.0（均为官方 Windows 预编译包，仅用 CPU）
工作目录：`D:\pipe3d\concrete_1\`；脚本：`scripts\run_pipeline.py`；日志：`scripts\logs\concrete_1\`

## 当天的两次修正

1. **主点**：OpenMVG 1.6 写出的主点被 OpenMVS 2.4.0 读成 (766.5, 671.5) 像素，正确值是 (383, 288)。修正前稠密点云、网格、纹理全部无效。当时写的"相机只前进、基线太小"这个原因是错的。
2. **内参**：原来焦距是假设的 740 像素、畸变为 0，重建出的管道是锥形（管径首尾比值 0.55），行走距离偏短 18.6%。标定为焦距 806、径向畸变 k1 = −0.33 后，管道是等径圆柱。标定过程和在其余四段视频上的检验见 `..\calibration\标定与检验.md`。

旧输出 `_wrong_pp\`（主点错误时的）和 `_before_calibration\`（标定前的）已于 2026-10-17 清理时移到 `D:\_清理归档_20261017\`。

## 结论

| 阶段 | 结果 |
|---|---|
| SfM（稀疏重建、相机轨迹） | 477 帧注册 476 帧，管径首尾比值 1.03，管壁离散 4.6 mm |
| 稠密点云 | 193 万点，96% 在管壁 ±15% 内，管壁离散 2–4 mm，全程覆盖整圈 |
| 网格 | 68 万顶点、136 万个面，97% 的顶点在管壁 ±15% 内 |
| 纹理 | 8192 像素纹理；2026-10-14 关掉接缝匀色后重做（原先 57% 的面是黑的，见 `..\concrete_2\结果小结.md` 的"纹理修正"） |

这一段参与了焦距标定，所以它的行走距离误差（+0.2%）不是独立结果。独立结果是第 2–5 段的 −0.6% 至 −5.0%。

## 实际运行的命令

```text
[2]   extract_frames.py VIDEO images --every 5 --clahe          每 5 帧取 1 帧（477 帧），对比度增强
[3]   openMVG_main_SfMInit_ImageListing -c 2 -k "806;0;383;0;806;288;0;0;1"
[3b]  set_distortion.py sfm_data.json -0.33                      写入径向畸变
[4]   openMVG_main_ComputeFeatures -m SIFT -p ULTRA
[5]   openMVG_main_ComputeMatches -g f -v 25                     配对+匹配+几何过滤（1.6 版合在一个程序里）
[7]   pick_pair.py                                               给候选初始图像对排序
[8]   openMVG_main_IncrementalSfM -f NONE -a frame_000880.png -b frame_000910.png
[9]   openMVG_main_ComputeSfM_DataColor
[10]  openMVG_main_openMVG2openMVS -o scene_openmvg.mvs
[10b] fix_mvs_k.py scene_openmvg.mvs scene.mvs 768 548           修正主点的存储形式
[11]  DensifyPointCloud --dense-config-file Densify.ini --resolution-level 0
[12]  ReconstructMesh
[13]  TextureMesh --export-type obj
```

第 2、4、5 步沿用了上午的结果（抽帧、特征、匹配与内参无关），没有重跑。

每处设置的原因：

- OpenMVG 用 1.6：2.0 的预编译包含 AVX-512 指令，本机 CPU（Ryzen 5 5600）不支持。
- 对比度增强、ULTRA、每 5 帧取 1 帧：原始设置下匹配点太少、视差太小，SfM 无法初始化。
- `-f NONE` 固定内参：相机只沿光轴前进，图像定不出焦距；放开优化时焦距发散到 5000 像素以上。
- 焦距 806、k1 = −0.33：由"管道等径"和"行走距离等于计数器"两个条件在本段视频上标定。
- 初始图像对：OpenMVG 自己选时有时选到平面的端盖上而失败。现在由 `pick_pair.py` 排出候选，第 8 步依次尝试，直到注册 90% 以上的帧。本段第 1 个候选（端盖附近）只注册了 2 帧，第 2 个候选成功。
- `fix_mvs_k.py`：OpenMVG 1.6 把主点存成"像素 ÷ 长边"，OpenMVS 2.4.0 按"(像素 + 0.5) ÷ 长边 − 0.5"来读，两者差半幅图像。脚本只改这两个数。
- `Densify.ini`：把 Optim Angle 12→3、Min Angle 3→0.5、View Min Score 2→0。用纯默认值时 388 帧"配到 0 个邻居"，只出 13 张深度图。其余稠密参数用 OpenMVS 默认值。

## 数值结果

真值：画面里烧录的里程计数器，逐帧读出（0.00 → 4.18 m）。计数器在机器人起步约 3.5 秒后才开始走，所以只比较计数器 0.25 m 之后的区间，共 3.93 m。

| 项目 | 标定前（740，无畸变） | 标定后（806，k1 = −0.33） |
|---|---|---|
| 注册帧数 | 476 / 477 | 476 / 477 |
| 稀疏点数 | 14717 | 15214 |
| 重投影误差（平均 / 中位数） | 0.56 / 0.39 像素 | 0.55 / 0.39 像素 |
| 管壁点离散（稀疏点） | 6.5 mm RMS | 4.6 mm RMS |
| 管径首尾比值 | 0.55 | 1.03 |
| 行走距离，全程一个比例 | 3.20 m（−18.6%） | 3.94 m（+0.2%，非独立） |
| 行走距离，逐段修正比例 | 3.45 m（−12.1%） | 3.93 m（+0.1%，非独立） |
| 稠密点云 | 178 万点，96% 在管壁 ±15% 内 | 193 万点，96% 在管壁 ±15% 内 |
| 稠密点的管壁离散 | 2.3–4.9 mm | 1.8–4.1 mm（靠近端盖的最后一段 11.6 mm） |
| 圆周覆盖 | 1.2 m 之前 78%–92%，之后 100% | 全程 100% |
| 网格 | 60 万顶点、121 万面，97% 在管壁上 | 68 万顶点、136 万面，97% 在管壁上 |
| 端盖位置 | 最后一个相机位置之后 0.47 m | 最后一个相机位置之后 0.56 m |

稠密点云和网格的"在管壁上"是相对 SfM 稀疏模型的管壁量的。标定后稀疏模型本身是等径的，所以全程一个比例换算出的模型就是直径约 300 mm 的圆柱。

耗时：稠密 41 分钟（同时有其他任务在跑），网格 1 分钟，纹理 8 分钟。

## 文件

- `sparse_points_and_cameras.ply`：稀疏点云和相机轨迹
- `sparse_eval.png`、`sparse_eval.json`、`sparse_eval_travel.csv`：稀疏重建的评估
- `dense_eval.png`、`dense_eval.json`：稠密点云相对管壁的评估
- `mesh_untextured.ply`、`mesh_eval.png`、`mesh_eval.json`：网格及其评估
- `textured_model\scene_textured.obj`：带纹理的模型（167 MB，2026-10-14 换成关掉接缝匀色后重做的版本）
- `views\`：三维视图（2026-10-14），`model_views.png` 是整页：管外透视、剖开露出内壁、按比例的侧视图（管节 1.78–1.92 m，端盖前的环 4.65 m）、0.81/1.90/3.30 m 处的模型内部视图与对应视频帧。出图方法见 `..\concrete_2\结果小结.md` 的"三维视图"
- 稠密点云（118 MB）留在 `D:\pipe3d\concrete_1\mvs\scene_dense.ply`
- `_before_calibration\`、`_wrong_pp\`：旧结果，2026-10-17 已移到 `D:\_清理归档_20261017\`
- 参数探索的对比表在 `..\exploration\variants.csv`
