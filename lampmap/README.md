# lampmap：灯光图模型

**一句话**：重建出的管道模型从相机位姿渲染出来之后，再乘一张"固定在画面上的灯光图"。机器人的灯跟着相机走，在直管里它投在管壁上的明暗（连同镜头暗角）每一帧几乎相同。这张图只用建模帧学习，然后在**没参与建模的视频帧**上检验。

日期：2026-10-19。PipeRecon3D 探索阶段方向二留下来的部分；同期的其余尝试已经放弃，见第 5 节。

## 1. 效果（每段 44 张留出帧的中位数，和原视频比）

| 数据 | 模型 | 现有模型 PSNR / LPIPS | **× 灯光图** PSNR / LPIPS | 参照：视频自己的下一帧 |
|---|---|---|---|---|
| concrete_2（Sheffield 混凝土管） | OpenMVS 网格 + TextureMesh | 25.88 dB / 0.443 | **29.82 dB / 0.413** | 33.55 dB / 0.126 |
| 实验室混凝土管 A | OpenMVS 网格 + TextureMesh | 20.87 dB / 0.411 | **25.26 dB / 0.369** | 29.36 dB / 0.160 |
| InPipe 172011（激光+视觉） | R3 高度场 + R4 逐像素贴图 | 17.54 dB / 0.350 | **23.35 dB / 0.279** | 34.81 dB / 0.072 |

- PSNR 提高 3.9 / 4.4 / 5.8 dB，LPIPS 同时下降。平滑明暗（6 px 以上的低频）残差减少了 80–86 %，例如 concrete_2 从 115 降到 23。
- 贴图错位、细节相关性基本不变，因为灯光图只改明暗，不改几何和贴图。
- 对比图：`results/对比图_concrete_2.jpg`、`对比图_concrete_lab.jpg`、`对比图_inpipe_172011.jpg`。三格依次是原视频帧、现有模型、乘灯光图后。
- 每段数据的完整结果在 `results/<数据>/`：
  - `summary.json`、`per_frame.csv`：指标；
  - `lamp_map.npy` / `.png`：学到的灯光图；
  - `renders/`：每 4 张留出帧存一张三联图。
- 这三组数字已经用本文件夹的代码重跑确认，和探索时的结果完全一致（2026-10-19）。

## 2. 用法

前提：模型已经用仓库主流程建好。
- 视觉：`scripts/vision/run_pipeline.py` 跑完第 13 步；
- InPipe：`scripts/run_inpipe.py` 跑完 r4。

```bash
# 视觉模型（任意管道视频的 work 文件夹）
python -I lampmap/lampmap.py D:/pipe3d/concrete_2 <原视频.mp4> lampmap/results/concrete_2
# 非 Sheffield 视频：加上建模时用的 --plain --crop --stretch-x
# InPipe 激光+视觉模型（路径来自 config/paths.json）
python -I lampmap/lampmap_inpipe.py lampmap/results/inpipe_172011
```

**参数**：
- `--n 44`：留出帧数；
- `--train 40`：学灯光图用的建模帧数；
- `--ahead 3.0`：渲染相机前方多少米内的网格；
- `--save 4`：每几张留出帧存一张三联图。

**用时**（Ryzen 5 5600，单进程）：concrete_2 约 3.5 min，实验室管 A 约 5.5 min，InPipe 约 1 min。

**LPIPS**：要用装了 torch + lpips 的 Python 才算，例如 `D:/pipe3d_fm/envs/da3/Scripts/python.exe`，并设 `TORCH_HOME=D:/pipe3d_fm/torch_cache`。用仓库默认的 Python（只有 numpy / OpenCV）跑，LPIPS 记为 NaN，其余指标不受影响。

## 3. 文件

| 文件 | 内容 |
|---|---|
| `lampmap.py` | 核心：渲染器（OpenMVS 网格，按注册的焦距、主点和 k1 畸变）、`learn_lamp_map`（学灯光图）、`apply_lamp`（乘灯光图）、`evaluate`（留出帧对比）；视觉 work 文件夹的命令行入口 |
| `lampmap_inpipe.py` | InPipe 的入口：R3 网格 + R4 贴图，位姿 = 融合位姿 + R2b 校正，和 `scripts/recon/r5_compare.py` 一致 |
| `imgmetrics.py` | 指标：PSNR、SSIM、LPIPS、错位、细节相关性、细节能量比、低频/高频残差 |

**依赖的仓库代码**（只读调用）：`scripts/paths.py`；`scripts/vision/render_views.py`、`evaluate.py`、`video_compare.py`；InPipe 另需 `scripts/recon/recon_common.py`、`scripts/fusion/vision_model.py`。

## 4. 方法细节和适用范围

- **学灯光图**：每张建模帧从自己的注册位姿渲染，先做整体亮度匹配（每通道一个增益加一个偏移，相当于去掉这一帧的曝光），再逐像素算 log(视频帧 / 渲染)，多帧平均，6 px 高斯平滑；没有帧覆盖到的像素由邻近值补上。
- **用灯光图**：渲染 × exp(灯光图)，再配上这一帧自己的整体增益和偏移（相机的自动曝光）。留出帧上只拟合这几个参数。
- **适用范围**：要求直管、相机在管内的位置大致不变。弯管、管口、管节台阶、相机明显晃动的地方，灯光分布会变，残差会变大。如果要处理这些情况，需要换成物理灯光模型：1/r² × 入射角余弦 × 灯的角度分布 × 暗角，用网格深度逐像素计算。端盖这类光滑面上的镜面反光目前也不建模。
- **读指标的注意事项**：模拟视频噪声大，PSNR 和 SSIM 会奖励把渲染图模糊掉，所以要和 LPIPS、细节相关性一起看。"视频下一帧"带着和原视频同样的噪声，是一个偏乐观的参照。

## 5. 同期试过、没有采用的（2026-10-19，供以后参考）

- **前馈三维大模型直接替换 OpenMVS 稠密重建**：DA3-Base、DA3-Large-1.1、MapAnything。在 concrete_2 上，三者都把近处管壁推远了约 2 倍；统一定一次尺度后深度误差约 30 %，每帧用 2 个参数做最优校正后近壁仍差 15–27 %；给位姿、加大视差都无效。唯一可用的信号是：DA3-Large 只看图像估出的焦距 805 px，和标定值 806 一致。
- **用原始帧平均重做贴图**（反照率 × 灯光图）：concrete_2 上只比本方法多 1.2 dB，实验室管 A 上多 0.7 dB；但贴图变糊（细节能量只有视频的 11–20 %），LPIPS 反而更差，实验室管 A 上还出现环状条纹，端盖有反光亮斑。
  - 原因 1：多帧之间约 1 px 的配准误差，平均之后把细节抹掉了；
  - 原因 2：OpenMVS 图集是很多 10–30 px 的小块，逐纹素融合会在块间串色；
  - 原因 3：每帧画面边缘留下台阶，形成周期约 1 帧行程（11.7 mm）的环。
  - 要做好，需要逐帧位姿精修或可微渲染，投入大，暂停。
