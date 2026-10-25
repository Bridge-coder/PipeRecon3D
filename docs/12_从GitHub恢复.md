# 12 从 GitHub 恢复（本机文件全部删掉之后）

本文件回答：只剩 GitHub 上的仓库和 Releases 时，怎样拿回 v2.0 的三维重建结果，怎样重新评分（复现 docs/11 的数字），怎样重新训练。
第 9 节是 2026-10-26 在本机做的恢复测试：全部只用 GitHub 上的文件，把 Pipe1 的场景重新导出，再给下载下来的模型重新评分。

## 1. 先看要做到哪一层

| 层次 | 能得到什么 | 需要下载 | 需要的环境 | 验证情况 |
|---|---|---|---|---|
| A. 看结果 | 4 个高斯模型、3 个烘焙网格、对比视频、展示图、全部评分 | Release `v2.0-results` | 烘焙网格：MeshLab / Blender；高斯模型：B 的环境，或 `export_ply.py` 转成 .ply 后用 SuperSplat 等查看器打开 | 文件已按 SHA-256 核对 |
| B. 重新评分 | 用下载的模型重算 docs/11 的表（同一批帧、同一批像素） | A + v2.0 的中间文件 + v1.0 Release 的工作目录 + 原始视频 | Windows Python + OpenMVG（导出场景）；WSL 发行版 pipe3d + gsplat（评分） | Pipe1 已测，见第 9 节 |
| C. 重新训练 | 从头训练出同样配置的模型和烘焙网格 | 同 B | 同 B；作者用 RTX 5060 Ti，每个模型训练 28–55 分钟，显存峰值 2.1–4.3 GB | 未重训；分数会有随机浮动（第 8 节） |
| D. 中间文件也从头做 | 管壁、中心线、灯光图、v1.0 的 SfM / OpenMVS 模型全部重算 | 只要原始数据 | 同 B + v1.0 全流程（README 第 3 节） | 未测；结果不会逐位相同（第 8 节） |

## 2. 代码

```bash
git clone https://github.com/Bridge-coder/PipeRecon3D.git
cd PipeRecon3D
git checkout explore          # v2.0 的代码（pipesplat/、docs/06-12）在 explore 分支；master 是 v1.0
```

## 3. 环境

**Windows 侧**（导出场景、对比表、管外视图；README 第 2 节）：

```bash
pip install -r requirements.txt
python tools/install_third_party.py      # OpenMVG 1.6 + OpenMVS 2.4.0，校验后解压到 third_party/
```

**WSL 侧**（训练、评分、烘焙；docs/07 第 4 节有每一步的下载量和原因）：

1. `C:\Users\<你>\.wslconfig` 给 WSL 足够内存，作者用的是 `memory=8GB`、`swap=8GB`（docs/07 第 5 节）。
2. 建发行版并第一次启动（启动时自己设 Linux 用户名和密码）：

   ```bash
   wsl --install Ubuntu-24.04 --name pipe3d --location D:\wsl\pipe3d --no-launch
   wsl -d pipe3d
   ```

3. 依次运行 4 个安装脚本（以 root；第 4 个是 CPU 重任务，单独跑）：

   ```bash
   wsl -d pipe3d -u root -- bash /mnt/<盘>/<路径>/PipeRecon3D/pipesplat/setup/01a_system_packages.sh
   wsl -d pipe3d -u root -- bash /mnt/<盘>/<路径>/PipeRecon3D/pipesplat/setup/01b_cuda_toolkit.sh
   wsl -d pipe3d -u root -- bash /mnt/<盘>/<路径>/PipeRecon3D/pipesplat/setup/02_python_env.sh
   wsl -d pipe3d -u root -- bash /mnt/<盘>/<路径>/PipeRecon3D/pipesplat/setup/03_build_gsplat.sh
   ```

   得到 `/opt/gsenv`（Python 3.12、torch 2.9.1 + CUDA 12.8、gsplat 1.6.0 固定在 commit `512d366`）。
   **换了显卡**：`03_build_gsplat.sh` 里的 `TORCH_CUDA_ARCH_LIST="12.0"` 只适用于 RTX 50 系列，按你的显卡改（例如 RTX 40 系列是 `8.9`）。
4. 评分要 LPIPS 的 AlexNet 权重，第一次运行时会自动下载到 `TORCH_CACHE`（需要联网）。

## 4. 路径配置

所有脚本都从 `scripts/paths.py` 取位置，不写死盘符。三层，后面的覆盖前面的：

1. 默认值：全部在仓库里，例如 `GS_ROOT = <仓库>/work/pipe3d_gs`；
2. `config/paths.json`：复制 `config/paths.example.json` 后修改，只改需要的键（这个文件不进 git）；
3. 环境变量 `PIPE3D_<键>`，例如 `PIPE3D_GS_ROOT=E:/gs`。

WSL 里读的是同一个文件，`D:/pipe3d_gs` 这类 Windows 路径会自动换成 `/mnt/d/pipe3d_gs`。
WSL 的 shell 脚本通过 `pipesplat/env.sh` 取这些路径。本次用到的键：

| 键 | 放什么 | 作者机器上的值 |
|---|---|---|
| `GS_ROOT` | 高斯这条线的工作区：`g0/<数据>/` 场景和各次训练、`review/`、`logs/` | `D:/pipe3d_gs` |
| `TORCH_CACHE` | LPIPS 权重缓存（`TORCH_HOME`） | `D:/pipe3d_gs/torch_cache` |
| `WORK_SHEFFIELD` | Sheffield 的 v1.0 工作目录（`concrete_2/` 等） | `D:/pipe3d` |
| `WORK_NEW` | 新视频测试的 v1.0 工作目录（`concrete_lab/`、`sewer_pipe1/work/`） | `D:/pipe3d_new` |
| `INPIPE_EXTRACT`、`INPIPE_VISION`、`RECON_BIG` 等 | InPipe 融合流程（README 第 3.2 节） | `D:/fusion/...` |
| `MVG_BIN`、`MVS_BIN` | OpenMVG / OpenMVS（默认就是 `install_third_party.py` 解压的位置） | — |

检查：`python scripts/paths.py`（每个键的值和是否存在）。

## 5. 下载并放到位

```bash
gh release download v2.0-results --repo Bridge-coder/PipeRecon3D --dir D:/dl/v2.0
gh release download v1.0-results --repo Bridge-coder/PipeRecon3D --dir D:/dl/v1.0 -p sheffield_concrete_2.zip -p concrete_lab.zip -p sewer_pipe1.zip -p inpipe_172011_vision_model.zip
```

下载后先对照各自的 `SHA256SUMS.txt` 校验（Windows：`certutil -hashfile 文件 SHA256`）。原始视频按 [05_数据下载.md](05_数据下载.md) 下载并核对 MD5。

| 文件 | 解压到 | 解压后 |
|---|---|---|
| `lamp_maps.zip`（v2.0） | 仓库根目录 | `lampmap/results/<数据>/lamp_map.npy`（v1.0 灯光图；仓库里 git 忽略了 .npy） |
| `<数据>_intermediates.zip`（v2.0） | `GS_ROOT` | `g0/<数据>/` 下的管壁、中心线、网格、导出记录 |
| `<数据>_gaussians_*.zip`（v2.0） | `GS_ROOT/g0/<场景>/` | 再把解出的文件夹改名为训练名：concrete_2、concrete_lab → `run_g0a`；InPipe → `g0/inpipe_vision/run_vhf`；Pipe1 → `run_vhf` |
| `<数据>_baked_mesh_*.zip`（v2.0） | 任意位置 | 直接用 MeshLab / Blender 打开 `model.obj` |
| `sheffield_concrete_2.zip`（v1.0） | `WORK_SHEFFIELD` | 把解出的 `sheffield_concrete_2/` 改名为 `concrete_2/` |
| `concrete_lab.zip`（v1.0） | `WORK_NEW` | `concrete_lab/` |
| `sewer_pipe1.zip`（v1.0） | `WORK_NEW/sewer_pipe1/` | 把解出的 `sewer_pipe1/` 改名为 `work/`，即 `WORK_NEW/sewer_pipe1/work/` |
| `inpipe_172011_vision_model.zip`（v1.0） | 任意位置，再把 `INPIPE_VISION` 指过去 | InPipe 的 v1.0 视觉模型（SfM 位姿 + OpenMVS 网格）：高斯模型是在它的坐标系里训练的 |

## 6. 重新导出场景（scene.npz：训练图像、留出帧、B 渲染）

scene.npz 带图像，每段约 0.4–1 GB，所以不上传，由原视频 + v1.0 工作目录重新导出（Windows，在仓库根目录；`<...>` 换成你的路径）：

```bash
# concrete_2（Sheffield 5 m 混凝土管第 2 段）
python -I pipesplat/export_g0.py vision <WORK_SHEFFIELD>/concrete_2 <视频>/300mm_concrete_5meters_ON_2lm_2u_CLOSED_2.mp4 <GS_ROOT>/g0/concrete_2 --lamp lampmap/results/concrete_2/lamp_map.npy
# 实验室管 A（15 m 管网视频的开头两节混凝土管）
powershell -ExecutionPolicy Bypass -File pipesplat/review/export_lab.ps1 -Video <视频>/300mm_mix_15meters_ON_3lm_3u_CLOSED_1_stop_at_14meters.mp4
# Pipe1（PowerShell 里先 $env:PIPE3D_PIPE_DIAMETER_M='0.435'）
python -I pipesplat/export_g0.py vision <WORK_NEW>/sewer_pipe1/work <视频>/Pipe1.avi <GS_ROOT>/g0/sewer_pipe1 --lamp <GS_ROOT>/g0/sewer_pipe1/lamp_map.npy --plain --crop 0 0 720 490 --stretch-x 1.0159
```

`export_lab.ps1` 也会运行 `export_surface.py`；它写出的 `surface.npz` 与中间文件包里的那份由同一个网格算出。
Pipe1 用中间文件包里的 `lamp_map.npy`，不用 `--lamp auto`：作者导出时就是自动学出的这一张，B 的渲染（以及评分用的像素）才与 docs/11 一致。

**InPipe** 要先把 v1.0 的激光 + 视觉融合流程跑到 R4（原始数据 10 GB，README 第 3.2 节），这一步**未做恢复测试**：

```bash
# INPIPE_VISION 指向解压的 v1.0 视觉模型，跳过 vision 阶段（不重做 SfM，坐标系才与训练时相同）
python scripts/run_inpipe.py --zip <原始数据>/realsense_slam_20260312_172011.zip extract lidar lidar_short tags vision_eval f1 legs f2 f3 f3_icp f4 f5 r1 r2 r3 r4
python -I pipesplat/export_g0.py inpipe <GS_ROOT>/g0/inpipe_172011 --lamp lampmap/results/inpipe_172011/lamp_map.npy --width 640
python -I pipesplat/export_inpipe_vision.py <GS_ROOT>/g0/inpipe_172011 <GS_ROOT>/g0/inpipe_vision
```

融合各阶段会重新计算（外参 `extrinsic_translation_final.json` 在仓库里），B 的渲染可能与作者的有细微差别，所以 InPipe 的 B 分数和评分像素不保证与 docs/11 完全相同。
导出后，再把中间文件包里的 `surface_vhf.npz`、`mesh_vhf.npz`、`centerline_vhf.npz` 放回 `g0/inpipe_vision/`：导出脚本会覆盖同名的 `surface.npz`，但不会覆盖这三个。

## 7. 重新评分、重新训练

**重新评分**（WSL；每个模型按它自己 summary 里的参数评分，打印新旧对比）：

```bash
wsl -d pipe3d -u root -- /opt/gsenv/bin/python /mnt/<盘>/<路径>/PipeRecon3D/pipesplat/rescore.py concrete_2/run_g0a concrete_lab/run_g0a inpipe_vision/run_vhf sewer_pipe1/run_vhf
```

**重新训练**（WSL；一次只跑一个；换个训练名，免得覆盖下载的模型）：

| 模型 | 命令（在 `pipesplat/review/` 下，`run_f2.sh 数据 训练名 选项`） |
|---|---|
| concrete_2 G0-A | `bash run_f2.sh concrete_2 run_g0a_new --anchor` |
| 实验室管 A G0-A | `bash run_f2.sh concrete_lab run_g0a_new --anchor --points-every 2` |
| InPipe Y1 | `bash run_f2.sh inpipe_vision run_vhf_new --anchor --lamp --lamp-ambient --lamp-smooth-mm 20 --surface surface_vhf.npz` |
| Pipe1 Y3 | `bash run_f2.sh sewer_pipe1 run_vhf_new --anchor --surface surface_vhf.npz` |

训练结束会自动按量尺 v2 评分（`summary_r2.json`）。**烘焙网格**：InPipe、Pipe1 用 `pipesplat/bake_vhf.py`（命令见 docs/11 第 9 节），concrete_2 用 `pipesplat/review/bake_x8b.sh allwall`。

**中间文件也重做**（第 1 节的 D 层）：管壁 `vision_surface.py`（参数见 docs/11 第 9 节）；concrete_2 / 实验室管 / Pipe1 的 `surface.npz` 用 `export_surface.py`；InPipe 的激光中心线和 R3 管壁来自融合流程（docs/09–10）。

## 8. 不能逐位复现的地方

- **重新训练**：高斯训练有随机性（GPU 上并行求和的顺序等），重训出的模型不会与下载的逐位相同，分数会有小幅浮动。
  本项目没有系统测过重训的浮动有多大；要复现 docs/11 的数字，请用下载的模型重新评分（第 9 节：与表中数字完全一致）。
- **InPipe 的融合流程**：从原始数据重算，B 的渲染和评分像素可能与作者的有细微差别（第 6 节）。
- **换显卡或换 gsplat 版本**：渲染的浮点结果可能有极小差别。gsplat 版本固定在 commit `512d366`。
- **不在 GitHub 上的东西**：
  - 框架流图的生成器（流图本身是 claude.ai 上的页面）；
  - 作者机器上被放弃的实验变体：没有进 docs/11 表格的那些训练；
  - 原始数据：按 docs/05 从原数据集网站下载。外部链接如果失效，就没法重做，建议另外备份一份原始视频。

## 9. 恢复测试记录（2026-10-26）

目的：只用 GitHub 上的文件和原始视频，能否重新得到 docs/11 里 Pipe1 Y3 的数字（第 1 节的 B 层）。
在本机新建空目录 `D:/restore_test`，用 `PIPE3D_GS_ROOT` 把 `GS_ROOT` 指过去，作者原来的工作区一个文件都不读。

| 步骤 | 输入 | 结果 |
|---|---|---|
| 1. 校验 | v1.0 `sewer_pipe1.zip`；v2.0 `sewer_pipe1_intermediates.zip`、`sewer_pipe1_gaussians_y3.zip`；`Pipe1.avi` | 前三个的 SHA-256 与 GitHub 上的一致；视频 MD5 与 docs/05 一致 |
| 2. 放置 | 按第 5 节解压、改名 | `work_new/sewer_pipe1/work/`、`gs/g0/sewer_pipe1/`（含 `run_vhf/`） |
| 3. 重新导出场景 | 第 6 节的 Pipe1 命令（中间文件包里的 `lamp_map.npy`） | 用时约 3 分钟，scene.npz 1.01 GB；与作者的 scene.npz **逐个数组逐位相同**：训练图像、留出帧、下一帧、B 渲染、掩码、位姿、点云，共 25 个数组 |
| 4. 重新评分 | `rescore.py sewer_pipe1/run_vhf --suffix _restore` | 序列内 34.087 dB、整段留出 31.771 dB，B 24.080 / 22.624 dB，LPIPS 0.314 / 0.329，都与 docs/11 相同；`summary` 里 258 个数、逐帧表 55 帧 × 87 列，**没有一个不同** |

测试中发现并修好的一处：`rescore.py` 第一版没有传 `--alpha-min 0.5`。训练结束时按量尺 v2 评分用的是这个阈值，
它只影响"模型覆盖率"和"深度误差"两列（PSNR、LPIPS 等是在 B 的掩码上算的，不受影响）。第一版的这两列与表中不同，加上之后完全一致。

没有测的：
- concrete_2、实验室管 A 的场景导出和评分：同一套代码，输入换成对应的 v1.0 工作目录。
- InPipe：需要 10 GB 原始数据，并把融合流程从头跑一遍（第 6 节）。
- 重新训练（第 7 节）。
- 在一台全新的机器上装环境（第 3 节）：本次测试用的是本机已经装好的 WSL 和 OpenMVG。
