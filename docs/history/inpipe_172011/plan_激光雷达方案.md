# 第一步：激光雷达方案（2026-10-14）

原则：本文件夹只做多模态融合。`运动估计三维重建\` 里的脚本和结果只读取、不修改；
大文件（ROS bag、点云、网格）放 D:（C: 只剩约 12 GB）。

---

## 0. 为什么要融合：两种传感器正好互补

| | 视觉（现有 OpenMVG + OpenMVS） | 激光雷达 |
|---|---|---|
| 沿管轴方向的运动 | 靠壁面纹理能约束 | **退化**：直管里截面处处相同，沿轴平移看不出来 |
| 绕管轴的滚转 | 靠纹理能约束 | 正圆管里退化 |
| 公制尺度 | 没有；目前靠"管径 = 300 mm"假设 + 计数器标定焦距 | 直接给（测距本身是公制） |
| 截面形状、半径 | 半径沿管有 ±5% 抖动，不能测变形 | 直接给，可测椭圆度、变形、积水线 |
| 颜色/纹理/缺陷外观 | 有 | 只有反射强度 |

一句话：**视觉管轴向，激光管径向和尺度**。融合后可以去掉"已知管径"这个假设，
并把现有模型的局部尺度抖动（各视频间相差约 9%，按管径局部校正后 ≤3.4%）用激光测得的真实半径校正掉。

---

## 1. 先定传感器形态（决定后面所有处理）

| 形态 | 输出 | 适用 | 要点 |
|---|---|---|---|
| **2D 激光雷达 / 激光环轮廓仪**，扫描面垂直管轴 | 每帧一个截面轮廓 | 小管（如 300 mm） | 自己不知道沿轴走了多远，轴向位置要靠编码器/计数器/视觉位姿；InPipe-SLAM 数据集就是这种（`/scan`） |
| 3D 激光雷达（机械式 / 固态） | 每帧一片 3D 点云 | 大管、箱涵 | 近距盲区：300 mm 管的壁距中心只有 0.15 m，大多数 3D 雷达打不到，要查规格书 minimum range |
| RGB-D / ToF 深度相机 | 深度图 | 中小管 | InPipe-SLAM 也带（RealSense），可作对照 |
| 激光环 + 现有相机（结构光） | 图像里的亮环 → 三角测量截面 | 小管 | 不是严格意义的雷达，但和视觉天然同步、同外参 |

选型时看两项：**最小量程**必须明显小于管半径；**测距精度**在 150 mm 半径上，毫米级和厘米级差别很大。

---

## 2. 激光数据怎么处理（以 2D 雷达为主线）

1. **读取**：ROS 2 bag（`.db3`）用纯 Python 的 `rosbags` 库读取，不需要装 ROS。
   每帧导出：时间戳、`ranges`、`intensities`、`angle_min`、`angle_increment`。
2. **预处理**：去掉 inf/nan/0；按量程上下限截断；屏蔽打到机器人自身的角度段；
   沿角度做中值滤波；剔除水面镜面反射、边缘混合像素、多路径造成的离群点。
   3D 雷达还要做运动畸变补偿（deskew，用 IMU 或插值后的位姿）。
3. **单帧截面拟合**（每个扫描一个结果）：极坐标 → 平面 xy；
   RANSAC + 代数圆拟合（Kasa/Taubin）+ 几何最小二乘精修 → 圆心（= 机器人在管内的偏移）、半径 R；
   直接椭圆拟合（Fitzgibbon）→ 长短轴、椭圆度；残差剖面 r(θ) − R；
   底部若有一段直线弦 → 积水线高度。
   **注意**：斜着切圆柱得到的也是椭圆（长轴 = R / cos 倾角），倾 10° 就有约 1.5% 的"假椭圆度"，
   所以真变形和扫描面倾斜必须用 IMU/视觉位姿分开。
4. **轴向定位**：2D 扫描本身没有沿轴位置 s，来源只能是编码器/计数器、视觉 SfM 位姿或 IMU，因此需要**时间同步**。
5. **叠加与展开**：所有点放进**统一管道坐标系 (s, θ, r)**（现有 `evaluate.py` 里的 `pipe_frame` 已经在用）；
   得到偏差图 r(s, θ) − R_标称；表面可以直接用圆柱高度场网格表示（比 Poisson 重建简单得多）。
6. **质量指标**：半径噪声标准差、与标称管径之差、椭圆度、多次往返重复性。

3D 雷达变体：用 KISS-ICP / FAST-LIO2 做里程计，同时检查配准 Hessian 的特征值——
直管里沿轴方向的特征值会很小，那就是退化方向，预计会出现轴向漂移。

---

## 3. 怎么和现有图像模型对齐（三种方式，由易到难）

### 方式 ①：有外参 + 时间同步（同一台机器人上同时采集的数据）——首选
- 投影链：`T_世界←雷达(t) = T_世界←相机(t) · T_相机←雷达`，相机位姿来自 OpenMVG。
- 唯一未知量是视觉模型的尺度 λ（可以按管段取 λ(s) 来吸收局部尺度抖动）：
  让激光点落在 OpenMVS 稠密面上，或直接令视觉半径 × λ = 激光半径。
- 残差的大小同时检验外参和时间同步准不准。

### 方式 ②：没有外参 / 不是同一次采集——模型对模型配准，在管道坐标系里由粗到精
1. **轴线对齐**：两边各拟合圆柱轴线并对齐，尺度 = R_激光 / R_视觉（逐段算）。
2. **退化方向用特征定**：轴向偏移用一维剖面互相关（激光半径/强度剖面 vs 视觉壁面亮度剖面；
   现有 `compare_models.py` 就是这个思路）；特征有接口环、检查井、支管、缺陷。
   滚转用重力（IMU）、积水线/管底或支管方向。
3. **精配准**：带尺度的 ICP（Sim(3)，点到面）或 GICP，但要约束退化方向；
   更贴合现有误差形态的做法是一维非刚性映射 s_视觉 → s_激光（分段线性/样条），因为现有误差主要就是轴向拉伸抖动。

### 方式 ③：紧耦合位姿融合（研究创新点）
- 因子图（GTSAM）：视觉相对位姿因子（无尺度）+ 激光截面对圆柱因子（约束横向偏移、俯仰、偏航、尺度）
  + 编码器/计数器因子（约束轴向）+ IMU（约束滚转）。只在激光约束良好的方向使用激光（退化感知）。
- 融合后的位姿写回 OpenMVS（`scene.mvs` 本来就能接受任何来源的位姿）→ 公制稠密模型；
  或者反过来，用图像给激光几何上色/贴纹理。
- 现成框架 LVI-SAM、R3LIVE、FAST-LIVO2 面向开阔场景的 3D 雷达，在管道里会遇到退化，可作为对比基线。

**推荐顺序**：先做 ②/① 出第一个融合结果，再把标定做严谨，最后做 ③。

### 标定与同步（① 和 ③ 的前提）
- 外参：2D 雷达–相机用 Zhang & Pless 2004 的棋盘格方法；3D 雷达有基于标定板和无标定板（互信息/边缘）两类方法。
  管道里可以直接用管子当标定物：两边的圆柱轴线约束 4 个自由度，剩下的轴向偏移和滚转靠一个特征（接口环、贴标记）。
  如果相机能看到雷达的激光线（去掉红外截止滤光片或用近红外相机），就有直接的对应点。
- 时间：硬件触发/PTP 最好；没有的话，用两边都能算出的运动信号做互相关来估时间偏移
  （例如圆心横向偏移 vs SfM 相机横向位置，或相机滚转 vs IMU 角速度）。

---

## 4. 数据（第一步最实际的问题）

现有 Sheffield 视频**没有激光数据**。可选：

1. **InPipe-SLAM**（Zenodo 18615583，CC BY 4.0）：真实管道机器人，ROS 2 bag，带参考轨迹；两个文件 31.19 GB 和 9.42 GB。
   **更正（2026-10-14 读了两个 zip 里的 metadata.yaml）**：README 写的 2D `/scan` 在真实数据里不存在，
   激光是 `/lidar_points`（sensor_msgs/PointCloud2，约 10 Hz，**3D 雷达**）+ `/lidar_imu`（约 400 Hz）；
   另有 RealSense（color / depth / infra1 / infra2 / imu）、第二台相机 `/camera/usbcam/image_raw`、
   `/encoders`（Float64MultiArray）、`/cmd_vel`、`/tf`、`/tf_static`、`/trajectory`（nav_msgs/Path）。
   每个 zip 只有一条序列，格式是 MCAP（不是 db3）：172011 时长 146 s（解压后 39 GB），161330 时长 452 s（解压后 121 GB）。
   管径和雷达型号要从数据里读。
   优点：同一机器人有相机+雷达+编码器，可以直接走方式 ①，RGB 帧还能跑我们现有的视觉流程。
2. **仿真**（不用下载）：用已有的 concrete_2 网格当"真值管道"，沿相机轨迹模拟 2D 激光扫描并加噪声、加人为变形，
   先在已知真值下把方式 ①/② 的算法调通。
3. **自己搭设备**：相机 + 2D 雷达（注意最小量程和测距精度）。

---

## 5. 分步路线（每步做完先看结果再进下一步）

0. 数据体检（两种模态一起，几分钟）：话题、帧数、频率、时长；`/scan` 的量程和角度范围；相机内参（camera_info）；
   相机↔雷达外参（tf_static）；编码器格式；参考轨迹；看几张图像的纹理和亮度；选一条**直管**序列（弯头以后再做）。
1. 选数据：InPipe-SLAM 小文件（9.42 GB，放 D:）或先做仿真。
2. 激光单模态（先做：数据小、几分钟出结果、给出真实管径和统一的 (s, θ, r) 坐标与时间基准）：
   bag → 扫描 → 圆/椭圆拟合 → 每帧 R、圆心、椭圆度 → 配合编码器生成 3D 点云和 (s, θ, r) 展开图。
   注意这其实是"激光 + 编码器"，编码器在管内会打滑。
3. 视觉单模态：在同一序列的 RGB 帧上跑视觉流程。尺度只用**一个**全局管径数值（数据集标称值，或激光测得的平均管径），
   和两篇论文"一个已知尺寸定尺度"的做法一致，保证它仍是纯视觉基线。
   旧文件夹的 `run_pipeline.py` 写死了 Sheffield 相机（f=806、k1=−0.33）、从 mp4 抽帧、日志写进旧文件夹的 `scripts\logs\`，
   所以要复制到本文件夹改：从 bag 抽帧、内参取自 camera_info、日志写本文件夹。
4. 融合：方式 ①（λ(s) 尺度估计）和方式 ②（模型配准）各做一次，互相验证；
   结果和两个单模态基线放在一起比（纯视觉 / 激光+编码器 / 融合），这就是论文里的对比实验。
5. 标定/同步做严谨；6. 紧耦合因子图。

超声以后按同一套 (s, θ, r) 管道坐标系接入：典型分工是激光测水面以上、声呐测水面以下。

---

## 6. 检索关键词

**管道多传感器检测**
- `multi-sensor pipe inspection robot` / `sewer inspection laser profiling CCTV`
- `pipe profiling laser sonar fusion` / `partially filled pipe sonar laser profile`
- `in-pipe robot 2D LiDAR cross-section` / `pipe ovality measurement laser scanner`
- `ring laser structured light pipe inspection`

**管道 SLAM 与退化**
- `in-pipe SLAM` / `pipe robot localization LiDAR degeneracy`
- `LiDAR degeneracy tunnel corridor` / `localizability-aware LiDAR registration` / `degeneracy-aware sensor fusion`

**标定**
- `camera 2D laser range finder extrinsic calibration`
- `targetless LiDAR camera extrinsic calibration` / `LiDAR camera time offset estimation`

**融合与配准**
- `LiDAR-visual-inertial odometry` / `monocular SfM scale recovery LiDAR`
- `similarity transform point cloud registration scaled ICP` / `cylinder fitting point cloud pipe`
- `non-rigid point cloud registration` / `LiDAR point cloud colorization texture mapping`

**中文（知网/万方）**
- 管道机器人 激光雷达 视觉 融合；管道内检测 激光轮廓 三维重建
- 排水管道 多传感器 检测 声呐 激光；管道 椭圆度 变形 检测 激光
- 激光雷达 相机 外参标定；退化 激光 SLAM 隧道

**入门文献（凭记忆列出，本次联网检索不可用，请以检索结果核对）**
- Duran, Althoefer, Seneviratne, "State of the art in sensor technologies for sewer inspection", IEEE Sensors Journal, 2002 —— 管道检测传感器综述
- Hansen, Alismail, Rander, Browning, "Visual mapping for natural gas pipe inspection", IJRR, 2015 —— 管内视觉建图
- Zhang & Pless, "Extrinsic calibration of a camera and laser range finder", IROS 2004 —— 2D 雷达–相机标定
- Zhang, Kaess, Singh, "On degeneracy of optimization-based state estimation problems", ICRA 2016 —— 退化分析
- Tuna et al., "X-ICP: Localizability-aware LiDAR registration for robust localization in extreme environments", IEEE T-RO —— 退化感知配准
- Koide et al., "General, single-shot, target-less, and automatic LiDAR-camera extrinsic calibration toolbox", ICRA 2023
- Umeyama, "Least-squares estimation of transformation parameters between two point patterns", TPAMI 1991 —— Sim(3) 闭式解
- LVI-SAM（Shan et al., ICRA 2021）、R3LIVE（Lin & Zhang, ICRA 2022）、FAST-LIVO2（Zheng et al.）—— 激光-视觉-惯性融合基线
