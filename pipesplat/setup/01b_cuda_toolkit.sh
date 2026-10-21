#!/bin/bash
# 第 0 步 / docs/07 第 4.3 节（后半）：安装 CUDA 12.8 工具包（只为编译 gsplat 提供 nvcc 和头文件）。
# 只装 cuda-toolkit-12-8；不装 cuda / cuda-drivers 元包——WSL 里的显卡驱动来自 Windows，装 Linux 驱动会出问题。
# 在 pipe3d 发行版里以 root 运行：wsl -d pipe3d -u root -- bash <本文件>
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
LOG=/mnt/d/pipe3d_gs/logs/setup_01b.log
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "== $(date '+%F %T') install cuda-toolkit-12-8"
apt-get install -y --no-install-recommends cuda-toolkit-12-8

echo "== $(date '+%F %T') clean apt cache"
apt-get clean

echo "== $(date '+%F %T') check"
/usr/local/cuda-12.8/bin/nvcc --version
gcc --version | sed -n 1p
python3 --version
df -h / | tail -n 1
echo "== $(date '+%F %T') done"
