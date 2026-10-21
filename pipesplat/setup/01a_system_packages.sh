#!/bin/bash
# 第 0 步 / docs/07 第 4.3 节（前半）：编译基础包 + NVIDIA 软件源，并查询 CUDA 12.8 工具包的准确下载量。
# 在 pipe3d 发行版里以 root 运行：wsl -d pipe3d -u root -- bash <本文件>
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
LOG=/mnt/d/pipe3d_gs/logs/setup_01a.log
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "== $(date '+%F %T') apt-get update"
apt-get update

echo "== $(date '+%F %T') build tools"
apt-get install -y --no-install-recommends build-essential python3-dev python3-venv git ninja-build wget ca-certificates

echo "== $(date '+%F %T') NVIDIA wsl-ubuntu keyring"
wget -q -O /tmp/cuda-keyring_1.1-1_all.deb \
  https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i /tmp/cuda-keyring_1.1-1_all.deb
apt-get update

echo "== $(date '+%F %T') size check for cuda-toolkit-12-8 (nothing is installed here)"
apt-get install --no-install-recommends --assume-no cuda-toolkit-12-8 | grep -E "^(Need to get|After this operation|[0-9]+ upgraded)" || true
apt-cache policy cuda-toolkit-12-8 | sed -n 1,3p   # sed 读完整个输出；head 会触发 SIGPIPE，在 pipefail 下让脚本以 141 退出

echo "== $(date '+%F %T') versions"
gcc --version | head -1
python3 --version
df -h / | tail -1
echo "== $(date '+%F %T') done"
