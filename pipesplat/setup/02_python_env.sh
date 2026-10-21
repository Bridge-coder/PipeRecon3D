#!/bin/bash
# 第 0 步 / docs/07 第 4.4 节：Python 3.12 venv + torch 2.9.1（PyPI 的 Linux 默认包即 CUDA 12.8 构建）+ 常用包。
# --no-cache-dir：WSL 虚拟磁盘不能设为稀疏，pip 缓存删掉后空间也不会还给 D:，所以不留缓存。
# 在 pipe3d 发行版里以 root 运行：wsl -d pipe3d -u root -- bash <本文件>
set -euo pipefail
LOG=/mnt/d/pipe3d_gs/logs/setup_02.log
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
VENV=/opt/gsenv

echo "== $(date '+%F %T') create venv $VENV"
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"

echo "== $(date '+%F %T') torch 2.9.1 + torchvision 0.24.1 (~4.07 GB download)"
"$VENV/bin/pip" install --no-cache-dir torch==2.9.1 torchvision==0.24.1

echo "== $(date '+%F %T') other packages"
"$VENV/bin/pip" install --no-cache-dir numpy scipy "opencv-python-headless<5" matplotlib imageio tqdm lpips jaxtyping rich ninja nvtx pillow

echo "== $(date '+%F %T') check torch on the GPU"
"$VENV/bin/python" - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
print("arch list", torch.cuda.get_arch_list())
x = torch.randn(4096, 4096, device="cuda")
y = x @ x
torch.cuda.synchronize()
print("matmul ok", bool(torch.isfinite(y).all()), "peak MB", round(torch.cuda.max_memory_allocated() / 1e6))
EOF

"$VENV/bin/pip" list > /mnt/d/pipe3d_gs/logs/pip_list_02.txt
du -sh "$VENV"
df -h / | tail -n 1
echo "== $(date '+%F %T') done"
