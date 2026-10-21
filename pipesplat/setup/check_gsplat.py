"""第 0 步检查：gsplat 在本机显卡上能否前向 + 反向，以及主线要用到的三项功能。

每项单独 try，一项失败不影响其他项的报告：
  1. 3DGS 光栅化（RGB + 深度，rasterization）
  2. 3DGUT：带径向畸变的相机（with_ut + radial_coeffs），对应 docs/06 第 4 步
  3. 2DGS 面元（rasterization_2dgs），对应 docs/06 第 4 步
  4. fused_ssim（训练损失里的 D-SSIM）
用法：/opt/gsenv/bin/python check_gsplat.py
"""
import time
import traceback

import torch

dev = "cuda"
torch.manual_seed(0)
print("torch", torch.__version__, "| device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
import gsplat  # noqa: E402

print("gsplat", gsplat.__version__)

W, H, N = 768, 548, 200_000          # concrete_2 的帧尺寸；20 万个高斯
K = torch.tensor([[806.0, 0, 383], [0, 806.0, 288], [0, 0, 1]], device=dev)[None]
viewmat = torch.eye(4, device=dev)[None]


def gaussians():
    # 在相机前方一段半径 0.15 的"管壁"上撒点（单位：米）
    th = torch.rand(N, device=dev) * 2 * torch.pi
    z = torch.rand(N, device=dev) * 3.0 + 0.2
    means = torch.stack([0.15 * torch.cos(th), 0.15 * torch.sin(th), z], -1).requires_grad_()
    quats = torch.randn(N, 4, device=dev).requires_grad_()
    scales = torch.full((N, 3), -6.0, device=dev).requires_grad_()   # log 尺度，约 2.5 mm
    opac = torch.zeros(N, device=dev).requires_grad_()                 # logit
    colors = torch.rand(N, 3, device=dev).requires_grad_()
    return means, quats, scales, opac, colors


def report(name, fn):
    torch.cuda.reset_peak_memory_stats()
    try:
        t = time.time()
        msg = fn()
        torch.cuda.synchronize()
        print(f"[OK]   {name}: {msg} | {1000 * (time.time() - t):.0f} ms | peak {torch.cuda.max_memory_allocated() / 1e6:.0f} MB")
    except Exception:
        print(f"[FAIL] {name}")
        traceback.print_exc()


def t_3dgs():
    m, q, s, o, c = gaussians()
    img, alpha, _ = gsplat.rasterization(m, q, torch.exp(s), torch.sigmoid(o), c, viewmat, K, W, H,
                                         render_mode="RGB+ED")
    (img[..., :3].mean() + img[..., 3].mean()).backward()
    return f"out {tuple(img.shape)}, alpha mean {alpha.mean().item():.3f}, grad|means| {m.grad.norm().item():.3e}"


def t_3dgut():
    m, q, s, o, c = gaussians()
    radial = torch.tensor([[-0.33, 0, 0, 0, 0, 0]], device=dev)        # Sheffield 相机 k1
    # packed=False：gsplat 的 3D 求值（with_eval3d）不支持 packed 模式，而 rasterization 默认 packed=True
    img, alpha, _ = gsplat.rasterization(m, q, torch.exp(s), torch.sigmoid(o), c, viewmat, K, W, H,
                                         packed=False, with_ut=True, with_eval3d=True, radial_coeffs=radial)
    img.mean().backward()
    return f"out {tuple(img.shape)}, alpha mean {alpha.mean().item():.3f}, grad|means| {m.grad.norm().item():.3e}"


def t_2dgs():
    m, q, s, o, c = gaussians()
    # 2DGS 的颜色要带相机维 [C, N, 3]（文档写 (C,) 可省，但 CUDA 核会拒绝 [N, 3]）
    out = gsplat.rasterization_2dgs(m, q, torch.exp(s), torch.sigmoid(o), c[None], viewmat, K, W, H)
    img = out[0]
    img.mean().backward()
    return f"out {tuple(img.shape)}, {len(out)} outputs (incl. normals), grad|means| {m.grad.norm().item():.3e}"


def t_ssim():
    from fused_ssim import fused_ssim
    a = torch.rand(1, 3, H, W, device=dev, requires_grad=True)
    b = torch.rand(1, 3, H, W, device=dev)
    v = fused_ssim(a, b)
    (1 - v).backward()
    return f"ssim(random) {v.item():.4f}, grad ok {bool(torch.isfinite(a.grad).all())}"


report("3DGS rasterization RGB+depth fwd/bwd", t_3dgs)
report("3DGUT distorted camera (k1 -0.33) fwd/bwd", t_3dgut)
report("2DGS surfels fwd/bwd", t_2dgs)
report("fused_ssim fwd/bwd", t_ssim)
