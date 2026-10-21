"""Does gsplat put a point where our camera model says it should? (radial k1 convention check)

One tiny Gaussian at a time at known undistorted normalised coordinates (x_u, y_u) at depth 1 m; the centroid of
its rendered blob is compared with the pixel our pipeline expects: x_d = x_u (1 + k1 r_u^2) (OpenMVG radial-k1, the
model the SfM intrinsics were registered with), u = f x_d + cx (pixel centres at integers, as in OpenMVG/OpenCV;
the K given to gsplat has cx + 0.5 because gsplat puts pixel centres at +0.5).
Rendered three ways: 3DGUT with radial_coeffs, plain pinhole (no distortion), and plain pinhole for reference.
usage: /opt/gsenv/bin/python check_distortion.py
"""
import numpy as np
import torch
import gsplat

dev = "cuda"
W, H, f, cx, cy, k1 = 768, 548, 806.0, 383.0, 288.0, -0.33
K = torch.tensor([[f, 0, cx + 0.5], [0, f, cy + 0.5], [0, 0, 1]], device=dev)[None]
view = torch.eye(4, device=dev)[None]
rad = torch.tensor([[k1, 0, 0, 0, 0, 0]], device=dev)
ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32),
                        indexing="ij")


def blob(xu, yu, ut):
    mean = torch.tensor([[xu, yu, 1.0]], device=dev)
    kw = dict(packed=False, with_ut=True, with_eval3d=True, radial_coeffs=rad) if ut else dict(packed=False)
    img, alpha, _ = gsplat.rasterization(mean, torch.tensor([[1.0, 0, 0, 0]], device=dev), torch.full((1, 3), 0.0015, device=dev),
                                         torch.ones(1, device=dev), torch.ones(1, 3, device=dev), view, K, W, H, **kw)
    a = alpha[0, ..., 0]
    s = a.sum()
    if s < 1e-6:
        return None
    return (a * xs).sum().item() / s.item(), (a * ys).sum().item() / s.item()


print(f"k1 {k1}, f {f}, principal point ({cx}, {cy}) in integer-centre pixels")
print(f"{'x_u':>6} {'y_u':>6} | {'expected distorted':>20} | {'3DGUT':>18} {'err px':>7} | {'pinhole expected':>18} | {'pinhole':>18} {'err px':>7}")
for xu, yu in [(0.0, 0.0), (0.2, 0.0), (0.4, 0.0), (0.0, -0.3), (0.5, 0.2), (-0.45, -0.25), (0.6, 0.3), (-0.7, 0.2)]:
    r2 = xu * xu + yu * yu
    ed = (f * xu * (1 + k1 * r2) + cx, f * yu * (1 + k1 * r2) + cy)
    ep = (f * xu + cx, f * yu + cy)
    gu, gp = blob(xu, yu, True), blob(xu, yu, False)
    fmt = lambda p: "      (outside)     " if p is None else f"({p[0]:7.1f}, {p[1]:7.1f})"  # noqa: E731
    eu = np.nan if gu is None else np.hypot(gu[0] - ed[0], gu[1] - ed[1])
    epn = np.nan if gp is None else np.hypot(gp[0] - ep[0], gp[1] - ep[1])
    print(f"{xu:6.2f} {yu:6.2f} | {fmt(ed):>20} | {fmt(gu):>18} {eu:7.2f} | {fmt(ep):>18} | {fmt(gp):>18} {epn:7.2f}")
