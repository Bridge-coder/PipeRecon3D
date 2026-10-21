"""Smoke test: can gsplat fit our pipe frames on this GPU? (step 0, docs/07 section 4.6; runs in the pipe3d WSL distro)

usage: /opt/gsenv/bin/python pipesplat/smoke.py SCENE_DIR [--iters 2000] [--no-ut]
       SCENE_DIR  output of export_scene.py (scene.npz); results are written next to it:
                  smoke.json (memory, speed, PSNR before/after), smoke_loss.csv, smoke_<train|held>_<frame>.png
                  (video frame | render | |difference| x4)

Deliberately minimal - no lamp model, no depth loss, no densification, no pose refinement (those are steps 1-4):
one Gaussian per textured mesh face, scale = the larger of distance to the nearest camera / focal (Gaussian-LIC2
7.1.1, made for dense LiDAR points) and half the face size (our points are mesh faces a few mm apart, many pixels
apart close to the camera - the LiDAR rule alone would leave holes), opacity 0.1, colour from the texture; loss 0.8 L1 + 0.2 D-SSIM (Gaussian-LIC2 eq. 21); raw distorted frames
rendered through 3DGUT with the registered radial k1. It only answers: does it run, how much memory and time,
and do the renders land on the frames (PSNR rising on frames it was not trained on).

With k1 < 0 the radial model x_d = x_u (1 + k1 r^2) folds back beyond r_u = 1 / sqrt(-3 k1): wall right beside the
camera would be projected into the picture a second time. Gaussians beyond 95 % of that radius are left out of
each render (they are outside the picture anyway).
"""
import argparse
import csv
import json
import os
import time

import cv2
import numpy as np
import torch
import gsplat
from fused_ssim import fused_ssim

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("scene"), ap.add_argument("--iters", type=int, default=2000), ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no-ut", action="store_true", help="plain pinhole projection (ignore k1) for comparison")
a = ap.parse_args()
torch.manual_seed(a.seed)
np.random.seed(a.seed)
dev = "cuda"
t0 = time.time()
d = np.load(os.path.join(a.scene, "scene.npz"))
W, H = (int(v) for v in d["size"])
K = torch.from_numpy(d["K"]).to(dev)
k1 = float(d["k1"])
rad = torch.tensor([[k1, 0, 0, 0, 0, 0]], device=dev, dtype=torch.float32)
imgs = torch.from_numpy(d["images"]).to(dev).float() / 255           # N, H, W, 3
views = torch.from_numpy(d["viewmats"]).to(dev)
held_imgs = torch.from_numpy(d["held_images"]).to(dev).float() / 255
held_views = torch.from_numpy(d["held_viewmats"]).to(dev)
pts = torch.from_numpy(d["points"]).to(dev)
cols = torch.from_numpy(d["colors"]).to(dev).float() / 255
eyes = torch.from_numpy(d["eyes"]).to(dev)
print(f"{len(imgs)} model frames + {len(held_imgs)} held-out, {W}x{H}, f {K[0, 0].item():.1f}, k1 {k1}, "
      f"{len(pts):,} points [{time.time() - t0:.1f} s]")

# ---- Gaussians ----
dist = torch.cdist(pts, eyes).min(dim=1).values                     # to the nearest camera centre
size = torch.from_numpy(d["face_size"]).to(dev)
means = torch.nn.Parameter(pts.clone())
log_scales = torch.nn.Parameter(torch.log(torch.maximum(dist / K[0, 0], 0.5 * size)).unsqueeze(1).repeat(1, 3))
quats = torch.nn.Parameter(torch.tensor([1.0, 0, 0, 0], device=dev).repeat(len(pts), 1))
opac = torch.nn.Parameter(torch.logit(torch.full((len(pts),), 0.1, device=dev)))
color = torch.nn.Parameter(torch.logit(cols.clamp(0.02, 0.98)))
opt = torch.optim.Adam([dict(params=[means], lr=1.6e-4), dict(params=[log_scales], lr=5e-3),
                        dict(params=[quats], lr=1e-3), dict(params=[opac], lr=5e-2), dict(params=[color], lr=5e-3)],
                       eps=1e-15)
ru_max = 0.95 / np.sqrt(-3 * k1) if k1 < 0 else np.inf


def render(view):
    cam = means @ view[:3, :3].T + view[:3, 3]
    z = cam[:, 2]
    ru2 = (cam[:, 0] ** 2 + cam[:, 1] ** 2) / z.clamp(min=1e-6) ** 2
    keep = (z > 0.01) & (ru2 < ru_max ** 2)
    idx = torch.nonzero(keep).squeeze(1)
    kw = dict(packed=False) if a.no_ut else dict(packed=False, with_ut=True, with_eval3d=True, radial_coeffs=rad)
    out, alpha, _ = gsplat.rasterization(means[idx], quats[idx], torch.exp(log_scales[idx]), torch.sigmoid(opac[idx]),
                                         torch.sigmoid(color[idx]), view[None], K[None], W, H, **kw)
    return out[0], alpha[0, ..., 0], keep.float().mean().item()


def psnr(x, y):
    return (10 * torch.log10(1 / ((x - y) ** 2).mean())).item()


def evaluate(tag):
    with torch.no_grad():
        tr = [psnr(render(views[i])[0].clamp(0, 1), imgs[i]) for i in range(len(imgs))]
        he = [psnr(render(held_views[i])[0].clamp(0, 1), held_imgs[i]) for i in range(len(held_imgs))]
    print(f"[{tag}] PSNR model frames median {np.median(tr):.2f} dB, held-out median {np.median(he):.2f} dB")
    return tr, he


tr0, he0 = evaluate("before")
torch.cuda.reset_peak_memory_stats()
log = []
t1 = time.time()
for it in range(1, a.iters + 1):
    i = np.random.randint(len(imgs))
    ren, alpha, kept = render(views[i])
    gt = imgs[i]
    l1 = (ren - gt).abs().mean()
    ssim = fused_ssim(ren.permute(2, 0, 1)[None].contiguous(), gt.permute(2, 0, 1)[None].contiguous())
    loss = 0.8 * l1 + 0.2 * (1 - ssim)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if it % 100 == 0 or it == 1:
        torch.cuda.synchronize()
        log.append(dict(iter=it, loss=loss.item(), l1=l1.item(), ssim=ssim.item(), psnr=psnr(ren.detach().clamp(0, 1), gt),
                        kept=kept, s=time.time() - t1))
        print(f"iter {it:5d} loss {loss.item():.4f} PSNR {log[-1]['psnr']:.2f} kept {kept:.0%} "
              f"{it / (time.time() - t1):.1f} it/s peak {torch.cuda.max_memory_allocated() / 1e6:.0f} MB")
torch.cuda.synchronize()
train_s = time.time() - t1
peak_alloc, peak_res = torch.cuda.max_memory_allocated() / 1e6, torch.cuda.max_memory_reserved() / 1e6
free, total = torch.cuda.mem_get_info()
tr1, he1 = evaluate("after")


def save(tag, key, img, view):
    with torch.no_grad():
        ren = render(view)[0].clamp(0, 1)
    diff = ((ren - img).abs() * 4).clamp(0, 1)
    trip = (torch.cat([img, ren, diff], dim=1).cpu().numpy() * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(a.scene, f"smoke{'_no_ut' if a.no_ut else ''}_{tag}_{key:06d}.png"), trip[:, :, ::-1])


for i in (0, len(imgs) // 2, len(imgs) - 1):
    save("train", int(d["frame_keys"][i]), imgs[i], views[i])
for i in range(len(held_imgs)):
    save("held", int(d["held_keys"][i]), held_imgs[i], held_views[i])

with open(os.path.join(a.scene, "smoke_no_ut_loss.csv" if a.no_ut else "smoke_loss.csv"), "w", newline="") as fh:
    wr = csv.DictWriter(fh, list(log[0]))
    wr.writeheader()
    wr.writerows(log)
summary = dict(gaussians=len(pts), iters=a.iters, train_seconds=round(train_s, 1), it_per_s=round(a.iters / train_s, 1),
               peak_allocated_MB=round(peak_alloc), peak_reserved_MB=round(peak_res),
               gpu_free_after_MB=round(free / 1e6), gpu_total_MB=round(total / 1e6), ut=not a.no_ut, k1=k1,
               psnr_model_frames=dict(before=float(np.median(tr0)), after=float(np.median(tr1))),
               psnr_held_out=dict(before=float(np.median(he0)), after=float(np.median(he1)),
                                  per_frame_after=[round(v, 2) for v in he1]),
               torch=torch.__version__, gsplat=gsplat.__version__, total_seconds=round(time.time() - t0, 1))
json.dump(summary, open(os.path.join(a.scene, "smoke.json" if not a.no_ut else "smoke_no_ut.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
