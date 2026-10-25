"""Step 1 (docs/06, docs/08): G0 = Gaussian-LIC2-style Gaussian mapping on our poses and geometry, then the in-sequence
and block evaluation against lampmap's B on the same pixels. Runs in the pipe3d WSL distro.

usage: TORCH_HOME=<TORCH_CACHE of scripts/paths.py> /opt/gsenv/bin/python pipesplat/train_g0.py SCENE_DIR OUT_DIR
           [--iters 30000] [--adc] [--no-blocks] [--save 4] [--seed 0]
       SCENE_DIR  output of export_g0.py (scene.npz)
       OUT_DIR    train.log, model.pt, summary.json, per_frame.csv, renders/*.jpg (video | B | G0)

What is Gaussian-LIC2 here (paper section in brackets) and what is not:
  same:  one Gaussian per geometry point, colour from projecting it into a training frame (6.1), scale from its depth
         over the focal length (7.1.1), opacity 0.1, identity rotation, full 3rd-order SH; loss
         0.8 L1 + 0.2 D-SSIM + 0.005 |D/O - D_s| on pixels with geometry depth (eq. 20-22); undistorted pinhole
         frames; no adaptive density control by default (7.1, appendix: ADC hurts their incremental mapping)
  ours:  poses = our SfM / fused poses (as their mapping evaluation, 9.4.1, uses given poses); geometry = OpenMVS
         dense cloud (vision) or the refined LiDAR wall points, depth from the 5 scans nearest to each frame (InPipe)
  differences, each for a stated reason:
    - offline, all training frames at once, 30k iterations with the vanilla 3DGS position-lr decay (their system is
      online; their offline comparison with LetsGo also trains 30k iterations)
    - scale = max(depth / focal, half the distance to the nearest point): our points are a few mm apart, and the
      smoke test (docs/07 4.7) showed the depth/focal rule alone leaves holes near the camera
    - the undistorted frames keep the whole raw picture (a larger pinhole canvas) instead of cropping it
    - no LiDAR blind-area compensation (their 6.2): coverage is measured here and treated in step 5
    - --adc turns on gsplat's default densification (offline it is the standard 3DGS choice) as a second run
  Points never seen by a training frame are dropped (they would never be optimised).
The k1 fold: with k1 < 0, x_d = x_u (1 + k1 r^2) turns back beyond r_u = 1/sqrt(-3 k1); the canvas stays inside 95 %.
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import torch
import gsplat
from gsplat.strategy import DefaultStrategy
from fused_ssim import fused_ssim
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "lampmap"))
import imgmetrics as im  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("scene"), ap.add_argument("out")
ap.add_argument("--iters", type=int, default=30000), ap.add_argument("--adc", action="store_true")
ap.add_argument("--no-blocks", action="store_true", help="train on the block frames too (block test then meaningless)")
ap.add_argument("--save", type=int, default=4), ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--dssim", type=float, default=0.2), ap.add_argument("--depth-w", type=float, default=0.005)
ap.add_argument("--eval-only", action="store_true", help="load OUT_DIR/model.pt and only score it")
ap.add_argument("--alpha-min", type=float, default=0.99, help="pixels scored: model opacity above this (and B's mask)")
ap.add_argument("--suffix", default="", help="appended to summary/per_frame file names (e.g. _alpha050)")
a = ap.parse_args()
torch.manual_seed(a.seed)
np.random.seed(a.seed)
dev = "cuda"
os.makedirs(os.path.join(a.out, "renders"), exist_ok=True)
T0 = time.time()
_logf = open(os.path.join(a.out, "train.log"), "a", encoding="utf-8")


def log(msg):
    line = f"{msg} [{time.time() - T0:.0f} s]"
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


log(f"args {vars(a)} | torch {torch.__version__} gsplat {gsplat.__version__}")
d = np.load(os.path.join(a.scene, "scene.npz"))
K = d["K"].astype(np.float64)
k1 = float(d["k1"])
w, h = (int(v) for v in d["size"])
fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
inpipe = "depth_scan" in d.files


# ------------------------------------------------------------------------------ camera canvas ---------------
def undistort_grid(xd, yd, k1):
    """inverse of x_d = x_u (1 + k1 r_u^2) by iteration (render_views.undistort_grid, copied: that module needs the
    Windows paths); NaN beyond 95 % of the radius the model reaches"""
    out = (xd ** 2 + yd ** 2 >= (0.95 * 2 / (3 * np.sqrt(-3 * k1))) ** 2) if k1 < 0 else np.zeros(xd.shape, bool)
    xd, yd = np.where(out, 0.0, xd), np.where(out, 0.0, yd)
    xu, yu = xd.copy(), yd.copy()
    for _ in range(20):
        g = 1 + k1 * (xu ** 2 + yu ** 2)
        xu, yu = xd / g, yd / g
    xu[out], yu[out] = np.nan, np.nan
    return xu, yu


if k1 == 0.0:
    CW, CH, ccx, ccy = w, h, cx, cy
    valid_raw = np.ones((h, w), bool)
    to_canvas = to_raw = None
    valid_canvas = np.ones((h, w), bool)
else:
    XU, YU = undistort_grid(*np.meshgrid((np.arange(w) - cx) / fx, (np.arange(h) - cy) / fy), k1)
    valid_raw = np.isfinite(XU)
    CW, CH = 2 * int(np.ceil(np.nanmax(np.abs(XU)) * fx)) + 4, 2 * int(np.ceil(np.nanmax(np.abs(YU)) * fy)) + 4
    ccx, ccy = CW / 2 - 0.5, CH / 2 - 0.5
    to_raw = (np.nan_to_num(ccx + XU * fx, nan=-10).astype(np.float32), np.nan_to_num(ccy + YU * fy, nan=-10).astype(np.float32))
    xu, yu = np.meshgrid((np.arange(CW) - ccx) / fx, (np.arange(CH) - ccy) / fy)
    g = 1 + k1 * (xu ** 2 + yu ** 2)
    ud, vd = (cx + fx * xu * g).astype(np.float32), (cy + fy * yu * g).astype(np.float32)
    valid_canvas = ((xu ** 2 + yu ** 2) < (0.95 / np.sqrt(-3 * k1)) ** 2) & (ud >= 0) & (ud <= w - 1) & (vd >= 0) & (vd <= h - 1)
    to_canvas = (ud, vd)
Kc = torch.tensor([[fx, 0, ccx + 0.5], [0, fy, ccy + 0.5], [0, 0, 1]], device=dev, dtype=torch.float32)  # gsplat: centres at +0.5
VALID = torch.from_numpy(valid_canvas).to(dev)
log(f"raw {w}x{h}, k1 {k1}; training canvas {CW}x{CH} (pinhole), valid {valid_canvas.mean():.1%} of it")


def canvas(img):
    return img if to_canvas is None else cv2.remap(img, to_canvas[0], to_canvas[1], cv2.INTER_LINEAR, borderValue=0)


# ------------------------------------------------------------------------------ frames, depth ---------------
views_all = torch.from_numpy(d["train_viewmats"]).to(dev)
eyes = d["train_eyes"]
block_idx = d["block_idx"]
train = np.setdiff1d(np.arange(len(views_all)), [] if a.no_blocks else block_idx)
raw_train = d["train_images"]
IM = {int(i): torch.from_numpy(canvas(raw_train[i])).pin_memory() for i in train}
log(f"{len(train)} training frames (left out: {0 if a.no_blocks else len(block_idx)} block frames in "
    f"{d['block_ranges'].round(3).tolist()} m)")

DP = torch.from_numpy(d["depth_points"]).to(dev)
if inpipe:                                     # LiDAR: the 5 scans nearest in time to a frame (Gaussian-LIC2 6.1)
    scan = torch.from_numpy(d["depth_scan"].astype(np.int64)).to(dev)
    order = torch.argsort(scan)
    DP, scan = DP[order], scan[order]
    starts = torch.searchsorted(scan, torch.arange(len(d["scan_t"]) + 1, device=dev))
    scan_t = d["scan_t"]

    def depth_points(t):
        near = np.argsort(np.abs(scan_t - t))[:5]
        return torch.cat([DP[starts[s]:starts[s + 1]] for s in near])
else:
    def depth_points(t):
        return DP


def depth_map(view, t):
    """sparse geometry depth on the canvas (0 = none), nearest point per pixel"""
    p = depth_points(t)
    c = p @ view[:3, :3].T + view[:3, 3]
    z = c[:, 2]
    k = z > 0.01
    c, z = c[k], z[k]
    i = torch.round(ccx + fx * c[:, 0] / z).long()
    j = torch.round(ccy + fy * c[:, 1] / z).long()
    k = (i >= 0) & (i < CW) & (j >= 0) & (j < CH)
    D = torch.full((CH * CW,), float("inf"), device=dev)
    D.scatter_reduce_(0, j[k] * CW + i[k], z[k], reduce="amin")
    D[torch.isinf(D)] = 0
    return D.view(CH, CW)


# ------------------------------------------------------------------------------ Gaussians --------------------
pts = torch.from_numpy(d["points"]).to(dev)
best_z = torch.full((len(pts),), float("inf"), device=dev)
rgb = torch.zeros(len(pts), 3, device=dev)
for n in train:                                 # colour = the training frame that sees the point closest
    v = views_all[n]
    c = pts @ v[:3, :3].T + v[:3, 3]
    z = c[:, 2]
    i = torch.round(ccx + fx * c[:, 0] / z.clamp(min=1e-6)).long()
    j = torch.round(ccy + fy * c[:, 1] / z.clamp(min=1e-6)).long()
    ok = (z > 0.01) & (i >= 8) & (i < CW - 8) & (j >= 8) & (j < CH - 8)
    ok &= VALID[j.clamp(0, CH - 1), i.clamp(0, CW - 1)]
    better = ok & (z < best_z)
    if better.any():
        img = IM[int(n)].to(dev)
        rgb[better] = img[j[better], i[better]].float() / 255
        best_z[better] = z[better]
seen = torch.isfinite(best_z)
pts, rgb, best_z = pts[seen], rgb[seen], best_z[seen]
nn_d, _ = cKDTree(pts.cpu().numpy()).query(pts.cpu().numpy(), k=2)
spacing = torch.from_numpy(nn_d[:, 1].astype(np.float32)).to(dev)
scale0 = torch.maximum(best_z / fx, 0.5 * spacing)
log(f"{int(seen.sum()):,} of {len(seen):,} points seen by a training frame; scale median {scale0.median().item() * 1000:.2f} mm "
    f"(depth/focal {(best_z / fx).median().item() * 1000:.2f} mm, half spacing {(0.5 * spacing).median().item() * 1000:.2f} mm)")

C0 = 0.28209479177387814
params = torch.nn.ParameterDict({
    "means": torch.nn.Parameter(pts.clone()),
    "scales": torch.nn.Parameter(torch.log(scale0).unsqueeze(1).repeat(1, 3)),
    "quats": torch.nn.Parameter(torch.tensor([1.0, 0, 0, 0], device=dev).repeat(len(pts), 1)),
    "opacities": torch.nn.Parameter(torch.logit(torch.full((len(pts),), 0.1, device=dev))),
    "sh0": torch.nn.Parameter(((rgb - 0.5) / C0).unsqueeze(1)),
    "shN": torch.nn.Parameter(torch.zeros(len(pts), 15, 3, device=dev)),
})
if a.eval_only:                                 # score an already trained model
    saved = torch.load(os.path.join(a.out, "model.pt"))
    params = torch.nn.ParameterDict({k: torch.nn.Parameter(saved[k].to(dev)) for k in params.keys()})
    log(f"eval only: loaded {len(params['means']):,} Gaussians from {a.out}/model.pt; scoring pixels with opacity > {a.alpha_min}")
scene_scale = 1.1 * float(np.max(np.linalg.norm(eyes - eyes.mean(0), axis=1)))
lrs = dict(means=1.6e-4 * scene_scale, scales=5e-3, quats=1e-3, opacities=5e-2, sh0=2.5e-3, shN=2.5e-3 / 20)
optimizers = {k: torch.optim.Adam([params[k]], lr=lr, eps=1e-15) for k, lr in lrs.items()}
sched = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / a.iters))
strategy = DefaultStrategy(refine_stop_iter=min(15000, a.iters // 2)) if a.adc else None
state = strategy.initialize_state(scene_scale=scene_scale) if a.adc else None
if a.adc:
    strategy.check_sanity(params, optimizers)
log(f"scene scale {scene_scale:.2f} m; ADC {'on' if a.adc else 'off'}; {a.iters} iterations")


def rasterize(view, mode="RGB+ED"):
    p = params
    return gsplat.rasterization(p["means"], p["quats"], torch.exp(p["scales"]), torch.sigmoid(p["opacities"]),
                                torch.cat([p["sh0"], p["shN"]], 1), view[None], Kc[None], CW, CH, sh_degree=3,
                                packed=False, render_mode=mode, near_plane=0.01)


# ------------------------------------------------------------------------------ training ---------------------
torch.cuda.reset_peak_memory_stats()
t_train = time.time()
seq, done = [], 0
lam, gam = a.dssim, a.depth_w
t_of = d["train_t"]
try:
    for it in range(1, 0 if a.eval_only else a.iters + 1):
        if not seq:
            seq = list(np.random.permutation(train))
        n = int(seq.pop())
        gt = IM[n].to(dev, non_blocking=True).float() / 255
        view = views_all[n]
        out, alpha, info = rasterize(view)
        if a.adc:
            strategy.step_pre_backward(params, optimizers, state, it, info)
        ren, dep = out[0, ..., :3], out[0, ..., 3]
        l1 = (ren - gt).abs()[VALID].mean()
        ssim = fused_ssim(torch.where(VALID[..., None], ren, gt).permute(2, 0, 1)[None].contiguous(),
                          gt.permute(2, 0, 1)[None].contiguous())
        Ds = depth_map(view, t_of[n])
        md = (Ds > 0) & VALID
        ld = (dep - Ds).abs()[md].mean() if md.any() else torch.zeros((), device=dev)
        loss = (1 - lam) * l1 + lam * (1 - ssim) + gam * ld
        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        sched.step()
        if a.adc:
            strategy.step_post_backward(params, optimizers, state, it, info, packed=False)
        done = it
        if it % 500 == 0 or it == 1:
            with torch.no_grad():
                ps = -10 * torch.log10(((ren - gt) ** 2)[VALID].mean()).item()
            log(f"iter {it:6d} loss {loss.item():.4f} L1 {l1.item():.4f} SSIM {ssim.item():.3f} depth {ld.item() * 1000:.1f} mm "
                f"PSNR {ps:.2f} | {len(params['means']):,} Gaussians | {it / (time.time() - t_train):.1f} it/s | "
                f"peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
except torch.cuda.OutOfMemoryError as e:
    log(f"OUT OF MEMORY at iteration {done + 1} with {len(params['means']):,} Gaussians: {e}")
torch.cuda.synchronize()
train_s = time.time() - t_train
peak = torch.cuda.max_memory_allocated() / 1e9
if not a.eval_only:
    torch.save({k: v.detach().cpu() for k, v in params.items()} | dict(K=Kc.cpu(), size=(CW, CH)), os.path.join(a.out, "model.pt"))
    log(f"trained {done} iterations in {train_s / 60:.1f} min, {len(params['means']):,} Gaussians, peak {peak:.2f} GB")


# ------------------------------------------------------------------------------ evaluation -------------------
@torch.no_grad()
def render_raw(view):
    out, alpha, _ = rasterize(view)
    ren = (out[0, ..., :3].clamp(0, 1) * 255).cpu().numpy()
    al = alpha[0, ..., 0].cpu().numpy()
    dep_c = out[0, ..., 3]
    if to_raw is not None:
        ren = cv2.remap(ren, *to_raw, cv2.INTER_LINEAR, borderValue=0)
        al = cv2.remap(al, *to_raw, cv2.INTER_LINEAR, borderValue=0)
    return ren, al, dep_c, alpha[0, ..., 0]


def score(tag, imgs, nxt, views, B, Bm, ids, pos, times):
    rows = []
    for k in range(len(ids)):
        view = torch.from_numpy(views[k]).to(dev)
        G, aG, dep_c, al_c = render_raw(view)
        mG = valid_raw & (aG > a.alpha_min)
        m = Bm[k] & mG
        real = imgs[k].astype(np.float32)
        Bk = B[k].astype(np.float32)
        row = dict(tag=tag, frame=int(ids[k]), pos=float(pos[k]), coverage=float(m.mean()), cov_G=float(mG.mean()),
                   cov_B=float(Bm[k].mean()))
        mats = {}
        for name, img, mask in (("G", G, m), ("B", Bk, m), ("N", nxt[k].astype(np.float32), m), ("Bown", Bk, Bm[k])):
            x = im.linear_match(img, real, mask)
            mats[name] = x
            for key, val in im.all_metrics(x, real, mask).items():
                row[f"{name}_{key}"] = val
        Ds = depth_map(view, times[k])
        md = (Ds > 0) & (al_c > a.alpha_min) & VALID
        rel = ((dep_c - Ds).abs() / Ds)[md]
        row["depth_rel_err"] = float(rel.median()) if len(rel) else np.nan
        rows.append(row)
        if a.save and k % a.save == 0 and not a.eval_only:
            trip = np.hstack([real, mats["B"], mats["G"]]).clip(0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(a.out, "renders", f"{tag}_{int(ids[k]):06d}.jpg"), trip[:, :, ::-1],
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
        log(f"  {tag} {k + 1}/{len(ids)} frame {int(ids[k])}: PSNR G {row['G_psnr']:.2f} B {row['B_psnr']:.2f} "
            f"N {row['N_psnr']:.2f} | LPIPS G {row['G_lpips']:.3f} B {row['B_lpips']:.3f} | depth err {row['depth_rel_err']:.1%}")
    return rows


rng = d["block_ranges"]
in_block = lambda p: bool(np.any([(p >= lo) & (p < hi) for lo, hi in rng]))  # noqa: E731
held_t = d["held_t"] if "held_t" in d.files else np.zeros(len(d["held_ids"]), np.int64)
rows = score("held", d["held_images"], d["held_next"], d["held_viewmats"], d["held_B"], d["held_mask"], d["held_ids"],
             d["held_pos"], held_t)
for r in rows:
    r["in_block"] = in_block(r["pos"])
bi = block_idx
rows_b = score("block", raw_train[bi], d["block_next"], d["block_viewmats"], d["block_B"], d["block_mask"], d["block_ids"],
               d["train_pos"][bi], t_of[bi])
for r in rows_b:
    r["in_block"] = True

keys = [k for k in rows[0] if k not in ("tag", "frame", "in_block")]
with open(os.path.join(a.out, f"per_frame{a.suffix}.csv"), "w", newline="", encoding="utf-8") as fh:
    wr = csv.DictWriter(fh, ["tag", "frame", "in_block"] + keys)
    wr.writeheader()
    wr.writerows(rows + rows_b)


def med(rs):
    return {k: float(np.nanmedian([r[k] for r in rs])) for k in keys} if rs else {}


summary = dict(scene=a.scene, args=vars(a), canvas=[CW, CH], train_frames=int(len(train)), iterations_done=done,
               train_minutes=round(train_s / 60, 2), it_per_s=round(done / max(train_s, 1e-9), 1), peak_GB=round(peak, 2),
               gaussians=int(len(params["means"])), block_ranges_m=rng.tolist(),
               held_all=med(rows), held_outside_blocks=med([r for r in rows if not r["in_block"]]),
               block=med(rows_b), n=dict(held=len(rows), held_outside_blocks=sum(not r["in_block"] for r in rows),
                                         block=len(rows_b)))
json.dump(summary, open(os.path.join(a.out, f"summary{a.suffix}.json"), "w"), indent=1)
M = ("psnr", "lpips", "shift_px", "detail_corr", "hf_ratio", "mse_low", "mse_high")
for title, s in (("in-sequence held-out, outside the blocks", summary["held_outside_blocks"]), ("block frames", summary["block"])):
    log(f"\n{title} (median; coverage {s['coverage']:.1%}, depth error G {s['depth_rel_err']:.1%}):")
    for tag, name in (("G", "G0"), ("B", "lampmap B"), ("N", "next frame"), ("Bown", "B on its own mask")):
        log(f"  {name:18s} " + "  ".join(f"{k} {s[f'{tag}_{k}']:.3f}" for k in M))
log("done")
