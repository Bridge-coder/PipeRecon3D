"""R4: texture of the wall, taken per texel from the colour images (not per LiDAR point).

Texels: 0.5 mm along the pipe x 0.25 deg around it (0.42 mm at r = 96 mm), placed on the R3 surface
(bilinear in the height field). For each of the 288 SfM colour frames every texel 5-60 cm ahead of the camera
is projected with the recorded K; its score is the pixel density on the wall, |cos(incidence)| / distance^2.
Pass 1 keeps the best score per texel, pass 2 averages the views within 60 % of it, weighted (score/best)^8,
so near the seams between views several views blend and elsewhere the best one dominates.
Photometric flat field: the robot's lamp and the lens vignetting are fixed in image coordinates, so the
per-pixel median over the frames (blurred) is divided out before sampling; the black/grey difference of the
two pipe sections is kept because the same gain map is used for every frame.
Exposure: the camera's auto exposure drifts by ~15 % along the run, so one gain per frame is solved from the
texels that several frames share (grey, log, alternating texel albedo / frame gain, smoothed over 5 frames).

Camera poses: the fused pose of f5_fuse.py (--poses fused), or that pose moved by the same rigid correction as
the LiDAR scan at that time (R2b, --poses refined). Reported for the choice: the weighted colour spread between
the views of a texel (lower = views agree), and the shift between the texture from the near views and one from
views at about 1.5-2.5 x the best distance (phase correlation on 96 x 96 texel tiles; a pose error that does
not fit the geometry moves the two apart).

output: RECON_BIG/texture_<poses>.png (+ _best.png: best view only, _far.png: far views),
        exposure_gain_<poses>.npy, flatfield.npy; results/inpipe_172011/recon/r4_texture_<poses>.json
usage: python -I r4_texture.py [--poses fused|refined] [--every N]
"""
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recon_common import BIG, EX, K, W, H, OUT, Centerline, FusedPoses  # noqa: E402
from vision_model import _expmap  # noqa: E402

args = sys.argv[1:]
POSES = args[args.index("--poses") + 1] if "--poses" in args else "refined"
EVERY = int(args[args.index("--every") + 1]) if "--every" in args else 1
DU_T, DT_T = 0.0005, 0.25
AHEAD = (0.05, 0.60)
MARGIN = 6
t0 = time.time()

cl = Centerline(os.path.join(BIG, "centerline.npz"))
hf = np.load(os.path.join(BIG, "heightfield.npz"))
a, b = (int(x) for x in hf["mesh_rows"])
Rg = hf["r_mesh"][a:b]
ucg, tcg = hf["u"][a:b], hf["th"]
du_g = float(hf["du"])
u_lo, u_hi = ucg[0] - du_g / 2, ucg[-1] + du_g / 2
nu_t = int(round((u_hi - u_lo) / DU_T))
nt_t = int(round(360 / DT_T))
u_t = u_lo + (np.arange(nu_t) + 0.5) * DU_T
th_t = 180 - (np.arange(nt_t) + 0.5) * DT_T                 # image row 0 = theta +180 (uv v = 1)
# radius at the texels: bilinear in the height field (theta wrapped); NaN where a neighbour cell is empty
Rp = np.concatenate([Rg[:, -1:], Rg, Rg[:, :1]], axis=1).astype(np.float32)
mx = ((th_t[:, None] - tcg[0]) / float(hf["dt"]) + 1).astype(np.float32) * np.ones((1, nu_t), np.float32)
my = ((u_t[None, :] - ucg[0]) / du_g).astype(np.float32) * np.ones((nt_t, 1), np.float32)
r_t = cv2.remap(Rp, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
valid = np.isfinite(r_t)
U, TH = np.broadcast_to(u_t[None, :], r_t.shape), np.broadcast_to(th_t[:, None], r_t.shape)
Xt = np.full(r_t.shape + (3,), np.nan, np.float32)
Nt = np.zeros(r_t.shape + (3,), np.float32)
pos, nrm, _ = cl.point(U[valid], TH[valid], r_t[valid])
Xt[valid], Nt[valid] = pos, nrm
del pos, nrm
print(f"texels: {nt_t} x {nu_t} ({DU_T * 1000:.1f} mm x {DT_T} deg), {valid.mean():.1%} on the surface [{time.time() - t0:.0f} s]", flush=True)

# --- cameras ---------------------------------------------------------------------------------------------
fp = FusedPoses()
cam = fp.cam
ks = np.arange(0, len(cam["t"]), EVERY)
sc = np.load(os.path.join(BIG, "scan_corrections.npz"))
tsc = sc["t"].astype(float)


def corr_at(t):
    f = lambda A: np.array([np.interp(float(t), tsc, A[:, j]) for j in range(3)])  # noqa: E731
    return f(sc["d"]), f(sc["w"]), f(sc["O"])


poses = []
for k in ks:
    t = cam["t"][k]
    R, C, _ = fp.pose(t)
    if POSES == "refined":
        d, w, O = corr_at(t)
        C = C + d + np.cross(w, C - O)
        R = R @ _expmap(w).T                          # body rotated by exp(w) in the world
    poses.append((int(cam["idx"][k]), R, C))
ucam = cl.coords(np.array([p[2] for p in poses]))[0]
print(f"{len(poses)} frames, poses: {POSES}; camera u {ucam.min():+.2f} .. {ucam.max():+.2f} m", flush=True)


def load(idx):
    return cv2.imread(os.path.join(EX, "images", "color", f"{idx:06d}.png")).astype(np.float32)


ff_path = os.path.join(BIG, "flatfield.npy")
if os.path.exists(ff_path):
    G = np.load(ff_path)
else:
    stack = np.stack([load(p[0]) for p in poses[::3]])
    G = np.median(stack, axis=0)
    del stack
    G = cv2.GaussianBlur(G, (0, 0), 25)
    G /= np.median(G[(G.sum(2) > 0)], axis=0)
    np.save(ff_path, G)
G = np.maximum(G, 0.05)
print(f"flat field: gain from {G.min():.2f} to {G.max():.2f} (relative to its median) [{time.time() - t0:.0f} s]", flush=True)
GAIN = 0.45 * 255                                      # corrected image = image / G * GAIN / median wall level


def project(k, c0, c1):
    """score, pixel x, y of the texel columns c0:c1 in frame k (score 0 where not seen)"""
    _, R, C = poses[k]
    X = Xt[:, c0:c1]
    q = X - C.astype(np.float32)
    xc = q @ R.T.astype(np.float32)
    z = xc[..., 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        px = K[0, 0] * xc[..., 0] / z + K[0, 2]
        py = K[1, 1] * xc[..., 1] / z + K[1, 2]
        D2 = (q * q).sum(-1)
        cosi = np.abs((q * Nt[:, c0:c1]).sum(-1)) / np.sqrt(D2)
        s = cosi / D2
    ok = (z > 0.03) & (px >= MARGIN) & (px < W - 1 - MARGIN) & (py >= MARGIN) & (py < H - 1 - MARGIN) & (cosi > 0.12)
    ok &= np.isfinite(s)
    return np.where(ok, s, 0.0).astype(np.float32), px.astype(np.float32), py.astype(np.float32)


def cols(k):
    c0 = int(np.clip((ucam[k] + AHEAD[0] - u_lo) / DU_T, 0, nu_t))
    c1 = int(np.clip((ucam[k] + AHEAD[1] - u_lo) / DU_T, 0, nu_t))
    return c0, c1


best = np.zeros(r_t.shape, np.float32)
for k in range(len(poses)):
    c0, c1 = cols(k)
    if c1 <= c0:
        continue
    s, _, _ = project(k, c0, c1)
    np.maximum(best[:, c0:c1], s, out=best[:, c0:c1])
print(f"pass 1 (best view per texel) done [{time.time() - t0:.0f} s]", flush=True)


def frames_with_band(lo, hi):
    """per frame: columns, relative score, mask of lo <= relative score <= hi, pixel coordinates"""
    for k in range(len(poses)):
        c0, c1 = cols(k)
        if c1 <= c0:
            continue
        s, px, py = project(k, c0, c1)
        bsub = best[:, c0:c1]
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = np.where(bsub > 0, s / bsub, 0)
        use = (rel >= lo) & (rel <= hi)
        if use.any():
            yield k, c0, c1, rel, use, px, py


LEVEL = float(np.median((load(poses[0][0]) / G)[H // 4:3 * H // 4:4, W // 8:7 * W // 8:4]))


def corrected(k):
    return load(poses[k][0]) / G * (GAIN / LEVEL)


# --- exposure: one gain per frame from the texels several frames share (sparse texel grid, grey, log) ----
SP = 6
rows_s = np.arange(0, nt_t, SP)
obs = {"t": [], "k": [], "v": [], "w": []}
for k, c0, c1, rel, use, px, py in frames_with_band(0.3, 1.0):
    cs = np.arange(c0 + (-c0) % SP, c1, SP)
    if not len(cs):
        continue
    sub = np.ix_(rows_s, cs - c0)
    m = use[sub]
    if not m.any():
        continue
    grey = cv2.remap(corrected(k).mean(2), np.ascontiguousarray(px[sub]), np.ascontiguousarray(py[sub]),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    obs["t"].append((rows_s[:, None] * nu_t + cs[None, :])[m])
    obs["k"].append(np.full(m.sum(), k))
    obs["v"].append(np.log(np.maximum(grey[m], 1.0)))
    obs["w"].append(rel[sub][m] ** 4)
obs_t, obs_k, obs_v, obs_w = (np.concatenate(obs[x]) for x in ("t", "k", "v", "w"))
_, tix = np.unique(obs_t, return_inverse=True)
gam = np.zeros(len(poses))
for _ in range(40):
    A = np.bincount(tix, obs_w * (obs_v + gam[obs_k])) / np.bincount(tix, obs_w)
    e = A[tix] - obs_v
    num, den = np.bincount(obs_k, obs_w * e, len(poses)), np.bincount(obs_k, obs_w, len(poses))
    g_new = np.where(den > 0, num / np.maximum(den, 1e-9), 0)
    g_new = np.convolve(np.pad(g_new, 2, mode="edge"), np.ones(5) / 5, "valid")   # exposure changes slowly
    gam = 0.5 * gam + 0.5 * (g_new - np.median(g_new))
gain_k = np.exp(gam)
print(f"exposure gains from {len(obs_v):,} shared observations: {gain_k.min():.3f} .. {gain_k.max():.3f} [{time.time() - t0:.0f} s]", flush=True)


def accumulate(lo, hi, power):
    acc = np.zeros(r_t.shape + (3,), np.float32)
    acc2 = np.zeros(r_t.shape + (3,), np.float32)
    wsum = np.zeros(r_t.shape, np.float32)
    nviews = np.zeros(r_t.shape, np.uint16)
    best_rgb = np.full(r_t.shape + (3,), 255.0, np.float32)
    for k, c0, c1, rel, use, px, py in frames_with_band(lo, hi):
        smp = cv2.remap(corrected(k) * gain_k[k], px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        w = np.where(use, rel ** power, 0).astype(np.float32)
        acc[:, c0:c1] += w[..., None] * smp
        acc2[:, c0:c1] += w[..., None] * smp * smp
        wsum[:, c0:c1] += w
        nviews[:, c0:c1] += use
        top = use & (rel >= 0.9999)
        best_rgb[:, c0:c1][top] = smp[top]
    seen = wsum > 0
    tex = np.full(r_t.shape + (3,), 255.0, np.float32)
    tex[seen] = acc[seen] / wsum[seen, None]
    var = np.zeros(r_t.shape, np.float32)
    var[seen] = (acc2[seen] / wsum[seen, None] - tex[seen] ** 2).mean(-1)
    return tex, best_rgb, np.sqrt(np.clip(var, 0, None)), seen, nviews


tex, best_rgb, spread, seen, nviews = accumulate(0.6, 1.0, 8)
tex[~valid] = 255
print(f"pass 2 (near views) done [{time.time() - t0:.0f} s]", flush=True)


def tile_shifts(A, B, size=96):
    """phase-correlation shift of B relative to A per tile (texels: x along u, y around)"""
    ga, gb = (cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) for x in (A, B))
    win = cv2.createHanningWindow((size, size), cv2.CV_32F)
    out = []
    for y in range(40, nt_t - 40 - size, size):
        for x in range(0, nu_t - size, size):
            a_, b_ = ga[y:y + size, x:x + size], gb[y:y + size, x:x + size]
            if (a_ >= 254).mean() > 0.05 or (b_ >= 254).mean() > 0.05 or a_.std() < 2:
                continue
            (dx, dy), resp = cv2.phaseCorrelate(a_, b_, win)
            if resp > 0.15:
                out.append((x, y, dx, dy, resp))
    return np.array(out)


# registration check: the texture from views at about 1.5-2.5 x the best distance against the near one
far, _, _, seen_far, _ = accumulate(0.15, 0.45, 2)
far[~(valid & seen_far)] = 255
sh = tile_shifts(tex, far)
print(f"pass 3 (far views) done, {len(sh)} tiles compared [{time.time() - t0:.0f} s]", flush=True)
multi = seen & (nviews >= 3)
stats = {"poses": POSES, "frames": len(poses), "texels": [int(nt_t), int(nu_t)], "texel_mm": [DU_T * 1000, DT_T],
         "u_range_m": [float(u_lo), float(u_hi)],
         "surface_share": float(valid.mean()), "seen_share_of_surface": float(seen[valid].mean()),
         "views_per_texel_median": float(np.median(nviews[seen])),
         "exposure_gain_range": [float(gain_k.min()), float(gain_k.max())],
         "colour_spread_between_views_grey_levels": {"median": float(np.median(spread[multi])),
                                                     "mean": float(np.mean(spread[multi])),
                                                     "p90": float(np.percentile(spread[multi], 90))},
         "near_vs_far_views_shift": {"tiles": int(len(sh)),
                                     "along_u_mm_median": float(np.median(sh[:, 2]) * DU_T * 1000),
                                     "along_u_mm_abs_p50_p90": [float(np.percentile(np.abs(sh[:, 2]), q) * DU_T * 1000) for q in (50, 90)],
                                     "around_deg_median": float(np.median(sh[:, 3]) * DT_T),
                                     "around_deg_abs_p50_p90": [float(np.percentile(np.abs(sh[:, 3]), q) * DT_T) for q in (50, 90)]}}
json.dump(stats, open(os.path.join(OUT, f"r4_texture_{POSES}.json"), "w"), indent=1)
np.save(os.path.join(BIG, f"exposure_gain_{POSES}.npy"), gain_k)
np.save(os.path.join(BIG, f"near_far_shift_{POSES}.npy"), sh)
cv2.imencode(".png", np.clip(tex, 0, 255).astype(np.uint8))[1].tofile(os.path.join(BIG, f"texture_{POSES}.png"))
cv2.imencode(".png", np.clip(best_rgb, 0, 255).astype(np.uint8))[1].tofile(os.path.join(BIG, f"texture_{POSES}_best.png"))
cv2.imencode(".png", np.clip(far, 0, 255).astype(np.uint8))[1].tofile(os.path.join(BIG, f"texture_{POSES}_far.png"))
np.save(os.path.join(BIG, f"texture_{POSES}_spread.npy"), spread.astype(np.float16))
print(json.dumps(stats, indent=1))
print(f"-> {BIG}/texture_{POSES}.png [{time.time() - t0:.0f} s]")
