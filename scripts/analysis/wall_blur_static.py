"""usage: python -I wall_blur_static.py [N_FRAMES=150] [FIRST=20]   (scans 0..~210: robot parked at the pipe mouth)

Decompose the LiDAR wall blur on the static start (robot not moving, frames 0..N-1).

Organized cloud: index k = col*128 + ring. With the robot static, each (ring, col) hits the same wall spot
every frame, so
  temporal std of r per (ring,col)      -> random range noise
  time-mean r minus a smooth pipe model -> systematic error (per-beam bias, range/incidence bias) + real surface
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths
BIG = paths.DIAG_BIG
os.makedirs(BIG, exist_ok=True)
from pipe_geometry import fit_cylinder, to_pipe_coords

EX = os.path.join(paths.INPIPE_EXTRACT, "lidar")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
F0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
X = np.empty((N, 230400, 3), np.float32)
ring = None
for n in range(N):
    a = np.load(os.path.join(EX, f"{F0 + n:06d}.npy"))
    X[n] = np.column_stack([a["x"], a["y"], a["z"]])
    if ring is None:
        ring = a["ring"].astype(int)
        inten = a["intensity"]
valid = np.isfinite(X).all(2) & (np.abs(X).sum(2) > 0)
vfrac = valid.mean(0)
keep = vfrac > 0.9
Xm = np.where(valid[..., None], X, np.nan)
mean = np.nanmean(Xm[:, keep], 0)
rng_m = np.linalg.norm(mean, axis=1)
sel = (rng_m > 0.2) & (rng_m < 0.6)
P = mean[sel].astype(float)
f = fit_cylinder(P, d0=np.array([0, 0, 1.0]), thresh=0.02)
d = f["dir"] if f["dir"][2] > 0 else -f["dir"]
s, th, r = to_pipe_coords(P, f["point"], d, ref_up=(0, -1, 0))
wall = np.abs(r - f["radius"]) < 0.025
print(f"frames {F0}..{F0+N-1}; points kept {keep.sum()}, 0.2-0.6 m {sel.sum()}, wall {wall.sum()}; R {1000*f['radius']:.1f} mm")

# smooth model: Fourier in theta up to order 3, each coefficient linear in s (ellipse + taper + tilt)
def design(s, th):
    cols = []
    for k in range(4):
        for g in ([np.ones_like(th)] if k == 0 else [np.cos(k * th), np.sin(k * th)]):
            cols += [g, g * s]
    return np.column_stack(cols)
A = design(s[wall], th[wall])
coef, *_ = np.linalg.lstsq(A, r[wall], rcond=None)
res_sys = r[wall] - A @ coef  # time-mean residual (systematic + real surface)
for _ in range(2):
    ok = np.abs(res_sys) < 4 * 1.4826 * np.median(np.abs(res_sys))
    coef, *_ = np.linalg.lstsq(A[ok], r[wall][ok], rcond=None)
    res_sys = r[wall] - A @ coef
ok = np.abs(res_sys) < 4 * 1.4826 * np.median(np.abs(res_sys))

# temporal noise: per point, radial coordinate per frame
idx_keep = np.flatnonzero(keep)[sel][wall]
Pt = Xm[:, idx_keep].astype(float)  # N x M x 3
M = Pt.shape[1]
st, tht, rt = to_pipe_coords(Pt.reshape(-1, 3), f["point"], d, ref_up=(0, -1, 0))
rt = rt.reshape(N, M)
rng_t = np.linalg.norm(Pt, axis=2)
tstd_r = np.nanstd(rt, 0)
tstd_range = np.nanstd(rng_t, 0)

def rms(x):
    return 1000 * np.sqrt(np.nanmean(x ** 2))
mad = lambda x: 1000 * 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x)))
print(f"\n[1] random noise (frame-to-frame std of each beam, robot static)")
print(f"    range std median {1000*np.nanmedian(tstd_range):.2f} mm, radial std median {1000*np.nanmedian(tstd_r):.2f} mm, p90 {1000*np.nanpercentile(tstd_r,90):.2f} mm")
print(f"[2] systematic (time-averaged point minus smooth ellipse/taper model)")
print(f"    rms {rms(res_sys[ok]):.2f} mm, robust sigma {mad(res_sys):.2f} mm")
# single frame residual vs model
r1 = rt[N // 2]
res1 = r1 - A @ coef
print(f"[3] single frame vs same model: rms {rms(res1[ok & np.isfinite(res1)]):.2f} mm, robust sigma {mad(res1[np.isfinite(res1)]):.2f} mm")

# per-ring bias
rg = ring[idx_keep]
rings = np.unique(rg)
rb = np.array([np.median(res_sys[ok & (rg == q)]) if (ok & (rg == q)).sum() > 30 else np.nan for q in rings])
good = np.isfinite(rb)
print(f"[4] per-beam (ring) bias: {good.sum()} rings with wall hits; std {1000*np.nanstd(rb):.2f} mm, range {1000*np.nanmin(rb):.1f} .. {1000*np.nanmax(rb):.1f} mm")
corr = res_sys - np.interp(rg, rings[good], rb[good])
print(f"    after removing per-ring bias: rms {rms(corr[ok]):.2f} mm, robust sigma {mad(corr):.2f} mm")

# range dependence of the remaining residual
rngw = rng_m[sel][wall]
print("[5] residual vs measured range (after ring bias), median / robust sigma, mm:")
for lo, hi in [(0.2, 0.25), (0.25, 0.3), (0.3, 0.35), (0.35, 0.4), (0.4, 0.5), (0.5, 0.6)]:
    m = ok & (rngw >= lo) & (rngw < hi)
    if m.sum() > 50:
        print(f"    {lo:.2f}-{hi:.2f} m  n={m.sum():6d}  median {1000*np.median(corr[m]):+6.2f}  sigma {mad(corr[m]):5.2f}   raw-median {1000*np.median(res_sys[m]):+6.2f}")
# incidence angle: angle between beam and wall normal (radial)
u = mean[sel][wall] / rngw[:, None]
Q = P[wall] - f["point"]
nrm = Q - np.outer(Q @ d, d)
nrm /= np.linalg.norm(nrm, axis=1)[:, None]
inc = np.degrees(np.arccos(np.clip(np.abs(np.sum(u * nrm, 1)), 0, 1)))
print("[6] incidence angle (0 = beam hits wall head-on) vs residual:")
for lo, hi in [(0, 40), (40, 55), (55, 65), (65, 75), (75, 90)]:
    m = ok & (inc >= lo) & (inc < hi)
    if m.sum() > 50:
        print(f"    {lo:2d}-{hi:2d} deg n={m.sum():6d}  median {1000*np.median(corr[m]):+6.2f}  sigma {mad(corr[m]):5.2f}  noise(temporal) {1000*np.nanmedian(tstd_r[m]):.2f}")
print(f"    incidence median {np.median(inc):.0f} deg")
# angular resolution footprint on the wall (spacing between neighbouring points)
np.savez(os.path.join(BIG, f"wall_blur_static_{F0}.npz"), s=s[wall], th=th[wall], r=r[wall],
         res_sys=res_sys, corr=corr, rng=rngw, inc=inc, ring=rg, tstd=tstd_r)
