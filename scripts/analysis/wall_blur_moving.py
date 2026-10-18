"""How thick is the accumulated LiDAR wall while the robot drives, and what thins it?

usage: python -I wall_blur_moving.py   (then wall_blur_plot.py for the figure)
writes results/inpipe_172011/diag/wall_blur_moving.json and DIAG_BIG/wall_blur_moving.npz

Metric everywhere: wall points binned in 1 cm (along the pipe) x 4 deg cells; residual = r - cell median;
reported as robust sigma and the p5-p95 band ("wall thickness").
  A  current fused world cloud (vision orientation + vision lateral + encoder), FUSED_BIG ply
  B  each scan placed by its OWN cylinder fit (LiDAR lateral + tilt) + encoder s
  C  B + per-beam (ring) bias, self-calibrated on even scans, evaluated on odd scans
  D  C + range-dependent bias (same self-calibration)
  E  D, only 0.25-0.45 m range
"""
import csv, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "fusion"))
import paths
from timebase import load_offsets, stream_times
from pipe_geometry import fit_cylinder, to_pipe_coords
from ply_io import read_ply_vertices
BIG = paths.DIAG_BIG
RES = os.path.join(paths.INPIPE_RESULTS, "diag")
os.makedirs(BIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

EX = paths.INPIPE_EXTRACT
OUT = os.path.join(paths.INPIPE_RESULTS, "fusion")
off = load_offsets(os.path.join(OUT, "time_offsets.json"))
tS, rS = stream_times(EX, off, "lidar")
tE, rE = stream_times(EX, off, "encoders")
sE = np.array([(float(r["v6"]) - float(r["v7"])) / 2 for r in rE])
s_enc = lambda t: np.interp(t, tE, sE)
fits = {int(r["idx"]): r for r in csv.DictReader(open(os.path.join(EX, "lidar_fit_short.csv"))) if r.get("diameter_mm")}
D_med = np.median([float(r["diameter_mm"]) for r in fits.values()])
UP = np.array([0.0, -1, 0])
S_ROBOT = (0.03, 0.35)
S_WALL = (0.25, 0.65)

acc = {k: [] for k in ("s", "th", "r", "ring", "rho", "scan")}
for t, row in zip(tS, rS):
    i = int(row["idx"])
    f = fits.get(i)
    se = s_enc(t)
    if f is None or not (S_ROBOT[0] <= se <= S_ROBOT[1]):
        continue
    if abs(float(f["diameter_mm"]) - D_med) > 12 or float(f["inlier_share"]) < 0.5:
        continue
    a = np.load(os.path.join(EX, "lidar", f"{i:06d}.npy"))
    P = np.column_stack([a["x"], a["y"], a["z"]]).astype(float)
    ok = np.isfinite(P).all(1) & (np.abs(P).sum(1) > 0)
    P, ring = P[ok], a["ring"][ok].astype(int)
    rho = np.linalg.norm(P, axis=1)
    d = np.array([float(f["dir_x"]), float(f["dir_y"]), float(f["dir_z"])])
    up = UP - (UP @ d) * d
    up /= np.linalg.norm(up)
    side = np.cross(d, up)
    p0 = (float(f["off_side_mm"]) * side + float(f["off_up_mm"]) * up) / -1000.0
    Q = P - p0
    s_loc = Q @ d
    rad = np.linalg.norm(Q - np.outer(s_loc, d), axis=1)
    sw = se + s_loc
    m = (rho > 0.2) & (rho < 0.6) & (s_loc > 0.05) & (np.abs(rad - float(f["diameter_mm"]) / 2000) < 0.025) \
        & (sw > S_WALL[0]) & (sw < S_WALL[1])
    th = np.degrees(np.arctan2(Q[m] @ side, Q[m] @ up))
    th = (th - float(f["slot_centre_deg"]) + 360) % 360 - 180
    for k, v in zip(acc, (sw[m], th, rad[m], ring[m], rho[m], np.full(m.sum(), i))):
        acc[k].append(v)
for k in acc:
    acc[k] = np.concatenate(acc[k])
s, th, r, ring, rho, scan = (acc[k] for k in ("s", "th", "r", "ring", "rho", "scan"))
nscan = len(np.unique(scan))
print(f"B: {nscan} scans while the robot drove s={S_ROBOT}, {len(s):,} wall points on s={S_WALL} m")


def cells(s, th):
    return ((s - S_WALL[0]) / 0.01).astype(int) * 1000 + ((th + 180) / 4).astype(int)


def cell_median(c, x):
    order = np.argsort(c, kind="stable")
    cs, xs = c[order], x[order]
    u, start, cnt = np.unique(cs, return_index=True, return_counts=True)
    med = np.array([np.median(xs[a:a + n]) for a, n in zip(start, cnt)])
    lut = dict(zip(u, med))
    return np.array([lut[k] for k in c]), dict(zip(u, cnt))


def report(name, res):
    res = res * 1000
    mad = 1.4826 * np.median(np.abs(res - np.median(res)))
    p5, p95 = np.percentile(res, [5, 95])
    print(f"  {name:62s} robust sigma {mad:5.2f} mm   p5-p95 band {p95 - p5:5.1f} mm")
    return mad, p95 - p5


C = cells(s, th)
odd = scan % 2 == 1
even = ~odd
out = {}
# B: own alignment, raw
med_o, _ = cell_median(C[odd], r[odd])
out["B"] = report("B  per-scan LiDAR cylinder alignment (no bias correction)", r[odd] - med_o)

# self-calibrate per-ring and per-range bias on EVEN scans
rbins = np.arange(0.2, 0.625, 0.025)
def calibrate(use_range):
    b_ring = np.zeros(128)
    b_rng = np.zeros(len(rbins))
    re, ce, rge, rhoe = r[even], C[even], ring[even], rho[even]
    ib = np.clip(np.digitize(rhoe, rbins) - 1, 0, len(rbins) - 1)
    for _ in range(5):
        corr = re - b_ring[rge] - (b_rng[ib] if use_range else 0)
        med, _ = cell_median(ce, corr)
        e = re - med
        for q in np.unique(rge):
            mq = rge == q
            b_ring[q] = np.median(e[mq] - (b_rng[ib[mq]] if use_range else 0))
        b_ring -= np.median(b_ring[np.unique(rge)])
        if use_range:
            for k in range(len(rbins)):
                mk = ib == k
                if mk.sum() > 100:
                    b_rng[k] = np.median(e[mk] - b_ring[rge[mk]])
    return b_ring, b_rng

b_ring, _ = calibrate(False)
used = np.unique(ring[even])
print(f"  per-ring bias (self-calibrated, {len(used)} rings): std {1000*np.std(b_ring[used]):.1f} mm, "
      f"range {1000*b_ring[used].min():+.1f} .. {1000*b_ring[used].max():+.1f} mm")
ro = r[odd] - b_ring[ring[odd]]
med_o, _ = cell_median(C[odd], ro)
out["C"] = report("C  + per-ring bias (calibrated on even scans, tested on odd)", ro - med_o)

b_ring2, b_rng = calibrate(True)
ib_o = np.clip(np.digitize(rho[odd], rbins) - 1, 0, len(rbins) - 1)
ro2 = r[odd] - b_ring2[ring[odd]] - b_rng[ib_o]
med_o2, _ = cell_median(C[odd], ro2)
out["D"] = report("D  + per-ring + range-dependent bias", ro2 - med_o2)
print("     range bias (mm) per 2.5 cm bin from 0.2 m: " + " ".join(f"{1000*b:+.1f}" for b in b_rng))
mm = (rho[odd] > 0.25) & (rho[odd] < 0.45)
med_o3, _ = cell_median(C[odd][mm], ro2[mm])
out["E"] = report("E  D, only 0.25-0.45 m range", ro2[mm] - med_o3)
# random part only: same scan, same cell -> spread inside one scan
key = C[odd] * 100000 + scan[odd]
med_s, _ = cell_median(key, ro2)
report("   (inside ONE scan, same cell: random noise + sub-cell shape)", ro2 - med_s)


# split: range selection alone vs calibration alone
mm = (rho[odd] > 0.25) & (rho[odd] < 0.45)
raw = r[odd][mm]
m1, _ = cell_median(C[odd][mm], raw)
report("B' raw, only 0.25-0.45 m range (no bias correction)", raw - m1)
rc = (r[odd] - b_ring[ring[odd]])[mm]
m2, _ = cell_median(C[odd][mm], rc)
report("C' per-ring only, 0.25-0.45 m range", rc - m2)
print(f"  share of odd-scan points kept by 0.25-0.45 m: {mm.mean():.0%}")
# how many of the cells keep >=5 points in E vs D
_, cntD = cell_median(C[odd], ro2)
_, cntE = cell_median(C[odd][mm], ro2[mm])
nt = 360 // 4
ncell = int((S_WALL[1] - S_WALL[0]) / 0.01) * nt
print(f"  cells (1 cm x 4 deg) with >=5 points: all ranges {sum(v>=5 for v in cntD.values())}/{ncell}, 0.25-0.45 m {sum(v>=5 for v in cntE.values())}/{ncell}")

# A: the current fused world cloud, same stretch of wall
legs = json.load(open(os.path.join(OUT, "vision_legs.json")))
lam = json.load(open(os.path.join(OUT, "f4_tag_scale.json")))["mm_per_unit_tag"] / 1000
V = read_ply_vertices(os.path.join(paths.FUSED_BIG, "lidar_coloured_world.ply"))
X = np.column_stack([V["x"], V["y"], V["z"]]).astype(float)
a1 = np.array(legs["legs"][0]["dir"])
B = np.array(legs["bend_point"]) * lam
u1 = (X - B) @ a1
# leg 1 runs up to the bend at u1 = 0; the wall stretch S_WALL lies before the bend: pick the same 0.5 m length
# measured back from the bend by the encoder-based arc of the bend (s_bend ~ 0.9 m)
u0 = -legs["bend_point_path_units"] * lam   # first camera's arc position (robot start), metres from the bend
mA = (u1 > u0 + S_WALL[0]) & (u1 < u0 + S_WALL[1])
XA = X[mA]
fA = fit_cylinder(XA, d0=a1, thresh=0.025, max_points=20000)
sA, thA, rA = to_pipe_coords(XA, fA["point"], fA["dir"], ref_up=(0, 0, 1))
okA = np.abs(rA - fA["radius"]) < 0.025
sA, thA, rA = sA[okA], np.degrees(thA[okA]), rA[okA]
CA = (((sA - sA.min()) / 0.01).astype(int) * 1000 + ((thA + 180) / 4).astype(int))
medA, _ = cell_median(CA, rA)
print(f"A: fused world cloud (4 M-point subsample), same 0.5 m of leg 1: {len(rA):,} wall points")
out["A"] = report("A  current fused world cloud (vision pose + vision lateral)", rA - medA)
json.dump({k: {"robust_sigma_mm": v[0], "band_p5_p95_mm": v[1]} for k, v in out.items()},
          open(os.path.join(RES, "wall_blur_moving.json"), "w"), indent=1)
np.savez(os.path.join(BIG, "wall_blur_moving.npz"), b_ring=b_ring2, b_rng=b_rng, rbins=rbins,
         s=s, th=th, r=r, ring=ring, rho=rho, scan=scan)
