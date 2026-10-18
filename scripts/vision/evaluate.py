"""Evaluation helpers for the pipe reconstruction.

Usage:
  evaluate.py counter-templates VIDEO OUT_PNG
      Group the digits of the on-screen distance counter into glyph clusters and save a picture of them.
  evaluate.py counter VIDEO OUT_CSV "0=0,1=7,..." [--nearest]
      Decode the counter of every frame (ground-truth travel) using cluster->digit labels read off that picture.
      --nearest: unlabelled clusters take the digit of the closest labelled one (bright videos: label the big ones)
  evaluate.py sparse CONVERTER_EXE SFM_DATA_BIN FRAMES_CSV COUNTER_CSV OUT_PREFIX
      Summarise an OpenMVG reconstruction: registered frames, intrinsics, pipe fit, scale from the
      300 mm pipe diameter, estimated travel versus the counter. Writes OUT_PREFIX.json / .png.
  evaluate.py dense CONVERTER_EXE SFM_DATA_BIN FRAMES_CSV COUNTER_CSV DENSE_PLY OUT_PREFIX
      Pipe geometry of the dense point cloud: diameter profile along the pipe, wall scatter, coverage of
      the circumference, end-cap position, and an unrolled picture of the pipe wall.
"""
import csv
import json
import os
import subprocess
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# nominal bore, the only scale of a vision-only model: 300 mm for the Sheffield pipes; another pipe: set the
# environment variable PIPE3D_PIPE_DIAMETER_M (also used by render_views.py and video_compare.py)
PIPE_DIAMETER_M = float(os.environ.get("PIPE3D_PIPE_DIAMETER_M", "0.300"))

# ----------------------------------------------------------------------------- counter ----------
X0, X1, Y0, Y1 = 280, 440, 552, 576          # crop of the original 720x576 frame that holds "D.DD m"
CELLS = [(38, 50), (61, 73), (73, 85)]       # column ranges (inside the crop) of the three digits
CELLS_10M = [(38, 50), (50, 62), (73, 85), (84, 96)]   # "DD.DD m" from 10 m on: the text grows to the right
DOT_CELL = (61, 73)                          # holds a digit below 10 m and the decimal point from 10 m on
MANUAL = {47: 0.00, 476: 0.74, 953: 1.58, 1430: 2.54, 1907: 3.54, 2312: 4.18}  # read by eye, CLOSED_1 only


def cluster_digits(video, thr=12.0):
    """Group the digit images of all frames into glyph clusters (no OCR library needed).
    Returns per-frame cluster ids (3 per frame below 10 m, 4 from 10 m on, none when the strip is
    overexposed) and the cluster mean images, in order of first appearance."""
    import cv2
    cap = cv2.VideoCapture(video)
    ids, templates, counts = [], [], []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(f[Y0:Y1, X0:X1], cv2.COLOR_BGR2GRAY)
        # white text on the dark image: fixed threshold. When something bright lies behind the text
        # (manhole benching, glare off the end cap) the threshold moves up between background and text.
        p50, p99 = np.percentile(g, [50, 99])
        b = (g > (140 if p50 < 100 else 0.5 * (p50 + p99))).astype(np.float32)
        if b.mean() > 0.4:                   # still no separation of text and background: unreadable
            ids.append([])
            continue
        row = []
        # rows 0-5 lie above the characters; bright background shows through there, so leave them out
        for a, c in (CELLS_10M if b[6:, DOT_CELL[0]:DOT_CELL[1]].sum() < 12 else CELLS):
            patch = b[:, a:c]
            dist = [np.abs(patch - (t > 0.5)).sum() for t in templates]
            k = int(np.argmin(dist)) if dist else -1
            if k < 0 or dist[k] > thr:
                templates.append(patch.copy())
                counts.append(1)
                k = len(templates) - 1
            else:
                templates[k] = (templates[k] * counts[k] + patch) / (counts[k] + 1)
                counts[k] += 1
            row.append(k)
        ids.append(row)
    cap.release()
    return ids, np.stack(templates), counts


def cmd_counter_templates(video, out_png):
    """Save a picture of the glyph clusters so that each can be given its digit by eye."""
    import cv2
    ids, templates, counts = cluster_digits(video)
    tiles = []
    for k, (t, c) in enumerate(zip(templates, counts)):
        big = cv2.resize((t * 255).astype(np.uint8), None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
        big = cv2.copyMakeBorder(big, 30, 4, 4, 4, cv2.BORDER_CONSTANT, value=60)
        cv2.putText(big, f"#{k} n={c}", (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 255, 1)
        tiles.append(big)
    per_row = 8
    while len(tiles) % per_row:
        tiles.append(np.full_like(tiles[0], 60))
    sheet = np.vstack([np.hstack(tiles[i:i + per_row]) for i in range(0, len(tiles), per_row)])
    cv2.imencode(".png", sheet)[1].tofile(out_png)
    print(f"{len(ids)} frames, {len(templates)} glyph clusters, sizes {counts}")
    print("saved", out_png)


def cmd_counter(video, out_csv, labels, nearest=False):
    """labels: 'clusterid=digit,...' as read off the cluster picture. Unlabelled clusters count as unreadable, or
    with nearest=True (bright backgrounds split the digits into hundreds of small clusters: label only the big ones)
    take the digit of the closest labelled cluster when it differs in at most 25 of the cell's pixels."""
    lab = {int(a): int(b) for a, b in (kv.split("=") for kv in labels.split(","))}
    ids, templates, counts = cluster_digits(video)
    if nearest:
        known = sorted(lab)
        for k in range(len(templates)):
            if k not in lab:
                d = [np.abs((templates[k] > 0.5).astype(float) - (templates[j] > 0.5)).sum() for j in known]
                if min(d) <= 25:
                    lab[k] = lab[known[int(np.argmin(d))]]
    values = np.full(len(ids), np.nan)
    for i, row in enumerate(ids):
        if row and all(int(k) in lab for k in row):
            values[i] = int("".join(str(lab[int(k)]) for k in row)) / 100
    unreadable = int(np.isnan(values).sum())
    for i in range(len(values)):          # an unreadable frame keeps the previous reading
        if np.isnan(values[i]):
            values[i] = values[i - 1] if i else 0.0
    if nearest:   # nearest-glyph readings have occasional wrong digits: the counter only runs forward, so a reading
        # more than 0.1 m off the median of the 51 frames around it is replaced by that median, then made monotonic
        pad = np.pad(values, 25, mode="edge")
        med = np.median(np.lib.stride_tricks.sliding_window_view(pad, 51), axis=1)
        fixed = np.abs(values - med) > 0.1
        values = np.maximum.accumulate(np.where(fixed, med, values))
        print(f"clean-up: {int(fixed.sum())} readings replaced by the running median, then forced monotonic")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["source_frame", "counter_m"])
        wr.writerows((i, f"{v:.2f}") for i, v in enumerate(values))
    d = np.round(np.diff(values), 2)
    steps = d[d != 0]
    print(f"frames: {len(values)}; unreadable: {unreadable}; counter first {values[0]:.2f} m, last {values[-1]:.2f} m")
    print(f"counter changes: {len(steps)}; step sizes seen: "
          f"{dict(zip(*[x.tolist() for x in np.unique(steps, return_counts=True)]))}; backward steps: {int((steps < 0).sum())}")
    if os.path.basename(video).endswith("CLOSED_1.mp4"):
        bad = {k: (v, float(values[k])) for k, v in MANUAL.items() if abs(values[k] - v) > 0.011}
        print("check against 6 values read by eye:", "all agree" if not bad else f"MISMATCH {bad}")
    print("saved", out_csv)


# ------------------------------------------------------------------------------ sparse ----------
def load_sfm(converter, sfm_bin):
    out_json = os.path.splitext(sfm_bin)[0] + "_full.json"
    subprocess.run([converter, "-i", sfm_bin, "-o", out_json, "-V", "-I", "-E", "-S"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    d = json.load(open(out_json, encoding="utf-8"))
    views = {}
    for v in d["views"]:
        x = v["value"]["ptr_wrapper"]["data"]
        views[x["id_view"]] = (x["filename"], x["id_pose"])
    intr = d["intrinsics"][0]["value"]
    intr_data = intr["ptr_wrapper"]["data"]
    poses = {e["key"]: np.array(e["value"]["center"], float) for e in d["extrinsics"]}
    pts = np.array([s["value"]["X"] for s in d["structure"]], float).reshape(-1, 3)
    nobs = np.array([len(s["value"]["observations"]) for s in d["structure"]], int)
    obs = [np.array([o["key"] for o in s["value"]["observations"]], int) for s in d["structure"]]
    return views, intr.get("polymorphic_name", "?"), intr_data, poses, pts, nobs, obs


def fit_circle(uv):
    """Robust algebraic circle fit in 2-D. Returns centre, radius, inlier mask."""
    keep = np.ones(len(uv), bool)
    for _ in range(6):
        a = np.c_[2 * uv[keep], np.ones(keep.sum())]
        b = (uv[keep] ** 2).sum(1)
        sol, *_ = np.linalg.lstsq(a, b, rcond=None)
        c = sol[:2]
        r = np.sqrt(max(sol[2] + c @ c, 1e-12))
        res = np.hypot(*(uv - c).T) - r
        mad = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep]))) + 1e-12
        new = np.abs(res) < 2.5 * mad
        if new.sum() < 20 or (new == keep).all():
            break
        keep = new
    return c, r, keep


def fit_pipe(pts, d0):
    """Cylinder fit: search the axis direction around d0 that makes the cross-section most circular."""
    d0 = d0 / np.linalg.norm(d0)
    e1 = np.cross(d0, [1.0, 0, 0] if abs(d0[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d0, e1)
    origin = np.median(pts, axis=0)
    best = None
    for span, step in ((0.20, 0.04), (0.04, 0.008), (0.008, 0.002)):
        ca, cb = (best[1], best[2]) if best else (0.0, 0.0)
        for a in np.arange(ca - span, ca + span + 1e-9, step):
            for b in np.arange(cb - span, cb + span + 1e-9, step):
                d = d0 + a * e1 + b * e2
                d /= np.linalg.norm(d)
                u = np.cross(d, e2)
                u /= np.linalg.norm(u)
                v = np.cross(d, u)
                uv = np.c_[(pts - origin) @ u, (pts - origin) @ v]
                c, r, keep = fit_circle(uv)
                res = np.hypot(*(uv[keep] - c).T) - r
                cost = np.sqrt((res ** 2).mean()) / r - 0.05 * keep.mean()
                if best is None or cost < best[0]:
                    best = (cost, a, b, d, u, v, c, r, keep)
    _, _, _, d, u, v, c, r, keep = best
    axis_point = origin + c[0] * u + c[1] * v
    return d, axis_point, r, keep, u, v


def slice_edges(a, b, radius, n=12):
    """Slice boundaries along the pipe between positions a and b (model units): n slices for a short run
    (the 5 m videos); for a run longer than 50 pipe radii, slices of about 2.3 radii (0.35 m) so that the
    manholes and joints of a longer network fall into slices of their own."""
    if b - a > 50 * radius:
        n = int(round((b - a) / (2.33 * radius)))
    return np.linspace(a, b, n + 1)


def pipe_frame(converter, work, part, frames, counter, straighten=True):
    """Pipe coordinates of one reconstruction (WORK/sfm/recon, or recon_PART for a separately reconstructed
    stretch): returns a dict with the camera track and `to_pipe`, a function mapping model points to
      axial position  along the fitted pipe axis, in metres, shifted so that the cameras agree with the counter
      angle           around the axis in degrees, 0 = up as seen by the camera, positive to the camera's right
      radius          distance from the centre of the cross-section, in metres
    straighten=True re-sets scale and centre slice by slice from the 300 mm bore (slices that are not a 300 mm
    circle - manholes - take the values of their neighbours); False keeps the model rigid: one scale for the
    whole reconstruction (median slice radius = 150 mm) and one straight axis."""
    half = PIPE_DIAMETER_M / 2
    sfm_bin = os.path.join(work, "sfm", f"recon_{part}" if part else "recon", "sfm_data.bin")
    views, _, intr, centres, spts, nobs, _ = load_sfm(converter, sfm_bin)
    full = json.load(open(os.path.splitext(sfm_bin)[0] + "_full.json", encoding="utf-8"))
    rot = {e["key"]: np.array(e["value"]["rotation"], float) for e in full["extrinsics"]}
    reg = sorted((frames[name], vid, pid) for vid, (name, pid) in views.items() if pid in centres)
    src = np.array([r[0] for r in reg])
    cams = np.array([centres[r[2]] for r in reg])
    mean_c = cams.mean(axis=0)
    _, _, vt = np.linalg.svd(cams - mean_c)
    d0 = vt[0] * np.sign((cams[-1] - cams[0]) @ vt[0])
    use = spts[nobs >= 3] if (nobs >= 3).sum() >= 200 else spts
    d, axis_pt, radius, _, u, v = fit_pipe(use, d0)
    d *= np.sign(d @ d0)
    off_axis = (cams - axis_pt) - np.outer((cams - axis_pt) @ d, d)
    if np.median(np.linalg.norm(off_axis, axis=1)) > 0.8 * radius:
        # the cameras lie outside the fitted cylinder: one straight cylinder does not describe this model (e.g. a long
        # real sewer that bends a little, with water and deposits on the invert). Take the axis along the camera path
        # and the radius from short sections about it; with straighten=True the slices below then follow the bends.
        print(f"pipe frame: cameras outside the fitted cylinder (radius {radius:.2f} units), axis from the camera path")
        d, axis_pt = d0, mean_c
        u = np.cross(d, [1.0, 0, 0] if abs(d[0]) < 0.9 else [0, 1.0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(d, u)
        rel = use - axis_pt
        s = rel @ d
        r0 = float(np.median(np.linalg.norm(rel - np.outer(s, d), axis=1)))
        radii = []
        for a in np.arange(s.min(), s.max(), 2.0 * r0):
            m = (s >= a) & (s < a + 2.0 * r0)
            if m.sum() >= 40:
                _, r_k, k_in = fit_circle(np.c_[rel[m] @ u, rel[m] @ v])
                if k_in.mean() >= 0.4:
                    radii.append(r_k)
        radius = float(np.median(radii)) if radii else r0
    s_cam, s_sp = (cams - axis_pt) @ d, (use - axis_pt) @ d
    # cross-section slice by slice (about 0.3 m): centre and radius where the slice is a clean circle
    lo, hi = s_cam.min(), np.percentile(s_sp, 99)
    edges = np.linspace(lo, hi, max(4, int(round((hi - lo) / (2.0 * radius)))) + 1)
    pos, rad, cu, cv, ok = [], [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (s_sp >= a) & (s_sp < b)
        pos.append(0.5 * (a + b))
        if m.sum() < 40:
            rad.append(np.nan), cu.append(np.nan), cv.append(np.nan), ok.append(False)
            continue
        c, r_k, k_in = fit_circle(np.c_[(use[m] - axis_pt) @ u, (use[m] - axis_pt) @ v])
        rad.append(r_k), cu.append(c[0]), cv.append(c[1]), ok.append(k_in.mean() >= 0.4)
    pos, rad, cu, cv, ok = (np.array(x) for x in (pos, rad, cu, cv, ok))
    med = np.median(rad[ok])
    ok &= np.abs(rad / med - 1) < 0.12            # a manhole or a bad fit is not a 300 mm circle
    if straighten:
        r_at = lambda x: np.interp(x, pos[ok], rad[ok])        # noqa: E731
        cu_at = lambda x: np.interp(x, pos[ok], cu[ok])        # noqa: E731
        cv_at = lambda x: np.interp(x, pos[ok], cv[ok])        # noqa: E731
    else:
        med_ok = float(np.median(rad[ok]))
        r_at = lambda x: np.full(np.shape(x), med_ok)          # noqa: E731
        cu_at = cv_at = lambda x: np.zeros(np.shape(x))        # noqa: E731
    grid = np.linspace(min(lo, s_sp.min()) - 5 * radius, max(hi, s_sp.max()) + 5 * radius, 6000)
    metric = np.r_[0, np.cumsum(np.diff(grid) * half / r_at(0.5 * (grid[1:] + grid[:-1])))]
    x_loc = lambda x: np.interp(x, grid, metric)           # noqa: E731
    # axial offset from the counter (frames where the counter is running)
    x_cam = x_loc(s_cam)
    run = counter[src] >= 0.25
    offset = float(np.median((counter[src] - x_cam)[run if run.sum() >= 10 else slice(None)]))
    # "up" = the cameras' image-up direction, projected into the cross-section
    up = np.mean([rot[r[2]].T @ np.array([0.0, -1.0, 0.0]) for r in reg], axis=0)
    up -= (up @ d) * d
    up /= np.linalg.norm(up)
    right = np.cross(d, up)

    def to_pipe(p):
        s = (p - axis_pt) @ d
        rel = (p - axis_pt) - np.outer(s, d) - np.outer(cu_at(s), u) - np.outer(cv_at(s), v)
        a, b = rel @ up, rel @ right
        return x_loc(s) + offset, np.degrees(np.arctan2(b, a)), np.hypot(a, b) * half / r_at(s)

    x, th, r = to_pipe(cams)
    # points a short way in front of and above each camera (viewing direction and image-up, in pipe coordinates)
    fwd = np.array([rot[q[2]][2] for q in reg])
    img_up = np.array([-rot[q[2]][1] for q in reg])
    return dict(part=part, src=src, cam_x=x, cam_theta=th, cam_r=r, to_pipe=to_pipe, offset=offset,
                cam_ahead=to_pipe(cams + 0.3 * radius * fwd), cam_above=to_pipe(cams + 0.3 * radius * img_up),
                intrinsics=intr, slice_x=x_loc(pos) + offset, slice_rad=rad / med, slice_ok=ok, n_views=len(views),
                points=len(spts))


def cmd_sparse(converter, sfm_bin, frames_csv, counter_csv, out_prefix):
    views, model, intr, poses, pts, nobs, obs = load_sfm(converter, sfm_bin)
    frames = {r["file"]: int(r["source_frame"]) for r in csv.DictReader(open(frames_csv, encoding="utf-8"))}
    counter = np.array([float(r["counter_m"]) for r in csv.DictReader(open(counter_csv, encoding="utf-8"))])
    reg = sorted((frames[name], vid, poses[pid]) for vid, (name, pid) in views.items() if pid in poses)
    out = {"frames_total": len(views), "frames_registered": len(reg), "points": int(len(pts))}
    print(f"registered frames: {len(reg)} of {len(views)};  3-D points: {len(pts)}")
    print(f"camera model {model}: focal {intr.get('focal_length'):.1f} px, principal point "
          f"({intr['principal_point'][0]:.1f}, {intr['principal_point'][1]:.1f}), distortion "
          f"{[round(k, 4) for k in intr.get('disto_k3', intr.get('disto_k1', []))]}")
    out["focal"] = intr.get("focal_length")
    out["principal_point"] = intr["principal_point"]
    if len(reg) < 5 or len(pts) < 50:
        print("too little was reconstructed to fit the pipe")
        json.dump(out, open(out_prefix + ".json", "w"), indent=1)
        return
    src = np.array([r[0] for r in reg])
    cams = np.array([r[2] for r in reg])
    # longest run of consecutively registered frames (in extraction order)
    order = sorted(frames.values())
    pos = {f: k for k, f in enumerate(order)}
    idx = np.array([pos[f] for f in src])
    breaks = np.where(np.diff(idx) > 1)[0]
    runs = np.split(np.arange(len(idx)), breaks + 1)
    longest = max(runs, key=len)
    print(f"registered range: source frames {src[0]}..{src[-1]}; {len(runs)} run(s); longest run {len(longest)} frames")
    out["runs"], out["longest_run"] = len(runs), int(len(longest))

    # pipe axis: start from the camera path direction, refine with the structure points
    mean_c = cams.mean(axis=0)
    _, _, vt = np.linalg.svd(cams - mean_c)
    d0 = vt[0] * np.sign((cams[-1] - cams[0]) @ vt[0])
    lateral = np.linalg.norm((cams - mean_c) - np.outer((cams - mean_c) @ d0, d0), axis=1)
    path_len = float(np.linalg.norm(np.diff(cams, axis=0), axis=1).sum())
    end_to_end = float(np.linalg.norm(cams[-1] - cams[0]))
    good = nobs >= 3
    use = pts[good] if good.sum() >= 200 else pts
    d, axis_pt, radius, keep, u, v = fit_pipe(use, d0)
    d *= np.sign(d @ d0)
    scale = (PIPE_DIAMETER_M / 2) / radius
    s_cam = (cams - axis_pt) @ d
    off_cam = np.linalg.norm((cams - axis_pt) - np.outer(s_cam, d), axis=1) * scale
    s_pts = (use - axis_pt) @ d
    r_pts = np.linalg.norm((use - axis_pt) - np.outer(s_pts, d), axis=1)
    res = (r_pts[keep] - radius) * scale
    print(f"camera path: length {path_len:.3f} units, end-to-end {end_to_end:.3f} units "
          f"(straightness {end_to_end / max(path_len, 1e-9):.3f}), sideways RMS {np.sqrt((lateral ** 2).mean()):.4f} units")
    print(f"pipe fit on {len(use)} points (seen in >=3 frames): radius {radius:.4f} units, "
          f"{keep.mean():.0%} within tolerance, wall scatter {np.sqrt((res ** 2).mean()) * 1000:.1f} mm RMS")
    print(f"scale from pipe diameter: {PIPE_DIAMETER_M} m / {2 * radius:.4f} units = {scale:.4f} m per unit")
    print(f"camera distance from pipe axis: {off_cam.mean() * 1000:.0f} mm on average (pipe radius is 150 mm)")
    # scale drift: fit the circle independently in slices along the pipe (a monocular reconstruction
    # may shrink or grow as it goes, which shows up as a changing pipe radius in model units)
    edges = slice_edges(s_cam.min(), np.percentile(s_pts, 99), radius)
    sl_pos, sl_rad = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (s_pts >= a) & (s_pts < b)
        if m.sum() < 80:
            continue
        uv = np.c_[(use[m] - axis_pt) @ u, (use[m] - axis_pt) @ v]
        _, r_k, k_in = fit_circle(uv)
        if k_in.mean() >= 0.4:
            sl_pos.append(0.5 * (a + b))
            sl_rad.append(r_k)
    sl_pos, sl_rad = np.array(sl_pos), np.array(sl_rad)
    print("pipe radius in model units, slice by slice along the pipe (constant = no scale drift):")
    print("   " + "  ".join(f"{x:.3f}" for x in sl_rad) + f"   -> last/first = {sl_rad[-1] / sl_rad[0]:.2f}")

    def radius_at(s):
        return np.interp(s, sl_pos, sl_rad)

    # how far ahead of a camera are the points it observes, in pipe radii? (a camera is localised against
    # those points, so its motion carries their scale, not the scale of the wall right beside it)
    s_of_view = {vid: sc for (_, vid, _), sc in zip(reg, s_cam)}
    ahead = []
    for k in np.where(good if good.sum() >= 200 else np.ones(len(pts), bool))[0][::3]:
        sp = (pts[k] - axis_pt) @ d
        ahead += [(sp - s_of_view[vv]) / radius_at(sp) for vv in obs[k] if vv in s_of_view]
    look = float(np.median(ahead)) if ahead else 4.5
    print(f"observed points lie a median of {look:.1f} pipe radii ({look * 0.15:.2f} m) ahead of the camera")

    # ground truth: the counter lags when the robot starts (slack cable), so compare over the stretch
    # where it is running: from the first frame showing >= 0.25 m to the frame where it reaches its final value
    cnt = counter[src]
    i1 = int(np.argmax(cnt >= 0.25)) if (cnt >= 0.25).any() else 0
    i2 = int(np.argmax(cnt >= cnt.max() - 1e-9))
    gt = float(cnt[i2] - cnt[i1])
    ds = np.diff(s_cam)
    mid = 0.5 * (s_cam[1:] + s_cam[:-1])
    half = PIPE_DIAMETER_M / 2
    r_local = radius_at(mid + look * radius_at(mid))
    cum_global = np.r_[0, np.cumsum(ds * half / np.median(sl_rad))]
    cum_start = np.r_[0, np.cumsum(ds * half / sl_rad[0])]
    cum_local = np.r_[0, np.cumsum(ds * half / r_local)]
    est = {"one scale from the whole pipe": cum_global, "one scale from the start section": cum_start,
           "scale re-set locally from the diameter": cum_local}
    print(f"travel while the counter runs (source frames {src[i1]}..{src[i2]}): counter says {gt:.2f} m")
    for label, cum in est.items():
        e = float(cum[i2] - cum[i1])
        print(f"   {label:<40} {e:6.3f} m   error {e - gt:+.3f} m ({(e - gt) / max(gt, 1e-9):+.1%})")
    e_g, e_s, e_l = (float(c[i2] - c[i1]) for c in est.values())
    out.update(radius_units=float(radius), scale_m_per_unit=float(scale), inlier_share=float(keep.mean()),
               wall_rms_mm=float(np.sqrt((res ** 2).mean()) * 1000), est_travel_m=e_g, gt_travel_m=gt,
               travel_error_m=e_g - gt, est_travel_start_scale_m=e_s, est_travel_local_scale_m=e_l,
               slice_radius_units=[float(x) for x in sl_rad], drift_last_over_first=float(sl_rad[-1] / sl_rad[0]),
               cam_offset_mm=float(off_cam.mean() * 1000), straightness=end_to_end / max(path_len, 1e-9),
               lookahead_radii=look)
    json.dump(out, open(out_prefix + ".json", "w"), indent=1)
    with open(out_prefix + "_travel.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["source_frame", "time_s", "counter_m", "est_one_scale_m", "est_local_scale_m", "camera_offset_mm"])
        wr.writerows((int(a), f"{a / 25:.2f}", f"{counter[a]:.2f}", f"{b:.4f}", f"{c:.4f}", f"{o * 1000:.1f}")
                     for a, b, c, o in zip(src, cum_global, cum_local, off_cam))
    gt_curve = cnt - cnt[i1]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    uu, vv = (use - axis_pt) @ u * scale, (use - axis_pt) @ v * scale
    ax[0, 0].scatter(s_pts * scale, vv, s=1, c="0.6")
    ax[0, 0].plot(s_cam * scale, (cams - axis_pt) @ v * scale, "r.-", ms=3, lw=0.8)
    ax[0, 0].set(title="side view along the pipe (red: camera path)", xlabel="along axis [m]", ylabel="[m]")
    ax[0, 0].axis("equal")
    ax[0, 1].scatter(uu[keep], vv[keep], s=1, c="0.5")
    ax[0, 1].scatter(uu[~keep], vv[~keep], s=1, c="orange")
    t = np.linspace(0, 2 * np.pi, 200)
    ax[0, 1].plot(0.15 * np.cos(t), 0.15 * np.sin(t), "b-", lw=1)
    ax[0, 1].plot((cams - axis_pt) @ u * scale, (cams - axis_pt) @ v * scale, "r.", ms=3)
    ax[0, 1].set(title="cross-section (blue: fitted 300 mm circle, orange: rejected points)", xlabel="[m]", ylabel="[m]")
    ax[0, 1].axis("equal")
    ax[1, 0].plot(gt_curve, cum_global - cum_global[i1], "b.-", ms=3, label="one scale for the whole pipe")
    ax[1, 0].plot(gt_curve, cum_local - cum_local[i1], "g.-", ms=3, label="scale re-set locally from the diameter")
    ax[1, 0].plot([0, gt], [0, gt], "k--", lw=0.8, label="ideal")
    ax[1, 0].legend()
    ax[1, 0].set(title="travel: reconstruction vs on-screen counter", xlabel="counter [m] (from 0.25 m)", ylabel="estimated [m]")
    ax[1, 1].plot(sl_pos * scale, sl_rad / sl_rad[0], "g.-")
    ax[1, 1].axhline(1, color="k", ls="--", lw=0.8)
    ax[1, 1].set(title="pipe radius in model units relative to the first slice (scale drift)", xlabel="along axis [m]", ylabel="ratio")
    fig.tight_layout()
    fig.savefig(out_prefix + ".png", dpi=110)
    print("saved", out_prefix + ".json", "and", out_prefix + ".png")


# ------------------------------------------------------------------------------- dense ----------
def read_ply_vertices(path):
    """Vertex table of a binary little-endian PLY file as a numpy structured array."""
    types = {"float": "f4", "float32": "f4", "double": "f8", "float64": "f8", "uchar": "u1", "uint8": "u1",
             "char": "i1", "int8": "i1", "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
             "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4"}
    with open(path, "rb") as f:
        lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("no end_header in " + path)
            lines.append(line.decode("ascii", "replace").strip())
            if lines[-1] == "end_header":
                break
        if not any("binary_little_endian" in ln for ln in lines[:3]):
            raise ValueError("only binary little-endian PLY is supported")
        n, fields, lists, in_vertex = 0, [], [], False
        for ln in lines:
            p = ln.split()
            if p[:1] == ["element"]:
                in_vertex = p[1] == "vertex"
                n = int(p[2]) if in_vertex else n
            elif p[:1] == ["property"] and in_vertex:
                if p[1] == "list":        # e.g. OpenMVS stores the views of every point: count + items
                    lists.append((np.dtype(types[p[2]]).itemsize, np.dtype(types[p[3]]).itemsize))
                elif lists:
                    raise ValueError("scalar property after a list property is not supported")
                else:
                    fields.append((p[2], "<" + types[p[1]]))
        dtype = np.dtype(fields)
        if not lists:
            return np.fromfile(f, dtype=dtype, count=n)
        # variable-length records: walk through them once to find where each one starts
        data = f.read()
        fsz = dtype.itemsize
        offs = np.empty(n, np.int64)
        pos = 0
        for i in range(n):
            offs[i] = pos
            q = pos + fsz
            for csz, isz in lists:
                cnt = data[q] if csz == 1 else int.from_bytes(data[q:q + csz], "little")
                q += csz + cnt * isz
            pos = q
        raw = np.frombuffer(data, np.uint8)
        out = np.empty(n, dtype)
        view = out.view(np.uint8).reshape(n, fsz)
        cols = np.arange(fsz)
        for a in range(0, n, 500000):
            b = min(n, a + 500000)
            view[a:b] = raw[offs[a:b, None] + cols]
        return out


def cmd_dense(converter, sfm_bin, frames_csv, counter_csv, dense_ply, out_prefix):
    """Dense point cloud measured against the pipe found in the sparse reconstruction:
    how much of it lies on the pipe wall, wall scatter, coverage of the circumference, end cap."""
    views, model, intr, poses, spts, nobs, _ = load_sfm(converter, sfm_bin)
    frames = {r["file"]: int(r["source_frame"]) for r in csv.DictReader(open(frames_csv, encoding="utf-8"))}
    reg = sorted((frames[name], vid, poses[pid]) for vid, (name, pid) in views.items() if pid in poses)
    cams = np.array([r[2] for r in reg])
    vtx = read_ply_vertices(dense_ply)
    pts = np.c_[vtx["x"], vtx["y"], vtx["z"]].astype(float)
    has_rgb = all(k in vtx.dtype.names for k in ("red", "green", "blue"))
    rgb = np.c_[vtx["red"], vtx["green"], vtx["blue"]] / 255.0 if has_rgb else np.full((len(pts), 3), 0.5)
    print(f"dense cloud: {len(pts):,} points; cameras registered: {len(reg)} of {len(views)}")

    # reference frame: pipe axis and slice-by-slice centre/radius from the sparse reconstruction
    mean_c = cams.mean(axis=0)
    _, _, vt = np.linalg.svd(cams - mean_c)
    d0 = vt[0] * np.sign((cams[-1] - cams[0]) @ vt[0])
    good = nobs >= 3
    use = spts[good] if good.sum() >= 200 else spts
    d, axis_pt, radius, _, u, v = fit_pipe(use, d0)
    d *= np.sign(d @ d0)
    s_cam = (cams - axis_pt) @ d
    s_sp = (use - axis_pt) @ d
    rows = []
    edges = slice_edges(s_cam.min(), np.percentile(s_sp, 99), radius)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (s_sp >= a) & (s_sp < b)
        if m.sum() < 80:
            continue
        c, r_k, k_in = fit_circle(np.c_[(use[m] - axis_pt) @ u, (use[m] - axis_pt) @ v])
        if k_in.mean() >= 0.4:
            rows.append((0.5 * (a + b), r_k, c[0], c[1]))
    sp = np.array(rows)
    half = PIPE_DIAMETER_M / 2

    def radius_at(x):
        return np.interp(x, sp[:, 0], sp[:, 1])

    s = (pts - axis_pt) @ d
    pu, pv = (pts - axis_pt) @ u, (pts - axis_pt) @ v
    cu, cv = np.interp(s, sp[:, 0], sp[:, 2]), np.interp(s, sp[:, 0], sp[:, 3])
    rr = np.hypot(pu - cu, pv - cv) / radius_at(s)     # distance from the axis relative to the local pipe radius
    theta = np.degrees(np.arctan2(pv - cv, pu - cu))
    grid = np.linspace(min(s.min(), s_cam.min()), max(s.max(), s_cam.max()), 4000)
    metric = np.r_[0, np.cumsum(np.diff(grid) * half / radius_at(0.5 * (grid[1:] + grid[:-1])))]
    m0 = np.interp(s_cam[0], grid, metric)

    def m_of(x):                                        # metres from the first camera, scale re-set locally
        return np.interp(x, grid, metric) - m0

    step = edges[1] - edges[0]
    rng_ax = (s >= sp[0, 0] - step / 2) & (s <= sp[-1, 0] + step / 2)
    on_wall = (rr > 0.85) & (rr < 1.15)
    wall = rng_ax & on_wall
    q = np.percentile(rr[rng_ax], [5, 25, 50, 75, 95])
    print(f"points inside the stretch of pipe covered by the sparse model: {rng_ax.mean():.0%} of the cloud")
    print(f"   their distance from the axis relative to the pipe radius: 5%={q[0]:.2f} 25%={q[1]:.2f} median={q[2]:.2f} "
          f"75%={q[3]:.2f} 95%={q[4]:.2f}   (1.00 = on the wall)")
    print(f"   within +-15% of the wall: {wall.sum() / rng_ax.sum():.0%};  inside the pipe (<0.85): "
          f"{(rng_ax & (rr <= 0.85)).sum() / rng_ax.sum():.0%};  outside (>1.15): {(rng_ax & (rr >= 1.15)).sum() / rng_ax.sum():.0%}")
    print("slice by slice along the pipe:")
    print("   pos[m]   points   on the wall   wall radius dense/sparse   wall scatter[mm]   circumference covered")
    slices = []
    n_rep = 16 if len(edges) == 13 else int(round((sp[-1, 0] - sp[0, 0]) / step)) + 1    # long run: one row per slice
    e2 = np.linspace(sp[0, 0] - step / 2, sp[-1, 0] + step / 2, n_rep + 1)
    for a, b in zip(e2[:-1], e2[1:]):
        m = (s >= a) & (s < b)
        if m.sum() < 500:
            continue
        band = m & (rr > 0.7) & (rr < 1.3)
        med = float(np.median(rr[band])) if band.sum() > 50 else float("nan")
        mad = float(1.4826 * np.median(np.abs(rr[band] - med))) if band.sum() > 50 else float("nan")
        cover = len(np.unique((theta[m & on_wall] // 10).astype(int))) / 36.0
        slices.append(dict(pos_m=float(m_of(0.5 * (a + b))), points=int(m.sum()), wall_share=float((m & on_wall).sum() / m.sum()),
                           radius_ratio=med, scatter_mm=mad * 150, coverage=cover))
        print(f"   {slices[-1]['pos_m']:6.2f} {m.sum():8,d}      {slices[-1]['wall_share']:5.0%}            {med:5.2f}"
              f"                  {mad * 150:6.1f}              {cover:5.0%}")

    # end cap: points well inside the pipe at or beyond the last camera position
    inside = (rr < 0.6) & (s > s_cam.max())
    cam_end_m = float(m_of(s_cam.max()))
    cap_m = float(np.median(m_of(s[inside]))) if inside.sum() > 500 else None
    if cap_m is not None:
        print(f"end cap: {int(inside.sum()):,} points, centred {cap_m - cam_end_m:.2f} m beyond the last camera position")
    out = dict(points=int(len(pts)), in_range_share=float(rng_ax.mean()), wall_share=float(wall.sum() / rng_ax.sum()),
               rr_quantiles=[float(x) for x in q], slices=slices,
               cap_beyond_last_camera_m=None if cap_m is None else cap_m - cam_end_m)
    json.dump(out, open(out_prefix + ".json", "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    k = rng.choice(len(pts), min(len(pts), 700000), replace=False)
    x_m = m_of(s[k])
    fig = plt.figure(figsize=(15, 17))
    gs = fig.add_gridspec(4, 3)
    ax0, ax1, ax2 = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :]), fig.add_subplot(gs[2, :])
    w = wall[k]
    ax0.scatter(x_m[w], theta[k][w], s=0.3, c=rgb[k][w], linewidths=0)
    ax0.set(title="pipe wall unrolled (points within +-15% of the wall radius), point colours",
            ylabel="angle around the pipe [deg]", ylim=(-180, 180))
    near = (rr[k] < 1.6) & rng_ax[k]
    sc = ax1.scatter(x_m[near], theta[k][near], s=0.3, c=np.clip((rr[k][near] - 1) * 100, -40, 40), cmap="coolwarm", linewidths=0)
    ax1.set(title="all points, coloured by distance from the axis relative to the wall radius [%] (blue = inside the pipe)",
            ylabel="angle [deg]", ylim=(-180, 180))
    fig.colorbar(sc, ax=ax1, pad=0.01)
    for a_ in (ax0, ax1):
        a_.set_xlim(m_of(sp[0, 0] - step / 2), m_of(sp[-1, 0] + step / 2))
        a_.set_xlabel("along the pipe from the first camera [m] (scale re-set locally)")
    scale_g = half / np.median(sp[:, 1])
    ax2.scatter((s[k] - s_cam[0]) * scale_g, pv[k] * scale_g, s=0.2, c=rgb[k], linewidths=0)
    ax2.plot((s_cam - s_cam[0]) * scale_g, ((cams - axis_pt) @ v) * scale_g, "r-", lw=1)
    ax2.set(title="side view, one scale for the whole model (red: camera path)", ylabel="[m]", xlabel="along the pipe [m]",
            xlim=(-0.5, (s_cam.max() - s_cam[0]) * scale_g + 1.5), ylim=(-0.6, 0.6))
    ax2.set_aspect("equal")
    ax3 = fig.add_subplot(gs[3, 0])
    ax3.hist(np.clip(rr[rng_ax], 0, 3), bins=150, color="0.4")
    ax3.axvline(1, color="b", lw=1)
    ax3.set(title="distance from the axis / wall radius", xlabel="1 = on the wall", ylabel="points")
    for j, frac in enumerate((0.25, 0.75)):
        axc = fig.add_subplot(gs[3, 1 + j])
        s0 = sp[0, 0] + frac * (sp[-1, 0] - sp[0, 0])
        m = np.where(np.abs(s - s0) < step / 4)[0]
        m = m[:: max(1, len(m) // 40000)]
        r0 = radius_at(s0)
        axc.scatter((pu[m] - np.interp(s0, sp[:, 0], sp[:, 2])) / r0 * 0.15, (pv[m] - np.interp(s0, sp[:, 0], sp[:, 3])) / r0 * 0.15,
                    s=0.5, c=rgb[m], linewidths=0)
        t = np.linspace(0, 2 * np.pi, 200)
        axc.plot(0.15 * np.cos(t), 0.15 * np.sin(t), "b-", lw=0.8)
        axc.set(title=f"cross-section {float(m_of(s0)):.1f} m from the first camera", xlim=(-0.4, 0.4), ylim=(-0.4, 0.4))
        axc.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_prefix + ".png", dpi=105)
    print("saved", out_prefix + ".json", "and", out_prefix + ".png")


if __name__ == "__main__":
    if sys.argv[1] == "dense":
        cmd_dense(*sys.argv[2:8])
    elif sys.argv[1] == "counter-templates":
        cmd_counter_templates(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "counter":
        cmd_counter(sys.argv[2], sys.argv[3], sys.argv[4], nearest="--nearest" in sys.argv[5:])
    elif sys.argv[1] == "sparse":
        cmd_sparse(*sys.argv[2:7])
    else:
        sys.exit(__doc__)
