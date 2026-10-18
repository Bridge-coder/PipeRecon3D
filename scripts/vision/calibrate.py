"""Scale-drift calibration: re-run only the SfM step of an existing work folder with a chosen focal length
and radial distortion, and measure how constant the pipe radius comes out.

Usage:
  calibrate.py run PROJECT_ROOT SRC_WORK CALIB_DIR NAME --focal F --k1 K --pair A B [--refine X]
      SRC_WORK   work folder of a finished run (images, sfm/matches with features and matches, frames.csv,
                 counter.csv); its features and matches are hard-linked, nothing in it is changed
      --k1       radial distortion of OpenMVG's "pinhole radial 1" model: x_d = x_u * (1 + k1 * r_u^2) in
                 normalised image coordinates (negative = barrel distortion)
      --pair     file names of the initial image pair
      --refine   passed to IncrementalSfM as -f (default NONE: intrinsics held fixed)
  calibrate.py table CALIB_DIR [OUT_CSV]
      One line per variant found in CALIB_DIR; also saved as CSV when OUT_CSV is given.
"""
import glob
import json
import os
import re
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
args = sys.argv[1:]


def opt(flag, default=None, n=1):
    if flag not in args:
        return default
    i = args.index(flag)
    return args[i + 1] if n == 1 else args[i + 1:i + 1 + n]


def table(calib):
    rows = [json.load(open(p)) for p in glob.glob(os.path.join(calib, "*", "result.json"))]
    rows.sort(key=lambda r: (r["focal_in"], -r["k1_in"], r["refine"]))
    print(f"{'variant':<22}{'focal':>7}{'k1':>8}{'refine':>18}{'k1 out':>9}{'frames':>9}{'points':>8}{'resid':>7}"
          f"{'radius last/first':>19}{'one scale':>11}{'local scale':>13}")
    for r in rows:
        fr = f"{r['registered']}/{r['frames']}"
        if r.get("drift") is None:
            print(f"{r['name']:<22}{r['focal_in']:>7g}{r['k1_in']:>8g}{r['refine']:>18}{'':>9}{fr:>9}   (too little reconstructed)")
            continue
        print(f"{r['name']:<22}{r['focal_in']:>7g}{r['k1_in']:>8g}{r['refine']:>18}{r['k1_out']:>9.4f}{fr:>9}{r['points']:>8}"
              f"{r['residual_px']:>7.2f}{r['drift']:>19.3f}{r['err_one_scale']:>+11.1%}{r['err_local_scale']:>+13.1%}")
    return rows


if args[0] == "table":
    rows = table(os.path.abspath(args[1]))
    if len(args) > 2:
        import csv
        cols = ["name", "focal_in", "k1_in", "refine", "k1_out", "registered", "frames", "points", "residual_px", "drift",
                "gt_travel_m", "est_one_scale_m", "err_one_scale", "est_local_scale_m", "err_local_scale", "wall_rms_mm"]
        with open(args[2], "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            wr.writeheader()
            wr.writerows(rows)
        print("saved", args[2])
    sys.exit()

root, src, calib = (os.path.abspath(a) for a in args[1:4])
name = args[4]
focal, k1, refine = float(opt("--focal")), float(opt("--k1")), opt("--refine", "NONE")
pair = opt("--pair", n=2)
MVG = os.path.join(root, "tools", "openMVG_v1.6", "ReleaseV1.6.Halibut.WindowsBinaries_VS2017")
SCRIPTS = os.path.join(root, "scripts")
var = os.path.join(calib, name)
matches, recon, logs = (os.path.join(var, d) for d in ("matches", "recon", "logs"))
old = os.path.join(src, "sfm", "matches")
sfm_json = os.path.join(matches, "sfm_data.json")
for d in (matches, recon, logs):
    os.makedirs(d, exist_ok=True)


def mvg(n):
    return os.path.join(MVG, f"openMVG_main_{n}.exe")


def run(tag, cmd):
    p = subprocess.run(cmd, cwd=var, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    text = p.stdout.decode("utf-8", errors="replace")
    with open(os.path.join(logs, tag + ".log"), "w", encoding="utf-8") as f:
        f.write("COMMAND: " + subprocess.list2cmdline(cmd) + "\n\n" + text + f"\nEXIT CODE: {p.returncode}\n")
    return p.returncode, text


# same images, features and matches as the source run; only the camera model differs
code, _ = run("ImageListing", [mvg("SfMInit_ImageListing"), "-i", os.path.join(src, "images"), "-o", matches, "-c", "2",
                               "-k", f"{focal:g};0;383;0;{focal:g};288;0;0;1"])
if code != 0:
    sys.exit(f"{name}: ImageListing failed")
sfm = json.load(open(sfm_json, encoding="utf-8"))
intr = sfm["intrinsics"][0]["value"]["ptr_wrapper"]["data"]
if "disto_k1" not in intr:
    sys.exit(f"{name}: no disto_k1 in the listed camera model: {sorted(intr)}")
intr["disto_k1"] = [k1]
json.dump(sfm, open(sfm_json, "w", encoding="utf-8"), indent=1)
for f in glob.glob(os.path.join(old, "*.feat")) + glob.glob(os.path.join(old, "*.desc")) + [
        os.path.join(old, n) for n in ("image_describer.json", "matches.putative.bin", "matches.f.bin")]:
    dst = os.path.join(matches, os.path.basename(f))
    if not os.path.exists(dst):
        os.link(f, dst)

code, text = run("IncrementalSfM", [mvg("IncrementalSfM"), "-i", sfm_json, "-m", matches, "-o", recon, "-f", refine,
                                    "-a", pair[0], "-b", pair[1]])
cal = re.search(r"#Camera calibrated:\s*(\d+)\s*from\s*(\d+)", text)
pts = re.search(r"#Tracks, #3D points:\s*(\d+)", text)
res = re.findall(r"Residual mean:\s*([0-9.eE+-]+)", text)
rec = dict(name=name, focal_in=focal, k1_in=k1, refine=refine, pair=pair, sfm_exit=code,
           registered=int(cal.group(1)) if cal else 0, frames=int(cal.group(2)) if cal else len(sfm["views"]),
           points=int(pts.group(1)) if pts else 0, residual_px=float(res[-1]) if res else None, drift=None)

sfm_bin = os.path.join(recon, "sfm_data.bin")
if os.path.exists(sfm_bin) and rec["registered"] >= 5:
    prefix = os.path.join(var, "sparse_eval")
    p = subprocess.run([sys.executable, "-I", os.path.join(SCRIPTS, "evaluate.py"), "sparse", mvg("ConvertSfM_DataFormat"),
                        sfm_bin, os.path.join(src, "frames.csv"), os.path.join(src, "counter.csv"), prefix],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    open(os.path.join(logs, "evaluate.log"), "w", encoding="utf-8").write(p.stdout.decode("utf-8", errors="replace"))
    if os.path.exists(prefix + ".json"):
        ev = json.load(open(prefix + ".json"))
        out = json.load(open(os.path.join(recon, "sfm_data_full.json"), encoding="utf-8"))
        intr_out = out["intrinsics"][0]["value"]["ptr_wrapper"]["data"]
        if "drift_last_over_first" in ev:
            gt = ev["gt_travel_m"]
            rec.update(drift=ev["drift_last_over_first"], slice_radius_units=ev["slice_radius_units"], gt_travel_m=gt,
                       est_one_scale_m=ev["est_travel_m"], est_local_scale_m=ev["est_travel_local_scale_m"],
                       err_one_scale=ev["est_travel_m"] / gt - 1, err_local_scale=ev["est_travel_local_scale_m"] / gt - 1,
                       longest_run=ev["longest_run"], wall_rms_mm=ev["wall_rms_mm"],
                       focal_out=intr_out["focal_length"], k1_out=intr_out["disto_k1"][0])
json.dump(rec, open(os.path.join(var, "result.json"), "w"), indent=1)
print(f"{name}: registered {rec['registered']}/{rec['frames']}, points {rec['points']}, radius last/first "
      + (f"{rec['drift']:.3f}, travel {rec['err_one_scale']:+.1%} (one scale) / {rec['err_local_scale']:+.1%} (local scale)"
         if rec["drift"] is not None else "n/a"))
