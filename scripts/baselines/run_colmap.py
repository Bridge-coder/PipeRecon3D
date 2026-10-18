"""Baseline: COLMAP (SfM + PatchMatch MVS) on the frames our pipeline extracted, for the comparison table.

usage: python -I run_colmap.py COLMAP_EXE IMAGES_DIR OUT_DIR [--fixed F CX CY K1] [--overlap N] [--dense]
       IMAGES_DIR  WORK/images of a run_pipeline.py run (same frames, same contrast enhancement as ours)
       --fixed     camera held fixed at these values (OpenMVG pixel convention, converted to COLMAP's +0.5);
                   without it COLMAP's defaults: SIMPLE_RADIAL, focal from its prior, refined in bundle adjustment
       --overlap   sequential matching window (default: COLMAP's default 10; ours matches 25 frames ahead)
       --dense     image_undistorter + patch_match_stereo (CUDA) + stereo_fusion on the largest model
       --dense-only  only that, on the sparse model already in OUT_DIR (e.g. after an interrupted run)

Everything else is COLMAP's default, as a user would run it on a video. Writes OUT_DIR/database.db,
sparse/<k>/ (binary) and sparse_txt/<k>/ (text, read by colmap_eval.py), dense/fused.ply and timing.json.
"""
import json
import os
import subprocess
import sys
import time

args = sys.argv[1:]
fixed = None
if "--fixed" in args:
    i = args.index("--fixed")
    fixed = [float(x) for x in args[i + 1:i + 5]]
    del args[i:i + 5]
overlap = None
if "--overlap" in args:
    i = args.index("--overlap")
    overlap = args[i + 1]
    del args[i:i + 2]
dense_only = "--dense-only" in args
if dense_only:
    args.remove("--dense-only")
dense = "--dense" in args or dense_only
if dense:
    args.remove("--dense") if "--dense" in args else None
COLMAP, IMAGES, OUT = args
os.makedirs(OUT, exist_ok=True)
db = os.path.join(OUT, "database.db")
timing = {}


def run(name, cmd):
    print(f"[{name}] " + subprocess.list2cmdline(cmd), flush=True)
    t0 = time.time()
    with open(os.path.join(OUT, f"{name}.log"), "w", encoding="utf-8", errors="replace") as log:
        code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    timing[name] = round(time.time() - t0, 1)
    print(f"   exit {code}, {timing[name]} s", flush=True)
    json.dump(timing, open(os.path.join(OUT, "timing.json"), "w"), indent=1)
    if code:
        sys.exit(f"{name} failed, see {OUT}/{name}.log")


if os.path.exists(os.path.join(OUT, "timing.json")):
    timing = json.load(open(os.path.join(OUT, "timing.json")))


sparse = os.path.join(OUT, "sparse")


def sparse_run():
    if os.path.exists(db):
        os.remove(db)
    fe = [COLMAP, "feature_extractor", "--database_path", db, "--image_path", IMAGES,
          "--ImageReader.single_camera", "1", "--ImageReader.camera_model", "SIMPLE_RADIAL"]
    if fixed:
        f, cx, cy, k1 = fixed
        fe += ["--ImageReader.camera_params", f"{f},{cx + 0.5},{cy + 0.5},{k1}"]
    run("feature_extractor", fe)
    run("sequential_matcher", [COLMAP, "sequential_matcher", "--database_path", db]
        + (["--SequentialMatching.overlap", overlap] if overlap else []))
    os.makedirs(sparse, exist_ok=True)
    mp = [COLMAP, "mapper", "--database_path", db, "--image_path", IMAGES, "--output_path", sparse]
    if fixed:
        mp += ["--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_extra_params", "0",
               "--Mapper.ba_refine_principal_point", "0"]
    run("mapper", mp)
    models = sorted(d for d in os.listdir(sparse) if d.isdigit())
    sizes = {}
    for m in models:
        txt = os.path.join(OUT, "sparse_txt", m)
        os.makedirs(txt, exist_ok=True)
        run(f"model_converter_{m}", [COLMAP, "model_converter", "--input_path", os.path.join(sparse, m),
                                      "--output_path", txt, "--output_type", "TXT"])
        with open(os.path.join(txt, "images.txt"), encoding="utf-8") as fh:
            sizes[m] = sum(1 for ln in fh if ln.strip() and not ln.startswith("#")) // 2
    print("models (registered images):", sizes)
    timing["models"] = sizes
    json.dump(timing, open(os.path.join(OUT, "timing.json"), "w"), indent=1)
    return sizes


if dense_only:
    sizes = timing.get("models", {})
else:
    sizes = sparse_run()
if dense and sizes:
    best = max(sizes, key=sizes.get)
    dd = os.path.join(OUT, "dense")
    run("image_undistorter", [COLMAP, "image_undistorter", "--image_path", IMAGES,
                              "--input_path", os.path.join(sparse, best), "--output_path", dd])
    run("patch_match_stereo", [COLMAP, "patch_match_stereo", "--workspace_path", dd])
    run("stereo_fusion", [COLMAP, "stereo_fusion", "--workspace_path", dd, "--output_path", os.path.join(dd, "fused.ply")])
    timing["dense_model"] = best
    json.dump(timing, open(os.path.join(OUT, "timing.json"), "w"), indent=1)
