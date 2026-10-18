"""LiDAR + vision fusion and the final pipe model for an InPipe-SLAM sequence, stage by stage.

usage: python -I scripts/run_inpipe.py [STAGE ...] [--from STAGE] [--to STAGE] [--zip BAG.zip] [--list]
       no stage names: every stage from --from (default the first) to --to (default the last)

Folders come from scripts/paths.py (config/paths.json): INPIPE_EXTRACT, INPIPE_VISION, INPIPE_RESULTS, FUSED_BIG,
RECON_BIG. Every stage writes a log to LOGS/inpipe/<stage>.log and stops the run if a step fails.

The settings are the ones used for sequence 172011 (docs/history/). Two inputs are calibration results rather
than outputs of a stage and must already be in INPIPE_RESULTS/fusion before f5 / r1:
  extrinsic_translation_final.json  t_cl combined from f3b (geometric), f3c (wall colour, lateral), f3g (board,
                                    axial) and a board-pattern check; the choice is recorded inside the file
                                    (02_融合结果.md, "F3 补充"). For another robot: run f3b/f3c/f3g, then write it.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths  # noqa: E402

PY = [sys.executable, "-I", "-X", "utf8"]          # utf8: logs are files, not a console
EX, VW, RES = paths.INPIPE_EXTRACT, paths.INPIPE_VISION, paths.INPIPE_RESULTS


def s(rel, *a):
    return PY + [os.path.join(HERE, rel)] + [str(x) for x in a]


def stages(zip_path):
    """(name, what it does, [commands]) in run order"""
    return [
        ("extract", "ROS 2 bag (MCAP, read from the zip) -> csv / npy / png", [s("lidar/extract_bag.py", zip_path, EX)]),
        ("lidar", "LiDAR single-modal baseline: per-scan cylinder fit (0.15-1.0 m) and the LiDAR + encoder map",
         [s("lidar/lidar_scan_fit.py", EX), s("lidar/lidar_map.py", EX, os.path.join(RES, "lidar"), "--roll", "slot")]),
        ("lidar_short", "per-scan cylinder fit in a short window (0.15-0.45 m, one straight leg) -> lidar_fit_short.csv",
         [s("lidar/lidar_scan_fit.py", EX, "--lo", "0.15", "--hi", "0.45", "--out", "lidar_fit_short.csv")]),
        ("tags", "AprilTag references: rear camera tag 80, front camera tag 91",
         [s("lidar/tag_track.py", EX, os.path.join(RES, "tag"), "--camera", "usbcam"),
          s("lidar/tag_track.py", EX, os.path.join(RES, "tag"), "--camera", "color")]),
        ("vision", "vision single-modal: OpenMVG + OpenMVS on the front colour camera (every 10th frame from 500)",
         [s("vision/run_vision.py", EX, VW, "--every", "10", "--first", "500", "--clahe",
            1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13)]),
        ("vision_eval", "vision-only travel (200 mm nominal bore) and dense-wall check",
         [s("vision/eval_sfm.py", EX, VW, os.path.join(RES, "vision")),
          s("vision/eval_dense.py", VW, os.path.join(RES, "vision", "dense_wall_check.csv"))]),
        ("f1", "F1 common clock for all streams -> fusion/time_offsets.json", [s("fusion/f1_time.py", EX, RES)]),
        ("legs", "two straight legs + bend of the vision model -> fusion/vision_legs.json",
         [s("fusion/vision_model.py", EX, VW, RES)]),
        ("f2", "F2 rotation LiDAR -> camera -> fusion/extrinsic_rotation.json", [s("fusion/f2_rotation.py", EX, VW, RES)]),
        ("f3", "F3 translation candidates: geometric (f3b), wall colour (f3c), AprilTag board (f3g)",
         [s("fusion/f3b_translation.py", EX, VW, RES), s("fusion/f3c_photometric.py", EX, VW, RES),
          s("fusion/f3g_board.py", EX, RES)]),
        ("f3_icp", "baseline: point-to-plane ICP against the dense vision cloud (does not converge in a pipe; "
                   "kept for the comparison table of f5_eval)",
         [s("fusion/f3_register.py", EX, VW, RES, "--leg", "2", "--range", "0.2", "0.6", "--scale", "tag",
            "--tag", "leg2")]),
        ("f4", "F4 scale: AprilTag triangulated in the vision model; LiDAR range-bias test",
         [s("fusion/f4_tag_scale.py", EX, VW, RES), s("fusion/f4_lidar_range.py", EX, RES)]),
        ("f5", "F5 fused trajectory, LiDAR wall coloured by the camera, evaluation table",
         [s("fusion/f5_fuse.py", EX, VW, RES), s("fusion/f5_eval.py", EX, VW, RES), s("fusion/f5_views.py", RES)]),
        ("r1", "R1 LiDAR wall points of every scan in the fused frame", [s("recon/r1_wall_points.py")]),
        ("r2", "R2 centerline + LiDAR range-bias self-calibration; R2b scan-to-map pose refinement",
         [s("recon/r2_centerline_calib.py"), s("recon/r2b_scan_refine.py")]),
        ("r3", "R3 height field r(u, theta) and its mesh", [s("recon/r3_heightfield.py")]),
        ("r4", "R4 per-texel texture from the colour frames (refined poses)", [s("recon/r4_texture.py", "--poses", "refined")]),
        ("r5", "R5 OBJ models, views, held-out check; close-ups",
         [s("recon/r5_compare.py"), s("recon/r5b_detail.py")]),
        ("r5_video", "R5 six-panel comparison video (compare_new.mp4)", [s("recon/r5_compare.py", "--video")]),
    ]


def main():
    args = sys.argv[1:]

    def opt(flag):
        if flag in args:
            i = args.index(flag)
            v = args[i + 1]
            del args[i:i + 2]
            return v
        return None

    zip_path, first, last = opt("--zip"), opt("--from"), opt("--to")
    table = stages(zip_path or "BAG.zip")
    names = [t[0] for t in table]
    if "--list" in args:
        for n, what, cmds in table:
            print(f"{n:12s} {what}")
        return
    for n in args + [x for x in (first, last) if x]:
        if n not in names:
            sys.exit(f"unknown stage {n}; stages: {' '.join(names)}")
    if args:
        todo = [t for t in table if t[0] in args]
    else:
        a = names.index(first) if first else 0
        b = names.index(last) if last else len(names) - 1
        todo = table[a:b + 1]
    if any(t[0] == "extract" for t in todo) and not zip_path:
        sys.exit("stage extract needs --zip BAG.zip (lidar/download_parallel.py fetches it from Zenodo)")
    logs = os.path.join(paths.LOGS, "inpipe")
    os.makedirs(logs, exist_ok=True)
    print(f"EXTRACT {EX}\nVISION  {VW}\nRESULTS {RES}\nFUSED   {paths.FUSED_BIG}\nRECON   {paths.RECON_BIG}\n")
    for n, what, cmds in todo:
        t0 = time.time()
        print(f"[{n}] {what}", flush=True)
        with open(os.path.join(logs, f"{n}.log"), "w", encoding="utf-8", errors="replace") as log:
            for cmd in cmds:
                log.write("COMMAND: " + subprocess.list2cmdline(cmd) + "\n")
                log.flush()
                code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                if code:
                    sys.exit(f"[{n}] failed (exit {code}): {subprocess.list2cmdline(cmd)}\n   see {log.name}")
        print(f"   done in {time.time() - t0:.0f} s  (log {os.path.join(logs, n + '.log')})", flush=True)


if __name__ == "__main__":
    main()
