"""Where the external programs, the data and the large intermediate files live.

Every script takes its locations from here, so the repository can be moved to another machine without editing
code. Three layers, later ones win:
  1. defaults below (everything inside the repository: third_party/, data/, work/, results/)
  2. config/paths.json (copy config/paths.example.json and edit; git-ignored, so it stays on this machine)
  3. environment variables PIPE3D_<KEY>, e.g. PIPE3D_RECON_BIG=E:/scratch/recon (used for test re-runs)
Relative paths in paths.json are taken relative to the repository root. The same file serves the WSL side
(pipesplat runs in the pipe3d distro): there a Windows drive path such as D:/pipe3d_gs becomes /mnt/d/pipe3d_gs.

Keys
  MVG_BIN         folder with the OpenMVG 1.6 Windows binaries (openMVG_main_*.exe)
  MVS_BIN         folder with the OpenMVS 2.4.0 Windows binaries (DensifyPointCloud.exe, ...)
  LOGS            folder for the step logs of run_pipeline.py / run_vision.py (one subfolder per work folder)
  INPIPE_EXTRACT  InPipe-SLAM sequence extracted by lidar/extract_bag.py (bag_info.json, lidar/, images/, ...)
  INPIPE_VISION   work folder of the vision run on that sequence (vision/run_vision.py)
  INPIPE_RESULTS  small results and figures of that sequence (fusion/, recon/, lidar/, tag/, vision/, viz/)
  FUSED_BIG       large outputs of fusion/f5_fuse.py (coloured LiDAR wall point cloud)
  RECON_BIG       large outputs of recon/r1..r5 (wall points, height field, textures, model/, viz/)
  DIAG_BIG        large outputs of analysis/wall_blur_*.py
  WORK_SHEFFIELD  work folders of the Sheffield runs (concrete_1 ... concrete_5, mix15_2; v1.0 Release zips)
  WORK_NEW        work folders of the new-video tests (concrete_lab, sewer_pipe1/work; v1.0 Release zips)
  GS_ROOT         the Gaussian line (pipesplat, docs/06-11): g0/<data>/ scenes and runs, review/, logs/, queue.pause
  TORCH_CACHE     TORCH_HOME for pipesplat (LPIPS / AlexNet weights)

usage: python scripts/paths.py        every key, and whether it exists
       python scripts/paths.py KEY    that one value only (for shell scripts: pipesplat/env.sh)
"""
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_DEFAULTS = {
    "MVG_BIN": "third_party/openMVG_v1.6/ReleaseV1.6.Halibut.WindowsBinaries_VS2017",
    "MVS_BIN": "third_party/openMVS_v2.4.0/vc17/x64/Release",
    "LOGS": "logs",
    "INPIPE_EXTRACT": "work/inpipe_172011/extract",
    "INPIPE_VISION": "work/inpipe_172011/vision",
    "INPIPE_RESULTS": "results/inpipe_172011",
    "FUSED_BIG": "work/inpipe_172011/fused",
    "RECON_BIG": "work/inpipe_172011/recon",
    "DIAG_BIG": "work/inpipe_172011/diag",
    "WORK_SHEFFIELD": "work/sheffield",
    "WORK_NEW": "work/new_videos",
    "GS_ROOT": "work/pipe3d_gs",
    "TORCH_CACHE": "work/pipe3d_gs/torch_cache",
}


def _native(v):
    """a Windows drive path as seen from WSL / Linux (D:/x -> /mnt/d/x); unchanged on Windows"""
    m = re.match(r"^([A-Za-z]):[/\\](.*)$", v)
    if os.name == "nt" or not m:
        return v
    return f"/mnt/{m.group(1).lower()}/" + m.group(2).replace("\\", "/")


def _load():
    cfg = dict(_DEFAULTS)
    path = os.path.join(REPO, "config", "paths.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cfg.update({k: v for k, v in json.load(f).items() if not k.startswith("_")})
    for k in list(cfg):
        cfg[k] = _native(os.environ.get("PIPE3D_" + k, cfg[k]))
    return {k: os.path.normpath(v if os.path.isabs(v) else os.path.join(REPO, v)) for k, v in cfg.items()}


P = _load()
MVG_BIN, MVS_BIN, LOGS = P["MVG_BIN"], P["MVS_BIN"], P["LOGS"]
INPIPE_EXTRACT, INPIPE_VISION, INPIPE_RESULTS = P["INPIPE_EXTRACT"], P["INPIPE_VISION"], P["INPIPE_RESULTS"]
FUSED_BIG, RECON_BIG, DIAG_BIG = P["FUSED_BIG"], P["RECON_BIG"], P["DIAG_BIG"]
WORK_SHEFFIELD, WORK_NEW, GS_ROOT, TORCH_CACHE = P["WORK_SHEFFIELD"], P["WORK_NEW"], P["GS_ROOT"], P["TORCH_CACHE"]


def mvg(name):
    """path of an OpenMVG program, e.g. mvg("IncrementalSfM")"""
    return os.path.join(MVG_BIN, f"openMVG_main_{name}.exe")


def mvs(name):
    """path of an OpenMVS program, e.g. mvs("DensifyPointCloud")"""
    return os.path.join(MVS_BIN, f"{name}.exe")


CONVERTER = mvg("ConvertSfM_DataFormat")   # sfm_data.bin -> json, used by every evaluation script


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(P[sys.argv[1]])
        sys.exit(0)
    for k, v in P.items():
        print(f"{k:15s} {v}   {'ok' if os.path.exists(v) else '(missing)'}")
    print(f"{'CONVERTER':15s} {CONVERTER}   {'ok' if os.path.exists(CONVERTER) else '(missing)'}")
