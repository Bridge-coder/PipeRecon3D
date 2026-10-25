# docs/10: independent check on the lab concrete pipe A (not used for any tuning): export scene + surface (Windows)
# Folders from scripts/paths.py (WORK_NEW = the v1.0 work folders, GS_ROOT); -Video: the raw video (docs/05)
param([string]$Video = "")
Set-Location (Split-Path (Split-Path $PSScriptRoot))          # the repository
$new = python -I scripts/paths.py WORK_NEW
$gs = python -I scripts/paths.py GS_ROOT
if (-not $Video) { $Video = "$new/mix15_3lm/data/300mm_mix_15meters_ON_3lm_3u_CLOSED_1_stop_at_14meters.mp4" }
python -I pipesplat/export_g0.py vision "$new/concrete_lab" "$Video" "$gs/g0/concrete_lab" --lamp lampmap/results/concrete_lab/lamp_map.npy
python -I pipesplat/export_surface.py vision "$new/concrete_lab" "$gs/g0/concrete_lab"
