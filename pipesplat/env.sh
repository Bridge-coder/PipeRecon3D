# Sourced by the WSL shell scripts of pipesplat (review/*.sh, setup/*.sh): where things are, from scripts/paths.py
# (config/paths.json or PIPE3D_* variables; a Windows drive path D:/x becomes /mnt/d/x), so no script names a drive.
#   REPO  the repository        P   its pipesplat/ folder        PY  the training Python (PIPE3D_PY, default /opt/gsenv)
#   GS    GS_ROOT: g0/<data>/ scenes and runs, review/, logs/   TORCH_HOME  TORCH_CACHE unless already set
P=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(dirname "$P")
PY=${PIPE3D_PY:-/opt/gsenv/bin/python}
GS=$(python3 "$REPO/scripts/paths.py" GS_ROOT)
export TORCH_HOME=${TORCH_HOME:-$(python3 "$REPO/scripts/paths.py" TORCH_CACHE)}
