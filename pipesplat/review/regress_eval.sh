#!/bin/bash
# docs/10: does today's train_gs.py score an old model exactly as before? (old ruler, opacity 0.99)
export TORCH_HOME=/mnt/d/pipe3d_gs/torch_cache
/opt/gsenv/bin/python /mnt/c/Users/PC/Desktop/PipeRecon3D/pipesplat/train_gs.py /mnt/d/pipe3d_gs/g0/concrete_2 /mnt/d/pipe3d_gs/g0/concrete_2/run_g0a --anchor --eval-only --alpha-min 0.99 --suffix _regress > /dev/null 2>&1
/opt/gsenv/bin/python - <<'PY'
import json
a = json.load(open("/mnt/d/pipe3d_gs/g0/concrete_2/run_g0a/summary.json"))
b = json.load(open("/mnt/d/pipe3d_gs/g0/concrete_2/run_g0a/summary_regress.json"))
for part in ("held_outside_blocks", "block"):
    diffs = {k: abs(a[part][k] - b[part][k]) for k in a[part] if isinstance(a[part][k], float)}
    worst = max(diffs, key=diffs.get)
    print(part, f"{len(diffs)} numbers, largest difference {diffs[worst]:.2e} ({worst}: {a[part][worst]:.4f} vs {b[part][worst]:.4f})")
PY
