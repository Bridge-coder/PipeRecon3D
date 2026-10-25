#!/bin/bash
# docs/10: does today's train_gs.py score an old model exactly as before? (old ruler, opacity 0.99)
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"   # REPO, P (pipesplat), PY, GS, TORCH_HOME from scripts/paths.py
export RUN=$GS/g0/concrete_2/run_g0a
$PY $P/train_gs.py $GS/g0/concrete_2 $RUN --anchor --eval-only --alpha-min 0.99 --suffix _regress > /dev/null 2>&1
$PY - <<'PY'
import json
import os
a = json.load(open(os.path.join(os.environ["RUN"], "summary.json")))
b = json.load(open(os.path.join(os.environ["RUN"], "summary_regress.json")))
for part in ("held_outside_blocks", "block"):
    diffs = {k: abs(a[part][k] - b[part][k]) for k in a[part] if isinstance(a[part][k], float)}
    worst = max(diffs, key=diffs.get)
    print(part, f"{len(diffs)} numbers, largest difference {diffs[worst]:.2e} ({worst}: {a[part][worst]:.4f} vs {b[part][worst]:.4f})")
PY
