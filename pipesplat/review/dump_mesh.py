"""docs/10 X8: the OpenMVS mesh of a vision work dir in the pipe frame of scene.npz (vertices + the textured faces,
exactly the faces export_tex_obj.py writes), for baking a texture on it in WSL (pipesplat/bake_gs_tex.py).
--with-blank (X8b): also the faces OpenMVS left without an image (render_views' "blank" faces, 1.4 % on concrete_2):
left out, they are the small holes of the OBJ (black specks from inside, white from outside); the bake gives them
colours from the Gaussian model's renders or fills them from their neighbours.
--blank-max-x X (X8b, concrete_2: 4.26): only those whose centre lies at x < X - beyond, in the end cap / socket mesh
of concrete_2, they made the renders of the frames facing the cap worse (docs/10 X8b).

usage: python -I pipesplat/review/dump_mesh.py WORK_DIR OUT.npz [--with-blank [--blank-max-x X]]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SCRIPTS = os.path.join(ROOT, "scripts")
for p in (os.path.join(SCRIPTS, "vision"), SCRIPTS):
    sys.path.insert(0, p)
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import paths  # noqa: E402
import render_views as rv  # noqa: E402

args = list(sys.argv[1:])
with_blank = "--with-blank" in args
if with_blank:
    args.remove("--with-blank")
max_x = float("inf")
if "--blank-max-x" in args:
    i = args.index("--blank-max-x")
    max_x = float(args[i + 1])
    del args[i:i + 2]
m = rv.load_model(paths.CONVERTER, args[0], [None], straighten=False)
blank = m["blank"]
add = blank & (m["xyz"][m["tri"]].mean(1)[:, 0] < max_x) if with_blank else np.zeros(len(blank), bool)
tri = m["tri"][~blank | add]
np.savez(args[1], xyz=m["xyz"].astype(np.float64), tri=tri.astype(np.int64))
print(f"{len(m['xyz']):,} vertices, {len(tri):,} faces ({int(add.sum()):,} of the {int(blank.sum()):,} faces without an "
      f"image included) -> {args[1]}")
