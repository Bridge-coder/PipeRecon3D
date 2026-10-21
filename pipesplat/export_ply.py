"""A trained Gaussian model (train_gs.py model.pt) as a standard 3D Gaussian Splatting .ply (the INRIA layout:
x y z, nx ny nz, f_dc_0..2, f_rest_0..44, opacity (logit), scale_0..2 (log), rot_0..3 (wxyz)), readable by common
viewers (e.g. SuperSplat, the INRIA SIBR viewer). Positions in metres in the frame of the scene (vision: pipe frame
of render_views.load_model; InPipe vision: SfM frame x AprilTag scale; InPipe LiDAR: fused world frame).
The lamp factor, ambient colour and per-frame exposure of --lamp models are not part of a plain .ply: such a model
looks like its own colours under even light (docs/10 section 5).

usage: /opt/gsenv/bin/python pipesplat/export_ply.py RUN_DIR [OUT.ply]   (runs anywhere torch is installed)
"""
import os
import sys

import numpy as np
import torch

run = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(run, "gaussians.ply")
m = torch.load(os.path.join(run, "model.pt"), map_location="cpu")
xyz = m["means"].numpy().astype(np.float32)
n = len(xyz)
f_dc = m["sh0"].reshape(n, 3).numpy().astype(np.float32)
f_rest = m["shN"].transpose(1, 2).reshape(n, -1).numpy().astype(np.float32)   # [N, 3 x 15], channel-major as INRIA
op = m["opacities"].reshape(n, 1).numpy().astype(np.float32)
sc = m["scales"].numpy().astype(np.float32)
rot = m["quats"].numpy().astype(np.float32)
rot /= np.linalg.norm(rot, axis=1, keepdims=True)
cols = [("x", xyz[:, 0]), ("y", xyz[:, 1]), ("z", xyz[:, 2])] + [(k, np.zeros(n, np.float32)) for k in ("nx", "ny", "nz")]
cols += [(f"f_dc_{i}", f_dc[:, i]) for i in range(3)] + [(f"f_rest_{i}", f_rest[:, i]) for i in range(f_rest.shape[1])]
cols += [("opacity", op[:, 0])] + [(f"scale_{i}", sc[:, i]) for i in range(3)] + [(f"rot_{i}", rot[:, i]) for i in range(4)]
data = np.stack([c for _, c in cols], 1).astype("<f4")
with open(out, "wb") as fh:
    fh.write(("ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n"
              + "".join(f"property float {k}\n" for k, _ in cols) + "end_header\n").encode())
    fh.write(data.tobytes())
print(f"{n:,} Gaussians -> {out} ({os.path.getsize(out) / 1e6:.0f} MB)")
