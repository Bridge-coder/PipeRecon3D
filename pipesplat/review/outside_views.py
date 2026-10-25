"""docs/10: outside / cut-away / side views of the old model (as in the v1.0 model_views figures), and the cameras
for rendering the new Gaussian model the same way (outside_views_gs.py, WSL). Runs on Windows.

The cameras are render_views.py's: outside (azimuth 36, elevation 13, fov 30), cut-away (55, 32, near half removed),
side elevations (orthographic, 640 px/m, rows of up to 5.2 m), all in the pipe frame (x along the pipe, z up).
  concrete_2 / concrete_lab: old = OpenMVS textured mesh (render_views.load_model, straighten=False) - the frame of the
                             Gaussian scene, so the pipe frame is the identity
  inpipe:                    old = R3 LiDAR height field + R4 texture (lampmap_inpipe.load_mesh) in the fused frame,
                             which has the vision axes and the AprilTag scale - the frame of the inpipe_vision Gaussian
                             scene; the pipe frame is built from the camera path (x) and the cameras' mean up (z)

usage: python -I pipesplat/review/outside_views.py DATA OUT_DIR     DATA = concrete_2 | concrete_lab | sewer_pipe1 | inpipe
writes OUT_DIR/old_<view>.png and OUT_DIR/cams.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (os.path.join(ROOT, "scripts", "vision"), os.path.join(ROOT, "scripts", "recon"), os.path.join(ROOT, "scripts"),
          os.path.join(ROOT, "lampmap")):
    sys.path.insert(0, p)
import paths  # noqa: E402
import render_views as rv  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
WORK = {"concrete_2": os.path.join(paths.WORK_SHEFFIELD, "concrete_2"), "concrete_lab": os.path.join(paths.WORK_NEW, "concrete_lab"),
        "sewer_pipe1": os.path.join(paths.WORK_NEW, "sewer_pipe1", "work")}     # Pipe1: run with PIPE3D_PIPE_DIAMETER_M=0.435 (docs/11)


def vertex_normals(xyz, tri):
    n = np.cross(xyz[tri[:, 1]] - xyz[tri[:, 0]], xyz[tri[:, 2]] - xyz[tri[:, 0]])
    vn = np.zeros_like(xyz)
    for k in range(3):
        np.add.at(vn, tri[:, k], n)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def main():
    data, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    if data in WORK:
        model = rv.load_model(paths.CONVERTER, WORK[data], [None], straighten=False)
        o, Rp = np.zeros(3), np.eye(3)
    else:
        import lampmap_inpipe as li
        m = li.load_mesh()
        sc = np.load(os.path.join(paths.GS_ROOT, "g0", "inpipe_vision", "scene.npz"))
        eyes, vm = sc["train_eyes"].astype(np.float64), sc["train_viewmats"].astype(np.float64)
        x = eyes[-1] - eyes[0]
        x /= np.linalg.norm(x)
        up = -vm[:, 1, :3].mean(0)                      # image up of the cameras, in the world
        z = up - (up @ x) * x
        z /= np.linalg.norm(z)
        y = np.cross(z, x)
        Rp = np.stack([x, y, z])                         # world -> pipe
        o = eyes.mean(0)
        xyz = (m["xyz"] - o) @ Rp.T
        model = dict(xyz=xyz, tri=m["tri"], uv=m["uv"], tex=m["tex"], tid=np.zeros(len(m["tri"]), int),
                     blank=m["blank"], centroid=xyz[m["tri"]].mean(1), vn=vertex_normals(xyz, m["tri"]))
    cams = {"o": o.tolist(), "Rp": Rp.tolist(), "SS": rv.SS, "views": {}}

    def keep(name, cam, img, cut_side=None, ortho=False):
        rv.save(os.path.join(out, f"old_{name}.png"), img)
        cams["views"][name] = dict(eye=cam["eye"].tolist(), R=cam["R"].tolist(), size=list(cam["size"]),
                                   focal=cam.get("focal"), scale=cam.get("scale"), ortho=ortho,
                                   cut_side=None if cut_side is None else cut_side.tolist())

    cam = rv.outside_camera(model, 36, 13, (3400, 1500))
    keep("whole", cam, rv.draw(model, cam, gain=rv.GAIN))
    cam = rv.outside_camera(model, 55, 32, (3400, 1500))
    side = cam["eye"] - np.array([cam["eye"][0], 0.0, 0.0])
    keep("cut", cam, rv.draw(model, cam, ~rv.toward_viewer(model, cam), light=0.25, gain=rv.GAIN), cut_side=side / np.linalg.norm(side))
    # side elevations (render_views.side_rows), orthographic
    p = model["xyz"][np.hypot(model["xyz"][:, 1], model["xyz"][:, 2]) < 0.6]
    x0, x1 = np.percentile(p[:, 0], 0.05) - 0.05, np.percentile(p[:, 0], 99.95) + 0.05
    z0, z1 = np.percentile(p[:, 2], 0.05) - 0.05, np.percentile(p[:, 2], 99.95) + 0.05
    n = int(np.ceil((x1 - x0) / 5.2 - 1e-9))
    rows = []
    for k in range(n):
        a, b = x0 + k * (x1 - x0) / n, x0 + (k + 1) * (x1 - x0) / n
        centre = np.array([0.5 * (a + b), 0.0, 0.5 * (z0 + z1)])
        cam = rv.camera(centre - [0, 50.0, 0], centre, (int(round((b - a) * 640)), int(round((z1 - z0) * 640))), scale=640)
        keep(f"side{k}_out", cam, rv.draw(model, cam, gain=rv.GAIN), ortho=True)
        side = cam["eye"] - np.array([cam["eye"][0], 0.0, 0.0])
        keep(f"side{k}_cut", cam, rv.draw(model, cam, ~rv.toward_viewer(model, cam), light=0.25, gain=rv.GAIN),
             cut_side=side / np.linalg.norm(side), ortho=True)
        rows.append([float(a), float(b), float(z0), float(z1)])
    cams["side_rows"] = rows
    json.dump(cams, open(os.path.join(out, "cams.json"), "w"), indent=1)
    print(f"{len(cams['views'])} views -> {out}")


if __name__ == "__main__":
    main()
