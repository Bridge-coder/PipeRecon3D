"""Textured mesh of the texture model TX (train_tex.py, docs/10 X4): the measured surface (vision = the OpenMVS mesh
in the pipe frame of scene.npz; InPipe = the R3 height-field mesh) with UVs into TX's unrolled texture.
Runs on Windows (reads the meshes the way export_surface.py does).

usage: python -I pipesplat/export_tex_obj.py vision WORK_DIR RUN_DIR [OUT_DIR=RUN_DIR/obj] [--mesh MESH.npz]
       python -I pipesplat/export_tex_obj.py inpipe RUN_DIR [OUT_DIR]
--mesh: vertices + faces from this file (e.g. bake_gs_tex.py's mesh_bake.npz: the OpenMVS mesh moved onto the Gaussian
model's surface) instead of the work dir's OpenMVS mesh.
writes OUT_DIR/model.obj + model.mtl + texture.png (+ cap.png): open in MeshLab / Blender.
Faces that cross the theta = +-pi seam get their own copies of the vertices (theta + 2 pi), so the texture does
not smear across the seam; end-cap faces of a closed pipe use the planar cap texture.
"""
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPTS = os.path.join(ROOT, "scripts")
for p in (os.path.join(SCRIPTS, "vision"), os.path.join(SCRIPTS, "recon"), SCRIPTS, os.path.join(ROOT, "lampmap"), HERE):
    sys.path.insert(0, p)
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)


def main():
    argv = list(sys.argv)
    mesh_npz = None
    if "--mesh" in argv:
        i = argv.index("--mesh")
        mesh_npz = argv[i + 1]
        del argv[i:i + 2]
    kind = argv[1]
    if kind == "vision":
        work, run = argv[2], argv[3]
        out = argv[4] if len(argv) > 4 else os.path.join(run, "obj")
        if mesh_npz:
            mz = np.load(mesh_npz)
            xyz, tri = mz["xyz"], mz["tri"]
        else:
            import paths
            import render_views as rv
            m = rv.load_model(paths.CONVERTER, work, [None], straighten=False)
            xyz, tri = m["xyz"], m["tri"][~m["blank"]]
    else:
        run = argv[2]
        out = argv[3] if len(argv) > 3 else os.path.join(run, "obj")
        import lampmap_inpipe as li
        m = li.load_mesh()
        xyz, tri = m["xyz"], m["tri"]
    t = json.load(open(os.path.join(run, "texture.json"), encoding="utf-8"))
    U0, du, dth, NU, NTH = t["U0"], t["du"], t["dth"], t["NU"], t["NTH"]
    if kind == "vision":
        u, th = xyz[:, 0], np.arctan2(-xyz[:, 1], xyz[:, 2])
    else:
        from recon_common import Centerline
        cl = Centerline(os.path.join(os.path.dirname(os.path.dirname(run)), "centerline.npz")
                        if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(run)), "centerline.npz"))
                        else os.path.join(li.BIG, "centerline.npz"))
        u, thd, _ = cl.coords(xyz)
        th = np.radians(thd)
    a, b, c = xyz[tri[:, 0]], xyz[tri[:, 1]], xyz[tri[:, 2]]
    n = np.cross(b - a, c - a)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    cap = np.zeros(len(tri), bool)
    if t.get("cap_x") is not None:
        cap = (xyz[tri].mean(1)[:, 0] > t["cap_x"]) & (np.abs(n[:, 0]) > 0.5)
    # UVs (OBJ: v = 0 at the bottom of the image; texture row r covers theta = -pi + r dth)
    vt_u = (u - U0) / (NU * du)
    tht = th[tri]
    seam = (tht.max(1) - tht.min(1)) > np.pi
    vt = [np.column_stack([vt_u, 1 - (th + np.pi) / (2 * np.pi)])]
    faces_uv = tri.copy()
    if seam.any():                                   # copies of the seam vertices with theta + 2 pi
        sv = np.unique(tri[seam][tht[seam] < 0])
        new = {v: len(xyz) + k for k, v in enumerate(sv)}
        vt.append(np.column_stack([vt_u[sv], 1 - (th[sv] + 3 * np.pi) / (2 * np.pi)]))
        ft = faces_uv[seam]
        neg = tht[seam] < 0
        ft[neg] = np.vectorize(new.get)(ft[neg])
        faces_uv[seam] = ft
    vt = np.concatenate(vt)
    if cap.any():                                    # planar cap texture: (y, z) in [-half, half]
        half = t["cap_half"]
        cv = np.unique(tri[cap])
        base = len(vt)
        idx = {v: base + k for k, v in enumerate(cv)}
        vt = np.concatenate([vt, np.column_stack([(xyz[cv, 1] + half) / (2 * half), 1 - (xyz[cv, 2] + half) / (2 * half)])])
        faces_uv[cap] = np.vectorize(idx.get)(tri[cap])
    os.makedirs(out, exist_ok=True)
    shutil.copy(os.path.join(run, "texture.png"), os.path.join(out, "texture.png"))
    with open(os.path.join(out, "model.mtl"), "w", encoding="utf-8") as fh:
        fh.write("newmtl wall\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nmap_Kd texture.png\n")
        if cap.any():
            shutil.copy(os.path.join(run, "cap.png"), os.path.join(out, "cap.png"))
            fh.write("newmtl cap\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nmap_Kd cap.png\n")
    with open(os.path.join(out, "model.obj"), "w", encoding="utf-8") as fh:
        fh.write("# PipeRecon3D texture model TX (pipesplat/train_tex.py); metres\nmtllib model.mtl\n")
        fh.write("".join(f"v {x:.5f} {y:.5f} {z:.5f}\n" for x, y, z in xyz))
        fh.write("".join(f"vt {p:.6f} {q:.6f}\n" for p, q in vt))
        for name, sel in (("wall", ~cap), ("cap", cap)):
            if not sel.any():
                continue
            fh.write(f"usemtl {name}\n")
            fh.write("".join(f"f {i + 1}/{j + 1} {k + 1}/{l + 1} {p + 1}/{q + 1}\n"
                             for (i, k, p), (j, l, q) in zip(tri[sel], faces_uv[sel])))
    print(f"{len(xyz):,} vertices, {len(tri):,} faces ({seam.sum():,} across the seam, {cap.sum():,} end cap) -> {out}")


if __name__ == "__main__":
    main()
