"""Export the measured surface (the existing model's mesh) next to scene.npz, for anchoring Gaussians (docs/09).

usage: python -I pipesplat/export_surface.py vision WORK_DIR OUT_DIR
       python -I pipesplat/export_surface.py inpipe OUT_DIR [--radius-correction FILE.npz --name surface_rc.npz]
writes OUT_DIR/surface.npz: centres, unit normals and size (sqrt(2 x area), ~ edge length) of every triangle
  vision = the OpenMVS mesh in pipe coordinates (render_views.load_model, the frame scene.npz uses)
  InPipe = the R3 LiDAR height-field mesh up to the pipe exit (lampmap_inpipe.load_mesh, world frame of the poses)
  --radius-correction (docs/10 X6): move every vertex along the radius of the R2 centerline by
      dr(u, theta) = a(u) + b(u) cos(theta) + c(u) sin(theta) from scripts/analysis/depth_referee.py (RealSense depth)
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPTS = os.path.join(ROOT, "scripts")
for p in (os.path.join(SCRIPTS, "vision"), os.path.join(SCRIPTS, "recon"), SCRIPTS, os.path.join(ROOT, "lampmap"), HERE):
    sys.path.insert(0, p)

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)


def faces(xyz, tri):
    a, b, c = xyz[tri[:, 0]], xyz[tri[:, 1]], xyz[tri[:, 2]]
    n = np.cross(b - a, c - a)
    area2 = np.linalg.norm(n, axis=1)
    keep = area2 > 1e-14
    return (((a + b + c) / 3)[keep].astype(np.float32), (n[keep] / area2[keep, None]).astype(np.float32),
            np.sqrt(area2[keep]).astype(np.float32))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="kind", required=True)
    v = sub.add_parser("vision")
    v.add_argument("work"), v.add_argument("out")
    i = sub.add_parser("inpipe")
    i.add_argument("out")
    i.add_argument("--radius-correction", default=None)
    i.add_argument("--name", default="surface.npz")
    a = ap.parse_args()
    t0 = time.time()
    if a.kind == "vision":
        import paths
        import render_views as rv
        m = rv.load_model(paths.CONVERTER, a.work, [None], straighten=False)
        c, n, s = faces(m["xyz"], m["tri"][~m["blank"]])
    else:
        import lampmap_inpipe as li
        m = li.load_mesh()
        xyz = m["xyz"]
        if a.radius_correction:
            from recon_common import BIG, Centerline
            cl = Centerline(os.path.join(BIG, "centerline.npz"))
            rc = np.load(a.radius_correction)
            u, th, r = cl.coords(xyz)
            t = np.radians(th)
            dr = np.interp(u, rc["u"], rc["a"]) + np.interp(u, rc["u"], rc["b"]) * np.cos(t) + np.interp(u, rc["u"], rc["c"]) * np.sin(t)
            xyz = cl.point(u, th, r + dr)[0]
            print(f"radius correction {a.radius_correction}: dr median {np.median(dr) * 1000:+.1f} mm, "
                  f"p5..p95 {np.percentile(dr, 5) * 1000:+.1f}..{np.percentile(dr, 95) * 1000:+.1f} mm")
        c, n, s = faces(xyz, m["tri"])
    os.makedirs(a.out, exist_ok=True)
    np.savez(os.path.join(a.out, getattr(a, "name", "surface.npz")), centres=c, normals=n, size=s)
    print(f"{len(c):,} triangles, size median {np.median(s) * 1000:.2f} mm, p95 {np.percentile(s, 95) * 1000:.2f} mm "
          f"[{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
