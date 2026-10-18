"""Rewrite the principal point of an OpenMVS scene written by OpenMVG 1.6 so that OpenMVS 2.4.0 reads it correctly.

Usage: fix_mvs_k.py IN_MVS OUT_MVS IMAGE_WIDTH IMAGE_HEIGHT

OpenMVG 1.6 stores the camera matrix without the image size and simply divides it by max(width, height):
cx/s, cy/s. OpenMVS 2.4.0 turns such a matrix back into pixels with (c + 0.5) * s - 0.5, i.e. it expects
(cx + 0.5)/s - 0.5. Left as it is, the principal point ends up half an image too far to the right and
down (768x548 frames: 766.5, 671.5 instead of 383, 288) and every dense result is computed with wrong rays.
Only the two principal-point numbers of each camera are changed; the file is otherwise copied byte for byte.
"""
import struct
import sys

src, dst, width, height = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
data = bytearray(open(src, "rb").read())
if data[:4] != b"MVSI":
    sys.exit(f"{src}: not an OpenMVS interface file")
version = struct.unpack_from("<I", data, 4)[0]
s = float(max(width, height))
pos = 12                                    # magic, version, reserved


def u64():
    global pos
    (v,) = struct.unpack_from("<Q", data, pos)
    pos += 8
    return v


def skip_string():
    global pos
    n = u64()
    pos += n


changed = 0
for p in range(u64()):                      # platforms
    skip_string()                           # platform name
    for c in range(u64()):                  # cameras
        skip_string()                       # camera name
        if version > 3:
            skip_string()                   # band name
        w = h = 0
        if version > 0:
            w, h = struct.unpack_from("<2I", data, pos)
            pos += 8
        k = struct.unpack_from("<9d", data, pos)
        if w or h:
            print(f"platform {p} camera {c}: image size {w}x{h} is stored, matrix is in pixels - left unchanged")
        elif not (0.0 < k[2] < 1.0 and 0.0 < k[5] < 1.0):
            sys.exit(f"platform {p} camera {c}: principal point {k[2]:.6f}, {k[5]:.6f} is not of the form "
                     f"pixels / max side - already converted? Nothing written.")
        else:
            cx, cy = k[2] * s, k[5] * s
            nx, ny = (cx + 0.5) / s - 0.5, (cy + 0.5) / s - 0.5
            struct.pack_into("<d", data, pos + 16, nx)
            struct.pack_into("<d", data, pos + 40, ny)
            changed += 1
            print(f"platform {p} camera {c}: focal {k[0] * s:.3f} px, principal point {cx:.3f}, {cy:.3f} px")
            print(f"   stored {k[2]:.6f}, {k[5]:.6f}  ->  {nx:.6f}, {ny:.6f}")
            print(f"   OpenMVS would have used {(k[2] + 0.5) * s - 0.5:.3f}, {(k[5] + 0.5) * s - 0.5:.3f} px; "
                  f"now uses {(nx + 0.5) * s - 0.5:.3f}, {(ny + 0.5) * s - 0.5:.3f} px")
        pos += 8 * (9 + 9 + 3)              # K, R, C
    n_poses = u64()
    pos += n_poses * 8 * (9 + 3)            # poses: R, C
if not changed:
    sys.exit("no camera matrix was changed; nothing written")
open(dst, "wb").write(data)
print(f"{changed} camera(s) changed; written {dst}")
