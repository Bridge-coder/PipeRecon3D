"""Download and unpack the two reconstruction programs this repository drives (Windows x64 binaries).

usage: python tools/install_third_party.py [--from-dir DIR]
       --from-dir  take the zips from DIR instead of downloading them (e.g. an offline copy)

  OpenMVG 1.6  (MPL-2.0)  https://github.com/openMVG/openMVG/releases/tag/v1.6
               -> third_party/openMVG_v1.6/ReleaseV1.6.Halibut.WindowsBinaries_VS2017/
  OpenMVS 2.4.0 (AGPL-3.0) https://github.com/cdcseacave/openMVS/releases/tag/v2.4.0
               -> third_party/openMVS_v2.4.0/vc17/x64/Release/

These are the default MVG_BIN / MVS_BIN of scripts/paths.py, so nothing else has to be configured.
Why these versions: OpenMVG 2.0's Windows binaries need AVX-512 (crash on e.g. Ryzen 5000); the scripts are
written against OpenMVG 1.6's program names and its principal-point convention (see vision/fix_mvs_k.py).
"""
import hashlib
import os
import shutil
import sys
import urllib.request
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TP = os.path.join(REPO, "third_party")
PACKAGES = [
    ("https://github.com/openMVG/openMVG/releases/download/v1.6/ReleaseV1.6.Halibut.WindowsBinaries_VS2017.zip",
     "0084adde4f3169d8f00d0eb9966c32962903aa8c77c2709a4d3c11cad0939223", "openMVG_v1.6"),
    ("https://github.com/cdcseacave/openMVS/releases/download/v2.4.0/OpenMVS_Windows_x64.zip",
     "0c31660c15c9ebc4c106873cf67564d9570d404aef7a6403451da1b6178b2167", "openMVS_v2.4.0"),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    src_dir = sys.argv[sys.argv.index("--from-dir") + 1] if "--from-dir" in sys.argv else None
    os.makedirs(os.path.join(TP, "_downloads"), exist_ok=True)
    for url, digest, folder in PACKAGES:
        name = url.rsplit("/", 1)[1]
        zpath = os.path.join(TP, "_downloads", name)
        if src_dir:
            shutil.copyfile(os.path.join(src_dir, name), zpath)
        elif not os.path.exists(zpath) or sha256(zpath) != digest:
            print(f"downloading {url}", flush=True)
            urllib.request.urlretrieve(url, zpath)
        got = sha256(zpath)
        if got != digest:
            sys.exit(f"{name}: SHA-256 {got} does not match the expected {digest}")
        dest = os.path.join(TP, folder)
        print(f"{name}: checksum ok, unpacking to {dest}", flush=True)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(dest)
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import paths
    for exe in (paths.mvg("IncrementalSfM"), paths.mvs("DensifyPointCloud")):
        print(("ok      " if os.path.exists(exe) else "MISSING ") + exe)


if __name__ == "__main__":
    main()
