"""Table of the Gaussian runs scored with ruler v2 (ROOT/<data>/run_*/summary_r2.json; docs/10 section 2).
Every row is scored on the same pixels (B's mask); the reference rows B, Nal (next frame aligned onto the frame) and
D (the frame itself denoised) come from the first run of each data set (they do not depend on the run).

usage: python -I pipesplat/summarize_r2.py [ROOT=D:/pipe3d_gs/g0] [--md OUT.md]
"""
import glob
import json
import os
import sys

md = sys.argv[sys.argv.index("--md") + 1] if "--md" in sys.argv else None
args = [x for x in sys.argv[1:] if not x.startswith("--") and x != md]
root = args[0] if args else "D:/pipe3d_gs/g0"
COLS = ("psnr", "lpips", "shift_px", "fid_fine", "fid_mid", "fid_coarse", "en_fine", "en_mid", "mse_low")
HEAD = ("PSNR", "LPIPS", "shift px", "fid fine", "fid mid", "fid coarse", "energy fine", "energy mid", "low-freq")
lines = []


def row(name, s, tag):
    return f"| {name} | " + " | ".join(f"{s[f'{tag}_{c}']:.3f}" if c not in ("psnr", "mse_low") else f"{s[f'{tag}_{c}']:.2f}"
                                       for c in COLS)


for ds in sorted(os.listdir(root)):
    runs = sorted(glob.glob(os.path.join(root, ds, "run_*", "summary_r2.json")), key=os.path.getmtime)
    if not runs:
        continue
    for part, title in (("held_outside_blocks", "in-sequence held-out frames (outside the blocks)"), ("block", "block frames (10 cm left out of training)")):
        lines.append(f"\n**{ds} - {title}**\n")
        lines.append("| model | " + " | ".join(HEAD) + " |")
        lines.append("|---" * (len(HEAD) + 1) + "|")
        first = json.load(open(runs[0]))[part]
        for tag, name in (("Nal", "ref: next frame aligned (noisy)"), ("D", "ref: frame denoised (from the frame itself)"), ("B", "B = lampmap (OpenMVS / R4 x lamp map)")):
            lines.append(row(name, first, tag) + " |")
        for p in runs:
            s = json.load(open(p))[part]
            lines.append(row(os.path.basename(os.path.dirname(p)), s, "G") + " |")
text = "\n".join(lines)
print(text)
if md:
    open(md, "w", encoding="utf-8").write(text + "\n")
