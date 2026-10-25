"""docs/10 X8 / X8b: ruler-v2 medians of scored runs per pipe section (in-sequence held-out frames outside the blocks,
from each run's per_frame_r2.csv; columns G_* = the run's own render): x <= 1.5 m (stains, joint), 1.5-3.25 m (plain
wall), > 3.25 m (the end cap fills the middle of the frame), plus the overall in-seq / block medians.

usage: python -I pipesplat/review/section_r2.py G_DIR RUN [RUN ...] [--md]
"""
import csv
import os
import sys

import numpy as np

SECTIONS = (("x <= 1.5 m", -9.0, 1.5), ("1.5-3.25 m", 1.5, 3.25), ("x > 3.25 m", 3.25, 9.0))
KEYS = ("psnr", "fid_fine", "fid_mid", "fid_coarse", "shift_px")


def med(rows, k):
    return float(np.median([float(r["G_" + k]) for r in rows]))


def main():
    md = "--md" in sys.argv
    args = [s for s in sys.argv[1:] if s != "--md"]
    gdir, runs = args[0], args[1:]
    head = ["run"] + [f"{name} ({{}})" for name, _, _ in SECTIONS] + ["in-seq", "block"]
    lines = []
    for run in runs:
        rows = list(csv.DictReader(open(os.path.join(gdir, run, "per_frame_r2.csv"), encoding="utf-8")))
        held = [r for r in rows if r["tag"] == "held" and r["in_block"] == "False"]
        block = [r for r in rows if r["tag"] == "block"]
        cells, counts = [run], []
        for _, lo, hi in SECTIONS:
            s = [r for r in held if lo < float(r["pos"]) <= hi]
            counts.append(len(s))
            cells.append(" / ".join(f"{med(s, k):.2f}" for k in KEYS[:4]))
        for s in (held, block):
            cells.append(" / ".join(f"{med(s, k):.3f}" if k != "psnr" else f"{med(s, k):.2f}" for k in KEYS))
        lines.append(cells)
    head = [h.format(c) if "{}" in h else h for h, c in zip(head, [None] + counts + [None, None])]
    if md:
        print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
        for c in lines:
            print("| " + " | ".join(c) + " |")
    else:
        print("sections: PSNR / fid fine / mid / coarse (medians); in-seq, block: + shift px")
        for c in [head] + lines:
            print("  ".join(f"{v:<28}" for v in c))


if __name__ == "__main__":
    main()
