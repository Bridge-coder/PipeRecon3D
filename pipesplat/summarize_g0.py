"""Table of the Gaussian runs (every ROOT/<data>/run_*/summary*.json) with the decision-rule numbers of docs/08 section 6.

usage: python -I pipesplat/summarize_g0.py [ROOT=D:/pipe3d_gs/g0]
"""
import glob
import json
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "D:/pipe3d_gs/g0"
ORDER = ["run_g0", "run_g0_adc", "run_g0a", "run_g0al", "run_g0al_s20", "run_g0ala_s20"]
print("| run | mask | Gaussians | in-seq G PSNR / LPIPS / shift / detail / hf / low | in-seq B PSNR / LPIPS | in-seq N PSNR / LPIPS "
      "| block G PSNR / LPIPS / shift / low | rule 1: dgap dB / dLPIPS | rule 2: depth err in / block |")
for ds in ("concrete_2", "inpipe_172011"):
    runs = sorted((os.path.basename(p) for p in glob.glob(os.path.join(root, ds, "run_*"))),
                  key=lambda r: (ORDER.index(r) if r in ORDER else len(ORDER), r))
    for run in runs:
        for suf, mask in (("", "0.99"), ("_alpha050", "0.5")):
            p = os.path.join(root, ds, run, f"summary{suf}.json")
            if not os.path.exists(p):
                continue
            s = json.load(open(p))
            i, b = s["held_outside_blocks"], s["block"]
            gi, gb = i["N_psnr"] - i["G_psnr"], b["N_psnr"] - b["G_psnr"]
            print(f"| {ds} {run} | {mask} | {s['gaussians']:,} | {i['G_psnr']:.2f} / {i['G_lpips']:.3f} / {i['G_shift_px']:.2f} / "
                  f"{i['G_detail_corr']:.3f} / {i['G_hf_ratio']:.3f} / {i['G_mse_low']:.1f} | {i['B_psnr']:.2f} / {i['B_lpips']:.3f} | "
                  f"{i['N_psnr']:.2f} / {i['N_lpips']:.3f} | {b['G_psnr']:.2f} / {b['G_lpips']:.3f} / {b['G_shift_px']:.1f} / "
                  f"{b['G_mse_low']:.1f} | {gb - gi:+.2f} / {b['G_lpips'] - i['G_lpips']:+.3f} | "
                  f"{i['depth_rel_err']:.1%} / {b['depth_rel_err']:.1%} |")
