"""docs/11: one table per data set - the earlier deliverable (B), published methods (3DGS, 3DGS-MCMC, 2DGS) and ours
(10-21 best and this round's), all from summary_r2.json (ruler v2: the same frames, the same pixels).
Also writes a JSON with the same numbers (for the flow diagram).

usage: python -I pipesplat/review/compare_methods.py [--md OUT.md] [--json OUT.json]
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
G = "D:/pipe3d_gs/g0"
SETS = [  # data set, title, [(run, label, kind)]  kind: base / pub / ours_old (10-21) / ours (best now) / ours_try (variant)
    ("concrete_2", "concrete_2（Sheffield 混凝土管）",
     [("run_3dgs", "3DGS", "pub"), ("run_mcmc", "3DGS-MCMC（gsplat 默认参数，训练发散）", "pub"), ("run_2dgs", "2DGS", "pub"),
      ("run_g0a", "本项目 G0-A（当前最好）", "ours"), ("run_g0a_opaque", "本项目 G0-A 不透明面元（Y4，未采用）", "ours_try"),
      ("run_g0a_pose", "本项目 G0-A + 位姿精修（未采用）", "ours_try")]),
    ("concrete_lab", "实验室管 A（独立检验）",
     [("run_3dgs", "3DGS", "pub"), ("run_mcmc", "3DGS-MCMC", "pub"), ("run_2dgs", "2DGS（种子 0：数值发散，NaN）", "pub"),
      ("run_2dgs_s1", "2DGS（种子 1：同样发散）", "pub"), ("run_g0a", "本项目 G0-A", "ours")]),
    ("inpipe_vision", "InPipe 172011（激光 + 视觉）",
     [("run_3dgs", "3DGS", "pub"), ("run_mcmc", "3DGS-MCMC", "pub"), ("run_2dgs", "2DGS", "pub"),
      ("run_g0ala_s20", "本项目 X7（10-21 最好）", "ours_old"), ("run_vhf", "本项目 Y1（视觉高度场管壁）", "ours")]),
    ("sewer_pipe1", "Pipe1 真实污水管（前 10 m）",
     [("run_3dgs", "3DGS", "pub"), ("run_mcmc", "3DGS-MCMC", "pub"), ("run_2dgs", "2DGS", "pub"),
      ("run_g0a", "本项目 G0-A，锚定在 OpenMVS 网格（对照）", "ours_try"), ("run_vhf", "本项目 Y3（视觉高度场管壁）", "ours")]),
]
B_LABEL = {"concrete_2": "OpenMVS 网格贴图 × 灯光图（v1.0 = B）", "concrete_lab": "OpenMVS 网格贴图 × 灯光图（v1.0 = B）",
           "inpipe_vision": "激光高度场 R3 + R4 贴图 × 灯光图（v1.0 = B）",
           "sewer_pipe1": "OpenMVS 网格贴图 × 灯光图（B，见过留出帧）"}
COLS = ("psnr", "lpips", "shift_px", "fid_fine", "fid_mid", "fid_coarse")


def fmt(s, tag):
    return [round(s[f"{tag}_{c}"], 2 if c in ("psnr", "shift_px") else 3) for c in COLS]


md = sys.argv[sys.argv.index("--md") + 1] if "--md" in sys.argv else None
js = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
out, lines = {}, []
for ds, title, runs in SETS:
    rows = []
    first = None
    for run, label, kind in runs:
        p = os.path.join(G, ds, run, "summary_r2.json")
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        first = first or j
        h, b = j["held_outside_blocks"], j["block"]
        rows.append(dict(run=run, label=label, kind=kind, held=fmt(h, "G"), block=fmt(b, "G"),
                         depth=round(float(h.get("depth_rel_err", float("nan"))) * 100, 1),
                         gaussians=j.get("gaussians"), minutes=j.get("train_minutes")))
    if first is None:
        continue
    h, b = first["held_outside_blocks"], first["block"]
    refs = [dict(run="B", label=B_LABEL[ds], kind="base", held=fmt(h, "B"), block=fmt(b, "B"), depth=None),
            dict(run="Nal", label="参照：对齐后的下一帧（带噪声的真实画面）", kind="ref", held=fmt(h, "Nal"), block=fmt(b, "Nal"),
                 depth=None)]
    out[ds] = dict(title=title, rows=refs + rows)
    lines.append(f"\n**{title}**（序列内留出帧 | 整段留出帧；PSNR dB · LPIPS · 错位 px · 细 / 中 / 粗 fid；深度误差 = 渲染深度对实测几何）\n")
    lines.append("| 方法 | 序列内 PSNR | LPIPS | 错位 | 细 / 中 / 粗 | 整段留出 PSNR | LPIPS | 错位 | 细 / 中 / 粗 | 深度误差 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in refs + rows:
        hh, bb = r["held"], r["block"]
        name = f"**{r['label']}**" if r["kind"] == "ours" else r["label"]
        dep = "—" if r["depth"] is None or r["depth"] != r["depth"] else f"{r['depth']:.1f} %"
        lines.append(f"| {name} | {hh[0]:.2f} | {hh[1]:.3f} | {hh[2]:.2f} | {hh[3]:.3f} / {hh[4]:.3f} / {hh[5]:.3f} | "
                     f"{bb[0]:.2f} | {bb[1]:.3f} | {bb[2]:.2f} | {bb[3]:.3f} / {bb[4]:.3f} / {bb[5]:.3f} | {dep} |")
text = "\n".join(lines)
print(text)
if md:
    open(md, "w", encoding="utf-8").write(text + "\n")
if js:
    json.dump(out, open(js, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
