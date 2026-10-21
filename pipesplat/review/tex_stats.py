"""docs/10: value range of a trained TX texture (is the [0, 1.5] clamp of train_tex.py hit?)"""
import sys
import torch
for run in sys.argv[1:]:
    t = torch.load(run + "/tex.pt", map_location="cpu")["tex"][0]
    g = t.mean(0).flatten()
    seen = (g - 0.3).abs() > 1e-4
    q = torch.quantile(g[seen][torch.randperm(int(seen.sum()))[:2000000]], torch.tensor([0.01, 0.5, 0.9, 0.99]))
    print(run.split("/g0/")[1], f"seen {seen.float().mean().item():.1%}; at the 1.5 clamp {(t >= 1.4999).any(0).float().mean().item():.1%}; "
          f"> 1 {(t > 1).any(0).float().mean().item():.1%}; grey p1/p50/p90/p99 {[round(x, 3) for x in q.tolist()]}")
