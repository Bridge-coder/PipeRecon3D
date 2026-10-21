"""docs/10: distribution of the Gaussians' sizes and opacities in trained runs (is there a tail of big blobs?)"""
import sys
import torch
for run in sys.argv[1:]:
    m = torch.load(run + "/model.pt")
    s = torch.exp(m["scales"]).max(1).values * 1000
    o = torch.sigmoid(m["opacities"])
    q = torch.quantile(s[torch.randperm(len(s))[:1000000]], torch.tensor([0.5, 0.9, 0.99, 0.999]))
    print(run.split("/g0/")[1], f"{len(s):,} Gaussians; largest scale mm p50/p90/p99/p99.9 {[round(x, 2) for x in q.tolist()]}; "
          f"max {s.max().item():.0f} mm; share > 5 mm {(s > 5).float().mean().item():.2%} (their mean opacity "
          f"{o[s > 5].mean().item():.2f}); > 20 mm {(s > 20).float().mean().item():.3%}; opacity median {o.median().item():.3f}")
