"""Image comparison of a render of the model against the real video frame, over the pixels the render covers (mask).
numpy + OpenCV; LPIPS only where torch + lpips are installed (otherwise NaN).

  linear_match   per-channel gain and offset of the render fitted to the real frame (the frame's exposure)
  psnr           over the mask, RGB, 8-bit range
  ssim           grey, Gaussian window sigma 1.5 (Wang et al. 2004), averaged over the mask eroded by the window.
                 On noisy analog video SSIM (and PSNR) also reward blur - read them together with lpips/detail_corr
  lpips          LPIPS (AlexNet, spatial) averaged over the mask (lower = closer)
  shift_px       median optical-flow (DIS) length between the CLAHE-normalised frames, where the frame has structure:
                 how far the model's texture lies from where the video shows it
  detail_corr    correlation of the 1-3 px detail (difference of Gaussians) of render and frame
  hf_ratio       fine-detail energy of the render relative to the frame's (> 1: more or sharper detail than the video)
  mse_low/high   squared grey residual below / above 6 px: low = smooth shading (lamp, vignetting), high = texture
"""
import cv2
import numpy as np

C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
_LPIPS = []


def grey(rgb):
    return rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)


def erode(m, r):
    return cv2.erode(m.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0


def linear_match(ren, real, m):
    """render with a per-channel gain and offset fitted (least squares) to the real frame over the mask"""
    out = ren.astype(np.float32).copy()
    for c in range(3):
        x, y = ren[..., c][m].astype(np.float64), real[..., c][m].astype(np.float64)
        a, b = np.polyfit(x, y, 1) if x.std() > 1e-6 else (1.0, y.mean() - x.mean())
        out[..., c] = a * out[..., c] + b
    return np.clip(out, 0, 255)


def psnr(a, b, m):
    d = a[m].astype(np.float64) - b[m].astype(np.float64)
    return float(10 * np.log10(255.0 ** 2 / max(np.mean(d ** 2), 1e-12)))


def ssim(a, b, m):
    x, y = grey(a), grey(b)
    g = lambda z: cv2.GaussianBlur(z, (0, 0), 1.5)  # noqa: E731
    mx, my = g(x), g(y)
    sxx, syy, sxy = g(x * x) - mx * mx, g(y * y) - my * my, g(x * y) - mx * my
    s = ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx * mx + my * my + C1) * (sxx + syy + C2))
    inner = erode(m, 5)
    return float(s[inner].mean()) if inner.any() else np.nan


def misalignment(ren, real, m, margin=15):
    a = _CLAHE.apply(np.clip(grey(real), 0, 255).astype(np.uint8))
    b = _CLAHE.apply(np.clip(grey(ren), 0, 255).astype(np.uint8))
    fl = _DIS.calc(a, b, None)
    inner = erode(m, margin)
    if inner.sum() < 100:
        return np.nan
    gs = cv2.GaussianBlur(a.astype(np.float32), (0, 0), 1.5)
    gm = np.hypot(cv2.Sobel(gs, cv2.CV_32F, 1, 0), cv2.Sobel(gs, cv2.CV_32F, 0, 1))
    sel = inner & (gm >= np.percentile(gm[inner], 80))
    return float(np.median(np.hypot(fl[..., 0], fl[..., 1])[sel]))


def hf_ratio(ren, real, m, sigma=2.0):
    hp = lambda z: z - cv2.GaussianBlur(z, (0, 0), sigma)  # noqa: E731
    return float(np.mean(hp(grey(ren))[m] ** 2) / max(np.mean(hp(grey(real))[m] ** 2), 1e-9))


def detail_corr(ren, real, m, s1=1.0, s2=3.0):
    bp = lambda z: cv2.GaussianBlur(z, (0, 0), s1) - cv2.GaussianBlur(z, (0, 0), s2)  # noqa: E731
    a, b = bp(grey(ren))[m].astype(np.float64), bp(grey(real))[m].astype(np.float64)
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / max(np.sqrt((a @ a) * (b @ b)), 1e-12))


def band_split(ren, real, m, sigma=6.0):
    w = m.astype(np.float32)
    ws = cv2.GaussianBlur(w, (0, 0), sigma) + 1e-6
    r = grey(real) - grey(ren)
    rl = cv2.GaussianBlur(r * w, (0, 0), sigma) / ws
    return float(np.mean(rl[m] ** 2)), float(np.mean((r - rl)[m] ** 2))


def lpips_masked(ren, real, m):
    """outside the mask the render is replaced by the real frame, so only the covered area counts; the map is
    averaged over the mask eroded by 15 px"""
    if not _LPIPS:
        try:
            import lpips
            import torch
        except ImportError:
            _LPIPS.append(None)
        else:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            _LPIPS.append((lpips.LPIPS(net="alex", spatial=True, verbose=False).to(dev).eval(), dev, torch))
    if _LPIPS[0] is None:
        return np.nan
    net, dev, torch = _LPIPS[0]
    a = np.where(m[..., None], ren, real).astype(np.float32)
    t = lambda z: torch.from_numpy(np.ascontiguousarray(z, np.float32) / 127.5 - 1).permute(2, 0, 1)[None].to(dev)  # noqa: E731
    with torch.no_grad():
        d = net(t(a), t(real))[0, 0].cpu().numpy()
    inner = erode(m, 15)
    return float(d[inner].mean()) if inner.any() else np.nan


def all_metrics(ren, real, m):
    """metrics of an already brightness-matched render"""
    lo, hi = band_split(ren, real, m)
    return dict(psnr=psnr(ren, real, m), ssim=ssim(ren, real, m), lpips=lpips_masked(ren, real, m),
                shift_px=misalignment(ren, real, m), detail_corr=detail_corr(ren, real, m),
                hf_ratio=hf_ratio(ren, real, m), mse_low=lo, mse_high=hi)


# ------------------------------------------------------------------ ruler v2 (docs/10 section 2) ----------
# The video is noisy (analog Sheffield video: about 3-4 grey levels per frame), so a perfect, noise-free model
# cannot match the frame's fine detail, and LPIPS against the noisy frame mostly scores the noise: the frame
# itself after denoising gets the same LPIPS (0.38 on concrete_2) as the Gaussian models. Ruler v2 measures
# detail against the part of the frame that is not noise, using the next video frame aligned onto the frame
# (same scene, independent noise) to tell the two apart, in three bands:
#   fine 0.7-1.4 px, mid 1.4-2.8 px, coarse 2.8-5.6 px (differences of Gaussian blurs)
#   corr_<b>  correlation of render and frame in the band (noise lowers it equally for every model)
#   fid_<b>   corr_<b> / sqrt(corr of frame and aligned next frame): the correlation with the noise-free frame,
#             1.0 = every detail in place (the aligned next frame itself gets sqrt of its corr: it carries noise)
#   en_<b>    band energy of the render / the frame's signal energy (cross-energy of frame and aligned next
#             frame): 1.0 = as much real detail as the video, < 1 blurred, > 1 invented detail or noise
BANDS = (("fine", 0.7, 1.4), ("mid", 1.4, 2.8), ("coarse", 2.8, 5.6))


def align_to(src, real):
    """src warped onto real by DIS optical flow (CLAHE-normalised, as misalignment())"""
    a = _CLAHE.apply(np.clip(grey(real), 0, 255).astype(np.uint8))
    b = _CLAHE.apply(np.clip(grey(src), 0, 255).astype(np.uint8))
    fl = _DIS.calc(a, b, None)
    h, w = a.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(src.astype(np.float32), gx + fl[..., 0], gy + fl[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def _band(z, s1, s2):
    return cv2.GaussianBlur(z, (0, 0), s1) - cv2.GaussianBlur(z, (0, 0), s2)


def ruler2(ren, real, nal, m):
    """ruler v2 band metrics of an already brightness-matched render; nal = next frame aligned onto real"""
    out = {}
    g_r, g_f, g_n = grey(ren), grey(real), grey(nal)
    inner = erode(m, 6)
    for name, s1, s2 in BANDS:
        a, b, c = (_band(z, s1, s2)[inner].astype(np.float64) for z in (g_r, g_f, g_n))
        a, b, c = a - a.mean(), b - b.mean(), c - c.mean()
        corr = (a @ b) / max(np.sqrt((a @ a) * (b @ b)), 1e-12)
        ceil = (c @ b) / max(np.sqrt((c @ c) * (b @ b)), 1e-12)
        sig = max(c @ b, 1e-9)
        out[f"corr_{name}"] = float(corr)
        out[f"fid_{name}"] = float(corr / np.sqrt(max(ceil, 1e-3)))
        out[f"en_{name}"] = float((a @ a) / sig)
        out[f"ceil_{name}"] = float(ceil)
    return out
