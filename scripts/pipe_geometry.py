"""Cylinder / circle fitting for pipe point clouds (numpy only: this PC has no scipy or open3d).

fit_cylinder(P) finds the straight cylinder that best fits a set of 3-D points (one LiDAR scan or a
short stretch of accumulated scans): axis direction by minimising the circle-fit residual of the points
projected onto the plane perpendicular to the axis, circle by an algebraic fit refined by Gauss-Newton,
outliers (robot parts, end caps, stray returns) rejected by a robust threshold and the fit repeated.
"""
import numpy as np


def perp_basis(d):
    """Two unit vectors that complete d to a right-handed orthonormal basis (e1, e2, d)."""
    d = d / np.linalg.norm(d)
    a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(d, a)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(d, e1)


def fit_circle_2d(uv, w=None, iters=10):
    """Weighted circle fit: algebraic (Kasa) start, then geometric Gauss-Newton. Returns cx, cy, r."""
    if w is None:
        w = np.ones(len(uv))
    u, v = uv[:, 0], uv[:, 1]
    A = np.column_stack([u, v, np.ones_like(u)]) * w[:, None]
    b = (u * u + v * v) * w
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r = np.sqrt(max(sol[2] + cx * cx + cy * cy, 1e-12))
    for _ in range(iters):
        du, dv = u - cx, v - cy
        dist = np.hypot(du, dv) + 1e-12
        res = dist - r
        J = np.column_stack([-du / dist, -dv / dist, -np.ones_like(u)])
        Jw = J * w[:, None]
        step, *_ = np.linalg.lstsq(Jw.T @ J, -(Jw.T @ res), rcond=None)
        cx, cy, r = cx + step[0], cy + step[1], r + step[2]
        if np.abs(step).max() < 1e-9:
            break
    return cx, cy, abs(r)


def project(P, d):
    """Coordinates of P in the plane perpendicular to d (u, v) and along d (s)."""
    e1, e2 = perp_basis(d)
    return np.column_stack([P @ e1, P @ e2]), P @ d, (e1, e2)


def circle_residuals(P, d, w=None):
    uv, s, (e1, e2) = project(P, d)
    cx, cy, r = fit_circle_2d(uv, w)
    res = np.hypot(uv[:, 0] - cx, uv[:, 1] - cy) - r
    return res, (cx, cy, r, e1, e2, s)


def _nelder_mead(f, x0, step=0.05, iters=200, tol=1e-7):
    n = len(x0)
    pts = [np.asarray(x0, float)] + [np.asarray(x0, float) + step * np.eye(n)[i] for i in range(n)]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = np.argsort(vals)
        pts, vals = [pts[i] for i in order], [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        c = np.mean(pts[:-1], axis=0)
        xr = c + (c - pts[-1])
        fr = f(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - pts[-1])
            fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c)
            fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0]] + [pts[0] + 0.5 * (p - pts[0]) for p in pts[1:]]
                vals = [vals[0]] + [f(p) for p in pts[1:]]
    i = int(np.argmin(vals))
    return pts[i], vals[i]


def _refine_direction(P, d0, w):
    e1, e2 = perp_basis(d0)

    def cost(ab):
        d = d0 + ab[0] * e1 + ab[1] * e2
        res, _ = circle_residuals(P, d / np.linalg.norm(d), w)
        return float(np.sum(w * res * res) / np.sum(w))

    ab, _ = _nelder_mead(cost, [0.0, 0.0])
    d = d0 + ab[0] * e1 + ab[1] * e2
    return d / np.linalg.norm(d)


def fit_cylinder(P, d0=None, thresh=None, rounds=4, max_points=20000, seed=0):
    """Fit a straight cylinder to points P (N x 3).

    d0: initial axis direction (e.g. the robot's forward axis); if None the three principal directions
        of P are tried and the one with the smallest circle residual is kept.
    thresh: inlier band in metres around the surface; if None it is 3.5 robust sigmas (MAD) of the
        residuals, recomputed each round.
    Returns dict: point (on the axis, nearest the centroid of the inliers), dir, radius, rms (of inliers),
    inliers (bool mask over P), n_inliers.
    """
    P = np.asarray(P, float)
    rng = np.random.default_rng(seed)
    sub = P if len(P) <= max_points else P[rng.choice(len(P), max_points, replace=False)]
    w = np.ones(len(sub))
    if d0 is None:
        evals, evecs = np.linalg.eigh(np.cov((sub - sub.mean(0)).T))
        best = None
        for k in range(3):
            res, _ = circle_residuals(sub, evecs[:, k])
            score = np.median(np.abs(res))
            if best is None or score < best[0]:
                best = (score, evecs[:, k])
        d = best[1]
    else:
        d = np.asarray(d0, float) / np.linalg.norm(d0)
    for _ in range(rounds):
        d = _refine_direction(sub, d, w)
        res, circ = circle_residuals(sub, d, w)
        sigma = 1.4826 * np.median(np.abs(res - np.median(res[w > 0]))[w > 0])
        band = thresh if thresh is not None else max(3.5 * sigma, 1e-4)
        w = (np.abs(res) < band).astype(float)
        if w.sum() < 50:
            raise ValueError("too few inliers for a cylinder fit")
    # final circle on the inliers of the full set, starting from the subset's weighted circle
    uv, s, (e1, e2) = project(P, d)
    cx, cy, r = circ[:3]
    for _ in range(3):
        inl = np.abs(np.hypot(uv[:, 0] - cx, uv[:, 1] - cy) - r) < band
        cx, cy, r = fit_circle_2d(uv[inl])
    res_all = np.hypot(uv[:, 0] - cx, uv[:, 1] - cy) - r
    inl = np.abs(res_all) < band
    point = cx * e1 + cy * e2 + np.mean(s[inl]) * d
    return {"point": point, "dir": d, "radius": r, "rms": float(np.sqrt(np.mean(res_all[inl] ** 2))),
            "inliers": inl, "n_inliers": int(inl.sum()), "band": float(band), "residuals": res_all}


def fit_ellipse(u, v):
    """Fitzgibbon direct least-squares ellipse. Returns centre (x0, y0), semi-axes a >= b and the angle of
    the major axis from the u axis (rad, -pi/2..pi/2)."""
    D = np.column_stack([u * u, u * v, v * v, u, v, np.ones_like(u)])
    S = D.T @ D
    C = np.zeros((6, 6))
    C[0, 2] = C[2, 0] = 2
    C[1, 1] = -1
    S1, S2, S3 = S[:3, :3], S[:3, 3:], S[3:, 3:]
    T = -np.linalg.solve(S3, S2.T)
    M = S1 + S2 @ T
    M = np.array([M[2] / 2, -M[1], M[0] / 2])
    w, V = np.linalg.eig(M)
    V = np.real(V)
    cond = 4 * V[0] * V[2] - V[1] ** 2
    a1 = V[:, np.argmax(cond > 0)]
    A, B, Cc, Dd, E, F = np.concatenate([a1, T @ a1])
    den = B * B - 4 * A * Cc
    x0 = (2 * Cc * Dd - B * E) / den
    y0 = (2 * A * E - B * Dd) / den
    num = 2 * (A * E * E + Cc * Dd * Dd - B * Dd * E + den * F)
    root = np.sqrt((A - Cc) ** 2 + B * B)
    ax1 = -np.sqrt(num * (A + Cc + root)) / den
    ax2 = -np.sqrt(num * (A + Cc - root)) / den
    ang = 0.5 * np.arctan2(-B, Cc - A)
    if ax1 < ax2:
        ax1, ax2, ang = ax2, ax1, ang + np.pi / 2
    return x0, y0, ax1, ax2, (ang + np.pi / 2) % np.pi - np.pi / 2


def to_pipe_coords(P, point, d, ref_up=(0, 0, 1)):
    """Pipe coordinates of P for a straight axis: s along the axis, theta around it (0 = direction of
    ref_up projected onto the cross-section, increasing right-handed about d), r distance from the axis."""
    d = np.asarray(d, float) / np.linalg.norm(d)
    up = np.asarray(ref_up, float)
    up = up - (up @ d) * d
    up /= np.linalg.norm(up)
    side = np.cross(d, up)
    Q = np.asarray(P, float) - point
    s = Q @ d
    a, b = Q @ up, Q @ side
    return s, np.arctan2(b, a), np.hypot(a, b)


if __name__ == "__main__":  # self-test on a synthetic noisy cylinder with outliers
    rng = np.random.default_rng(1)
    d_true = np.array([0.95, 0.2, 0.1])
    d_true /= np.linalg.norm(d_true)
    e1, e2 = perp_basis(d_true)
    th = rng.uniform(0, 2 * np.pi, 5000)
    s = rng.uniform(-0.3, 0.3, 5000)
    R, c0 = 0.1, np.array([0.01, -0.02, 0.005])
    P = c0 + np.outer(s, d_true) + R * (np.outer(np.cos(th), e1) + np.outer(np.sin(th), e2))
    P += rng.normal(0, 0.002, P.shape)
    P = np.vstack([P, rng.uniform(-0.3, 0.3, (500, 3))])
    f = fit_cylinder(P)
    ang = np.degrees(np.arccos(min(1.0, abs(f["dir"] @ d_true))))
    print(f"radius {f['radius']:.5f} (true {R}), axis error {ang:.3f} deg, rms {f['rms'] * 1000:.2f} mm, "
          f"inliers {f['n_inliers']} of {len(P)} (5000 true)")
