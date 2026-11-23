"""Sampling-uncertainty models for fingerprint distances.

Two ideas, both optional and off by default so schema-2.0 behaviour is preserved:

1. Statistic-specific effective sample size. Under a Gaussian AR(1) approximation with lag-1
   autocorrelation rho, the variance of a statistic computed from n serially dependent points is
   inflated by a long-run-variance factor f(rho; statistic), so n_eff = n / f. The factor is NOT the same
   for every statistic (measured in results/research/audit.json):

       mean / trend           f = 1 + 2 sum_k (1 - k/n) rho^k            ~ (1+rho)/(1-rho)
       quantiles / W1 / PSI   f = 1 + 2 sum_k (1 - k/n) (2/pi) asin(rho^k)   (sign-process correlation)
       scale (std, volatility) f = 1 + 2 sum_k (1 - k/n) rho^(2k)         ~ (1+rho^2)/(1-rho^2)
       rank correlation       f = (1 + r_x r_y)/(1 - r_x r_y), r = (6/pi) asin(rho/2) (two feature series)
       lag-k autocorrelation  f = max(1 - rho^2, 0.25)  (Bartlett: the estimate gets *more* precise)
       rates                  f = 1

2. An empirical same-concept noise profile: distances between disjoint windows of one concept's own
   retained observations (repeated-window calibration; moving-block bootstrap when too little data),
   giving P(D | same concept) per view instead of one subtracted constant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

KINDS = ("mean", "quantile", "scale", "acf", "rate")


@dataclass(frozen=True)
class EssConfig:
    shrink: float = 15.0        # rho is shrunk by n/(n+shrink): short windows give noisy rho estimates
    rho_max: float = 0.95
    f_min: float = 0.4          # never claim more than 1/f_min times the nominal sample size
    n_min: float = 3.0
    max_lag: int = 60


def lrv_factor(rho: float, kind: str, n: float, max_lag: int = 60) -> float:
    """Long-run-variance inflation of a statistic of kind ``kind`` from n points of an AR(1) series."""
    k = np.arange(1, int(max(1, min(n - 1, max_lag))) + 1, dtype=float)
    w = 1.0 - k / n
    r = rho ** k
    if kind == "mean":
        g = r
    elif kind == "quantile":
        g = (2 / np.pi) * np.arcsin(r)
    elif kind == "scale":
        g = r ** 2
    else:
        raise ValueError(f"no long-run-variance model for kind {kind!r}")
    return float(1.0 + 2.0 * np.sum(w * g))


def effective_n(n: float, rho: float, kind: str = "mean", cfg: EssConfig = EssConfig()) -> float:
    """Effective number of independent observations for a statistic of ``kind``.

    NaN/undefined rho (constant or too-short window) means no discount. rho is shrunk towards 0 and
    clipped to +-rho_max; the result is bounded to [n_min, n / f_min] and never invalid.
    """
    n = float(n)
    if not np.isfinite(n) or n <= 0:
        raise ValueError("n must be positive")
    r = 0.0 if rho is None or not np.isfinite(rho) else float(rho) * n / (n + cfg.shrink)
    r = float(np.clip(r, -cfg.rho_max, cfg.rho_max))
    if kind == "acf":
        f = max(1.0 - r * r, 0.25)
    elif kind == "rate":
        f = 1.0
    else:
        f = lrv_factor(r, kind, n, cfg.max_lag)
    f = max(f, cfg.f_min)
    return float(np.clip(n / f, min(cfg.n_min, n), n / cfg.f_min))


def effective_n_corr(n: float, rho_i: float, rho_j: float, cfg: EssConfig = EssConfig()) -> float:
    """Effective n for a rank correlation between two serially dependent features."""
    n = float(n)
    adj = lambda r: 0.0 if r is None or not np.isfinite(r) else (6 / np.pi) * np.arcsin(np.clip(float(r) * n / (n + cfg.shrink), -.95, .95) / 2)
    p = adj(rho_i) * adj(rho_j)
    f = max((1 + p) / (1 - p), cfg.f_min)
    return float(np.clip(n / f, min(cfg.n_min, n), n / cfg.f_min))


def nu_vector(n_a: float, rho_a: np.ndarray, n_b: float, rho_b: np.ndarray, kind: str, cfg: EssConfig = EssConfig()) -> np.ndarray:
    """Per-feature noise scale sqrt(1/n_eff_a + 1/n_eff_b) for a statistic of ``kind`` (feature-specific rho)."""
    ra = np.broadcast_to(np.asarray(rho_a, float), np.shape(rho_a))
    rb = np.broadcast_to(np.asarray(rho_b, float), np.shape(rho_b))
    ea = np.array([effective_n(n_a, r, kind, cfg) for r in ra])
    eb = np.array([effective_n(n_b, r, kind, cfg) for r in rb])
    return np.sqrt(1.0 / ea + 1.0 / eb)


# ------------------------------------------------------------------------------------------------
# empirical same-concept profile
# ------------------------------------------------------------------------------------------------
@dataclass
class NoiseProfile:
    """Empirical distribution of un-floored per-view distances between two windows of the SAME concept."""
    n_a: int
    n_b: int
    same: dict[str, np.ndarray] = field(default_factory=dict)
    method: str = "disjoint"
    shift: dict[str, np.ndarray] = field(default_factory=dict)   # un-floored D between a window and one shifted by `delta` sigma
    delta: float = 1.0

    def quantile(self, view: str, p: float) -> float | None:
        s = self.same.get(view)
        return None if s is None or len(s) == 0 else float(np.quantile(s, p))

    def power(self, view: str, alpha: float = 0.05) -> float | None:
        """P(a ``delta``-sigma change exceeds the same-concept (1-alpha) quantile): can this window size resolve it?"""
        s, h = self.same.get(view), self.shift.get(view)
        if s is None or h is None or len(s) == 0 or len(h) == 0:
            return None
        return float(np.mean(h > np.quantile(s, 1 - alpha)))

    def pvalue(self, view: str, d0: float) -> float | None:
        """P(D_same >= d0), add-one smoothed: small = the difference exceeds what sampling noise produces."""
        s = self.same.get(view)
        return None if s is None or len(s) == 0 else float((1 + np.sum(s >= d0)) / (len(s) + 1))


def build_profile(X_ref: np.ndarray, n_a: int, n_b: int, make_fp, compare_fn, B: int = 24,
                  rng: np.random.Generator | None = None, t_ref: np.ndarray | None = None, delta: float = 1.0) -> NoiseProfile:
    """Estimate P(D | same concept) from one concept's retained observations ``X_ref`` (rows in time order).

    Pairs of non-overlapping contiguous segments (sizes n_a, n_b) are drawn at random, so serial
    dependence is preserved and the two windows are independent samples of the concept. If the
    reference is shorter than n_a + n_b, a moving-block bootstrap (block ~ sqrt(n)) is used instead; that
    reuses data and is optimistic, which is recorded in ``method``.

    ``make_fp(X, t)`` builds a feature fingerprint; ``compare_fn(fa, fb)`` returns an object with
    ``.components`` (view -> un-floored D). The caller passes a comparator configured with kappa = 0.
    """
    rng = rng or np.random.default_rng(0)
    R = len(X_ref)
    out: dict[str, list[float]] = {}
    hit: dict[str, list[float]] = {}
    q = np.quantile(X_ref, [0.25, 0.75], axis=0)
    sig = np.maximum((q[1] - q[0]) / 1.349, 1e-9)                      # robust per-feature sigma of the reference
    method = "disjoint" if R >= n_a + n_b + 2 else "block_bootstrap"
    for _ in range(B):
        if method == "disjoint":
            lo = rng.integers(0, R - n_a - n_b + 1)
            gap = rng.integers(0, R - n_a - n_b - lo + 1)
            ia = slice(lo, lo + n_a)
            ib = slice(lo + n_a + gap, lo + n_a + gap + n_b)
            A, Bm = X_ref[ia], X_ref[ib]
        else:
            def boot(n):
                blk = max(2, int(np.sqrt(n)))
                starts = rng.integers(0, max(1, R - blk + 1), int(np.ceil(n / blk)))
                return np.concatenate([X_ref[s:s + blk] for s in starts])[:n]
            A, Bm = boot(n_a), boot(n_b)
        fa, fb = make_fp(A, np.arange(len(A))), make_fp(Bm, np.arange(len(Bm)))
        comps = compare_fn(fa, fb).components
        for k, v in comps.items():
            if v is not None:
                out.setdefault(k, []).append(float(v))
        fs = make_fp(Bm + delta * sig, np.arange(len(Bm)))             # same concept, shifted by delta sigma
        for k, v in compare_fn(fa, fs).components.items():
            if v is not None:
                hit.setdefault(k, []).append(float(v))
    return NoiseProfile(n_a, n_b, {k: np.asarray(v) for k, v in out.items()}, method,
                        {k: np.asarray(v) for k, v in hit.items()}, delta)
