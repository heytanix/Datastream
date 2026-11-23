"""Multi-view distance  D = (w_s D_stat + w_d D_dist + w_t D_time + w_p D_pred) over the available views.

Design rules
  * every component is mapped to [0, 1) with  1 - exp(-excess / tau)
  * "excess" is the raw difference minus a sampling-noise allowance (analytic floor, or an empirical
    same-concept profile), so two windows of one concept score ~0 whatever their sizes
  * a component that cannot be computed is *absent*, not zero; ``completeness`` reports how much evidence existed
  * if nothing is available the distance is 1.0 (maximally dissimilar), never 0

All options added after schema 2.0 default to the schema-2.0 behaviour (``ess_mode="legacy"``,
``noise_mode="analytic"``, ``psi_binning="a_deciles"``, ``time_norm="legacy"``, ``corr_transform="abs"``,
``group_agg="mean_worst"``, ``weight_mode="availability"``), so existing results are reproducible.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .mvfp import QGRID, TIME_KEYS, FeatureFingerprint, PredictiveFingerprint
from .noise import EssConfig, KINDS, NoiseProfile, effective_n_corr, nu_vector

_EPS = 1e-9
VIEWS = ("stat", "dist", "time", "pred")
DIST_METRICS = ("wasserstein", "psi", "both", "js", "wasserstein+js", "energy", "wasserstein+energy")
GROUP_AGGS = ("mean_worst", "mean", "median", "trimmed", "weighted", "quantile")


@dataclass(frozen=True)
class SimilarityConfig:
    weights: dict = field(default_factory=lambda: {"stat": 0.40, "dist": 0.30, "time": 0.10, "pred": 0.20})
    tau: dict = field(default_factory=lambda: {"stat": 0.6, "dist": 0.5, "time": 0.35, "pred": 0.25,
                                                "psi": 0.25, "corr": 0.3, "time_z": 2.0})
    kappa: float = 1.0                       # noise-floor multiplier
    dist_metric: str = "both"                # see DIST_METRICS ("both" = wasserstein + psi, the schema-2.0 default)
    use_moment_stats: bool = False           # include mean/std (outlier sensitive) in D_stat
    group_max_weight: float = 0.3            # "mean_worst": blend of mean and worst group
    feature_groups: tuple | None = None      # tuple of tuples of column indices; default: one group per column
    corr_weight: float = 0.25                # share of the correlation term inside D_dist
    similarity: str = "linear"               # "linear": 1 - D   |  "exp": exp(-exp_scale * D)
    exp_scale: float = 3.0
    # -- uncertainty model
    ess_mode: str = "legacy"                 # "legacy" global AR(1) | "per_stat" statistic- and feature-specific
    noise_mode: str = "analytic"             # "analytic" floors | "profile" empirical same-concept profile | "none"
    ess: EssConfig = field(default_factory=EssConfig)
    profile_quantile: float = 0.9            # D at or below this quantile of D_same counts as sampling noise
    max_resolvable: float = 0.5              # analytic mode: noise level at which two windows cannot be told apart
    min_power: float = 0.8                   # profile mode: required power to detect a delta-sigma change, else "unresolvable"
    # -- distribution / temporal / correlation terms
    psi_binning: str = "a_deciles"           # "a_deciles" (schema 2.0, asymmetric) | "pooled" (symmetric)
    tail_weight: float = 0.0                 # weight of the tail-ratio term (q99-q01)/IQR inside D_stat (0 = off)
    time_norm: str = "legacy"                # "legacy" |abs diff| | "standardised" transformed, in standard-error units
    corr_transform: str = "abs"              # "abs" |delta r| | "fisher" |delta atanh r| / SE
    max_corr_pairs: int = 200
    # -- aggregation
    group_agg: str = "mean_worst"            # see GROUP_AGGS
    group_quantile: float = 0.75
    trim: float = 0.2
    group_weights: dict | None = None        # group name -> weight for group_agg="weighted"
    weight_mode: str = "availability"        # "availability" renormalise | "fixed" missing views count as missing_prior | "calibrated"
    missing_prior: float = 0.5
    weights_source: str = "default"          # provenance tag; "calibrated" when produced by a calibration run

    def __post_init__(self):
        if self.dist_metric not in DIST_METRICS:
            raise ValueError(f"dist_metric must be one of {DIST_METRICS}")
        if self.similarity not in ("linear", "exp"):
            raise ValueError("similarity must be linear or exp")
        if self.ess_mode not in ("legacy", "per_stat") or self.noise_mode not in ("analytic", "profile", "none"):
            raise ValueError("bad ess_mode / noise_mode")
        if self.psi_binning not in ("a_deciles", "pooled") or self.time_norm not in ("legacy", "standardised"):
            raise ValueError("bad psi_binning / time_norm")
        if self.corr_transform not in ("abs", "fisher") or self.group_agg not in GROUP_AGGS:
            raise ValueError("bad corr_transform / group_agg")
        if self.weight_mode not in ("availability", "fixed", "calibrated"):
            raise ValueError("bad weight_mode")
        if any(self.weights.get(v, 0) < 0 for v in VIEWS):
            raise ValueError("weights must be non-negative")


@dataclass
class Distances:
    components: dict[str, float | None]      # stat / dist / time / pred -> [0,1) or None if unavailable
    combined: float                          # weighted, in [0, 1]
    similarity: float                        # in [0, 1]
    completeness: float                      # available feature-view weight / total feature-view weight
    groups: dict[str, float] = field(default_factory=dict)
    noise: dict[str, str] = field(default_factory=dict)     # view -> within_noise | above_floor (analytic) | exceeds_noise (profile, p<=0.05) | unresolvable
    noise_source: str = "analytic"

    def as_dict(self) -> dict:
        r = lambda v: None if v is None else round(float(v), 4)
        return {"components": {k: r(v) for k, v in self.components.items()}, "D": r(self.combined),
                "similarity": r(self.similarity), "completeness": r(self.completeness), "noise": dict(self.noise)}


def _norm(excess: float, tau: float) -> float:
    return 1.0 - math.exp(-max(excess, 0.0) / tau)


def _agg_groups(v: np.ndarray, cfg: SimilarityConfig, names: Sequence[str] | None = None) -> float:
    """Combine group distances robustly (see GROUP_AGGS)."""
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    if not ok.any():
        return math.nan
    w = None
    if names is not None and cfg.group_weights:
        w = np.array([cfg.group_weights.get(n, 1.0) for n in names], float)[ok]
    v = v[ok]
    m = cfg.group_agg
    if m == "mean_worst":
        return float((1 - cfg.group_max_weight) * v.mean() + cfg.group_max_weight * v.max())
    if m == "mean":
        return float(v.mean())
    if m == "median":
        return float(np.median(v))
    if m == "trimmed":
        s = np.sort(v)
        k = int(cfg.trim * len(s))
        return float(s[k:len(s) - k].mean()) if len(s) - 2 * k >= 1 else float(s.mean())
    if m == "quantile":
        return float(np.quantile(v, cfg.group_quantile))
    if m == "weighted":
        return float(np.average(v, weights=w if w is not None else np.ones_like(v)))
    raise ValueError(m)


def _groups(cfg: SimilarityConfig, cols: Sequence[int]) -> list[list[int]]:
    if cfg.feature_groups is None:
        return [[c] for c in cols]
    known = set(cols)
    return [[c for c in g if c in known] for g in cfg.feature_groups if any(c in known for c in g)]


def _aggregate(per_col: dict[int, float], cfg: SimilarityConfig) -> tuple[float | None, dict]:
    """Columns -> feature groups (Level 2) -> one number."""
    groups = _groups(cfg, list(per_col))
    gv, names = [], []
    for g in groups:
        vals = np.array([per_col[c] for c in g if c in per_col], float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            gv.append(vals.mean())
            names.append("+".join(map(str, g)))
    if not gv:
        return None, {}
    return _agg_groups(np.array(gv), cfg, names), {n: round(float(x), 4) for n, x in zip(names, gv)}


def _robust_sigma(st: dict) -> np.ndarray:
    sig = (st["q75"] - st["q25"]) / 1.349
    return np.where(np.isfinite(sig) & (sig > _EPS * (np.abs(st["median"]) + 1)), sig, st["std"])


def _cdf(q: np.ndarray, x: np.ndarray) -> np.ndarray:
    """CDF implied by a quantile sketch; a constant sketch is a step function."""
    if q[-1] - q[0] <= _EPS * (abs(q[0]) + 1):
        return (x >= q[0]).astype(float)
    return np.interp(x, q, QGRID, left=0.0, right=1.0)


def _bin_probs(qa: np.ndarray, qb: np.ndarray, binning: str) -> tuple[np.ndarray, np.ndarray]:
    if binning == "pooled":                              # symmetric: deciles of the pooled sketches
        edges = np.quantile(np.concatenate([qa, qb]), np.linspace(0.1, 0.9, 9))
    else:                                                # schema 2.0: deciles of a only
        edges = qa[2:20:2]
    pa = np.diff(np.concatenate([[0.0], _cdf(qa, edges), [1.0]]))
    pb = np.diff(np.concatenate([[0.0], _cdf(qb, edges), [1.0]]))
    pa, pb = np.clip(pa, 1e-3, None), np.clip(pb, 1e-3, None)
    return pa / pa.sum(), pb / pb.sum()


def psi(qa: np.ndarray, qb: np.ndarray, binning: str = "a_deciles") -> float:
    """Population Stability Index of b against a, 10 bins, from quantile sketches (symmetric if binning='pooled')."""
    pa, pb = _bin_probs(qa, qb, binning)
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def js_divergence(qa: np.ndarray, qb: np.ndarray, binning: str = "pooled") -> float:
    """Jensen-Shannon divergence (nats, <= ln 2) on the same bins."""
    pa, pb = _bin_probs(qa, qb, binning)
    m = 0.5 * (pa + pb)
    return float(0.5 * np.sum(pa * np.log(pa / m)) + 0.5 * np.sum(pb * np.log(pb / m)))


def wasserstein1(qa: np.ndarray, qb: np.ndarray) -> float:
    """W1 between two 1-D empirical distributions from their quantile functions (any sample sizes)."""
    return float(np.mean(np.abs(qa - qb)))


def energy_distance(qa: np.ndarray, qb: np.ndarray) -> float:
    """sqrt of the energy distance 2E|X-Y| - E|X-X'| - E|Y-Y'|, treating the sketch points as samples."""
    ab = np.mean(np.abs(qa[:, None] - qb[None, :]))
    aa = np.mean(np.abs(qa[:, None] - qa[None, :]))
    bb = np.mean(np.abs(qb[:, None] - qb[None, :]))
    return float(math.sqrt(max(2 * ab - aa - bb, 0.0)))


# -- uncertainty ------------------------------------------------------------------------------
def _nu_legacy(a: FeatureFingerprint, b: FeatureFingerprint) -> float:
    return math.sqrt(1 / a.effective_n() + 1 / b.effective_n())


def _nus(a: FeatureFingerprint, b: FeatureFingerprint, cfg: SimilarityConfig) -> dict[str, np.ndarray]:
    """Per numeric feature noise scale for each statistic kind."""
    k = len(a.num_idx)
    if cfg.ess_mode == "legacy":
        v = np.full(k, _nu_legacy(a, b))
        return {kd: v for kd in KINDS}
    ra, rb = a.temporal["acf1"], b.temporal["acf1"]
    return {kd: nu_vector(a.n, ra, b.n, rb, kd, cfg.ess) for kd in KINDS}


def _floor_on(cfg: SimilarityConfig) -> float:
    return cfg.kappa if cfg.noise_mode == "analytic" else 0.0


# -- per-view distances -----------------------------------------------------------------------
def d_stat(a: FeatureFingerprint, b: FeatureFingerprint, cfg: SimilarityConfig) -> tuple[float | None, dict, dict]:
    kap = _floor_on(cfg)
    nu = _nus(a, b, cfg)
    sa, sb = a.statistics, b.statistics
    sigA, sigB = _robust_sigma(sa), _robust_sigma(sb)
    s = np.maximum(0.5 * (sigA + sigB), _EPS * (np.abs(sa["median"]) + np.abs(sb["median"]) + 1))
    loc_keys = ["median", "q10", "q25", "q75", "q90"] + (["mean"] if cfg.use_moment_stats else [])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                  # all-NaN columns -> NaN, handled below
        loc = np.nanmean(np.stack([np.clip(np.abs(sa[k] - sb[k]) / s, 0, 5) for k in loc_keys]), axis=0) if len(a.num_idx) else np.zeros(0)
    e = _EPS * (np.abs(sa["median"]) + 1)
    spread = np.clip(np.abs(np.log((sigA + e) / (sigB + e))), 0, 3)
    skew = np.clip(np.abs(sa["skew"] - sb["skew"]) / 1.7, 0, 3)
    miss = np.clip(5 * np.abs(sa["missing"] - sb["missing"]), 0, 5)
    tw = cfg.tail_weight
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if tw > 0:
            ra = (sa["q99"] - sa["q01"]) / np.maximum(sa["q75"] - sa["q25"], _EPS * (np.abs(sa["median"]) + 1))
            rb = (sb["q99"] - sb["q01"]) / np.maximum(sb["q75"] - sb["q25"], _EPS * (np.abs(sb["median"]) + 1))
            tail = np.clip(np.abs(np.log(np.maximum(ra, _EPS) / np.maximum(rb, _EPS))), 0, 3)
        else:
            tail = np.full(len(a.num_idx), np.nan)
    wsum = 2 * np.isfinite(loc) + np.isfinite(spread) + 0.5 * np.isfinite(skew) + np.isfinite(miss) + tw * np.isfinite(tail)
    num = 2 * np.nan_to_num(loc) + np.nan_to_num(spread) + 0.5 * np.nan_to_num(skew) + np.nan_to_num(miss) + tw * np.nan_to_num(tail)
    raw = num / np.where(wsum > 0, wsum, np.nan)
    if cfg.ess_mode == "per_stat":      # weighted mean of the term-specific scales (= kappa*nu when all scales are equal)
        fnum = 2.5 * nu["quantile"] + nu["scale"] + nu["rate"] + tw * 2 * nu["scale"]
        floor = kap * fnum / np.where(wsum > 0, wsum, np.nan)
    else:
        floor = np.full(len(a.num_idx), kap * _nu_legacy(a, b))
    per_col = {}
    for c, r, l, fl in zip(a.num_idx, raw, loc, floor):
        per_col[c] = _norm(r - (fl if np.isfinite(fl) else 0.0), cfg.tau["stat"]) if np.isfinite(r) and np.isfinite(l) else math.nan
    base_nu = _nu_legacy(a, b)
    for i, c in enumerate(a.cat_idx):                    # categorical: total variation, includes mass on unseen categories
        fa, fb = a.categorical["freq"][i], b.categorical["freq"][i]
        keys = set(fa) | set(fb)
        tv = 0.5 * sum(abs(fa.get(k, 0.0) - fb.get(k, 0.0)) for k in keys)
        m = abs(a.categorical["missing"][i] - b.categorical["missing"][i])
        cfloor = kap * 1.5 * math.sqrt(max(len(keys), 1) * base_nu ** 2 / 2)
        per_col[c] = _norm((3 * tv + 5 * m) / 2 - cfloor / 2, cfg.tau["stat"])
    D, grp = _aggregate(per_col, cfg)
    fl = float(np.nanmean(floor)) if len(floor) and np.isfinite(floor).any() else 0.0
    return D, grp, {"floor": fl, "tau": cfg.tau["stat"]}


def _divergences(qa, qb, metric: str, sig: float, nu_q: float, B: int, cfg: SimilarityConfig) -> float:
    kap = _floor_on(cfg)
    parts = []
    scale = max(sig, _EPS * (abs(qa[10]) + 1))
    if metric in ("wasserstein", "both", "wasserstein+js", "wasserstein+energy"):
        parts.append(_norm(min(wasserstein1(qa, qb) / scale, 5.0) - kap * nu_q, cfg.tau["dist"]))
    if metric in ("psi", "both"):
        parts.append(_norm(psi(qa, qb, cfg.psi_binning) - kap * (B - 1) * nu_q ** 2, cfg.tau["psi"]))
    if metric in ("js", "wasserstein+js"):               # 8 x JS ~ PSI for small differences, so the same tau applies
        parts.append(_norm(8 * js_divergence(qa, qb, "pooled") - kap * (B - 1) * nu_q ** 2, cfg.tau["psi"]))
    if metric in ("energy", "wasserstein+energy"):
        parts.append(_norm(min(energy_distance(qa, qb) / scale, 5.0) - kap * nu_q, cfg.tau["dist"]))
    return float(np.mean(parts))


def d_dist(a: FeatureFingerprint, b: FeatureFingerprint, cfg: SimilarityConfig) -> tuple[float | None, dict, dict]:
    kap = _floor_on(cfg)
    nu = _nus(a, b, cfg)
    qa, qb = a.distributions["quantiles"], b.distributions["quantiles"]
    sig = np.maximum(0.5 * (_robust_sigma(a.statistics) + _robust_sigma(b.statistics)), _EPS)
    per_col: dict[int, float] = {}
    for i, c in enumerate(a.num_idx):
        if not (np.isfinite(qa[i]).all() and np.isfinite(qb[i]).all()):
            per_col[c] = math.nan
            continue
        per_col[c] = _divergences(qa[i], qb[i], cfg.dist_metric, sig[i], float(nu["quantile"][i]), 10, cfg)
    feat, grp = _aggregate(per_col, cfg)
    ca, cb = a.distributions["corr"], b.distributions["corr"]
    ok = np.isfinite(ca) & np.isfinite(cb)
    if ok.any():
        idx = np.flatnonzero(ok)
        if len(idx) > cfg.max_corr_pairs:                # deterministic evenly spaced subset, same on both sides
            idx = idx[np.linspace(0, len(idx) - 1, cfg.max_corr_pairs).astype(int)]
        k = len(a.num_idx)
        pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
        if cfg.corr_transform == "fisher":
            za, zb = np.arctanh(np.clip(ca[idx], -.98, .98)), np.arctanh(np.clip(cb[idx], -.98, .98))
            ra, rb = a.temporal["acf1"], b.temporal["acf1"]
            if cfg.ess_mode == "per_stat":
                se = np.array([1.06 * math.sqrt(1 / max(effective_n_corr(a.n, ra[pairs[p][0]], ra[pairs[p][1]], cfg.ess) - 3, 1.0)
                                                + 1 / max(effective_n_corr(b.n, rb[pairs[p][0]], rb[pairs[p][1]], cfg.ess) - 3, 1.0))
                               for p in idx])
            else:
                se = np.full(len(idx), 1.06 * math.sqrt(1 / max(a.effective_n() - 3, 1.0) + 1 / max(b.effective_n() - 3, 1.0)))
            corr = _norm(float(np.mean(np.abs(za - zb))) - kap * 0.8 * float(np.mean(se)), cfg.tau["corr"])
        else:
            corr = _norm(float(np.mean(np.abs(ca[idx] - cb[idx]))) - kap * float(np.mean(nu["mean"])), cfg.tau["corr"])
        feat = corr if feat is None else (1 - cfg.corr_weight) * feat + cfg.corr_weight * corr
    return feat, grp, {"floor": kap * float(np.mean(nu["quantile"])) if len(a.num_idx) else 0.0, "tau": cfg.tau["dist"]}


_TIME_KIND = {"trend": "mean", "acf1": "acf", "acf5": "acf", "acf10": "acf", "season": "acf",
              "volatility": "scale", "roc": "scale", "abrupt": "rate"}
_TIME_COEF = {"trend": 1.0, "acf1": 1.0, "acf5": 1.0, "acf10": 1.0, "season": 1.5, "volatility": 0.71, "roc": 0.71, "abrupt": 0.5}


def _time_transform(key: str, v: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        if key in ("trend", "acf1", "acf5", "acf10", "season"):
            return np.arctanh(np.clip(v, -0.98, 0.98))               # Fisher z for bounded correlation-type statistics
        if key in ("volatility", "roc"):
            return np.log(np.maximum(v, _EPS))                       # ratios: compare on the log scale
        return 2 * np.arcsin(np.sqrt(np.clip(v, 0, 1)))              # rates: variance-stabilising arcsine


def d_time(a: FeatureFingerprint, b: FeatureFingerprint, cfg: SimilarityConfig) -> tuple[float | None, dict, dict]:
    kap = _floor_on(cfg)
    nu = _nus(a, b, cfg)
    per_col: dict[int, float] = {}
    if cfg.time_norm == "legacy":
        D = np.stack([np.abs(a.temporal[k] - b.temporal[k]) * (5.0 if k == "abrupt" else 1.0) for k in TIME_KEYS])
        floor = kap * float(np.mean(nu["mean"])) if len(a.num_idx) else 0.0
        for i, c in enumerate(a.num_idx):
            col = D[:, i]
            per_col[c] = _norm(float(np.nanmean(col)) - floor, cfg.tau["time"]) if np.isfinite(col).any() else math.nan
        D0, grp = _aggregate(per_col, cfg)
        return D0, grp, {"floor": floor, "tau": cfg.tau["time"]}
    # standardised: every statistic becomes a z-score |transformed difference| / its standard error
    Z = []
    for k in TIME_KEYS:
        ta, tb = _time_transform(k, a.temporal[k]), _time_transform(k, b.temporal[k])
        Z.append(np.abs(ta - tb) / np.maximum(nu[_TIME_KIND[k]] * _TIME_COEF[k], _EPS))
    Z = np.stack(Z)
    floor = kap * 0.8                                                # E|z| of pure noise (half-normal)
    for i, c in enumerate(a.num_idx):
        col = Z[:, i]
        per_col[c] = _norm(float(np.nanmean(col)) - floor, cfg.tau["time_z"]) if np.isfinite(col).any() else math.nan
    D0, grp = _aggregate(per_col, cfg)
    return D0, grp, {"floor": floor, "tau": cfg.tau["time_z"]}


def d_pred(a: PredictiveFingerprint, b: PredictiveFingerprint, cfg: SimilarityConfig, min_labeled: int = 10) -> float | None:
    """Distance between two predictive behaviours (stored vs observed). None if either lacks evidence."""
    if a.n_labeled < min_labeled or b.n_labeled < min_labeled or a.classes != b.classes:
        return None
    nu = 0.5 * math.sqrt(1 / a.n_labeled + 1 / b.n_labeled)
    parts = [2 * abs(a.error_rate - b.error_rate)]
    if a.calibration_error is not None and b.calibration_error is not None:
        parts.append(2 * abs(a.calibration_error - b.calibration_error))
    if a.residual_quantiles is not None and b.residual_quantiles is not None:
        parts.append(2 * float(np.mean(np.abs(a.residual_quantiles - b.residual_quantiles))))
    rc = np.abs(a.class_recall - b.class_recall)
    if np.isfinite(rc).any():
        parts.append(2 * float(np.nanmean(rc)))
    rows = np.isfinite(a.confusion).all(axis=1) & np.isfinite(b.confusion).all(axis=1)
    if rows.any():
        parts.append(2 * float(np.mean(0.5 * np.abs(a.confusion[rows] - b.confusion[rows]).sum(axis=1))))
    return _norm(float(np.mean(parts)) - cfg.kappa * nu, cfg.tau["pred"])


# -- combination -------------------------------------------------------------------------------
def _apply_profile(d0: float, view: str, profile: NoiseProfile, cfg: SimilarityConfig) -> tuple[float, str]:
    q = profile.quantile(view, cfg.profile_quantile)
    if q is None:
        return d0, "no_profile"
    p = profile.pvalue(view, d0)
    adj = max(0.0, d0 - q) / max(1.0 - q, _EPS)
    # power is probed with a location shift, so it is only meaningful for the stat and dist views; the temporal
    # view is judged by its p-value alone
    if view in ("stat", "dist"):
        pw = profile.power(view)
        unresolvable = (pw < cfg.min_power) if pw is not None else q >= cfg.max_resolvable
    else:
        unresolvable = False
    status = "unresolvable" if unresolvable else ("exceeds_noise" if p is not None and p <= 0.05 else "within_noise")
    return min(adj, 1.0), status


def compare(a: FeatureFingerprint, b: FeatureFingerprint, cfg: SimilarityConfig | None = None,
            pred: tuple[PredictiveFingerprint, PredictiveFingerprint] | None = None,
            views: Sequence[str] = ("stat", "dist", "time", "pred"), profile: NoiseProfile | None = None) -> Distances:
    """Compare two feature fingerprints (and optionally a stored/observed predictive pair).

    With ``cfg.noise_mode == "profile"`` and a ``profile`` (distribution of un-floored same-concept distances for
    windows of these sizes) each view's excess over the profile quantile is used instead of an analytic floor.
    """
    cfg = cfg or SimilarityConfig()
    if (a.num_idx, a.cat_idx) != (b.num_idx, b.cat_idx):
        raise ValueError("fingerprints describe different feature layouts")
    use_profile = cfg.noise_mode == "profile" and profile is not None
    comps: dict[str, float | None] = {v: None for v in VIEWS}
    noise: dict[str, str] = {}
    groups: dict[str, float] = {}
    fns = {"stat": d_stat, "dist": d_dist, "time": d_time}
    for v in ("stat", "dist", "time"):
        if v not in views:
            continue
        D, g, meta = fns[v](a, b, cfg)
        if v == "stat":
            groups = g
        if D is None:
            continue
        if use_profile:
            D, noise[v] = _apply_profile(D, v, profile, cfg)
        else:
            floor_level = _norm(meta["floor"], meta["tau"]) if cfg.noise_mode == "analytic" else 0.0
            # analytic mode has no significance test: "above_floor" only says the raw distance exceeds the noise allowance;
            # "exceeds_noise" (p <= 0.05) is reserved for the empirical profile
            noise[v] = "unresolvable" if floor_level >= cfg.max_resolvable else ("above_floor" if D > 0 else "within_noise")
        comps[v] = D
    if "pred" in views and pred is not None:
        comps["pred"] = d_pred(pred[0], pred[1], cfg)
    avail = {k: v for k, v in comps.items() if v is not None and math.isfinite(v) and cfg.weights.get(k, 0) > 0}
    req = [k for k in VIEWS if k in views and cfg.weights.get(k, 0) > 0 and (k != "pred" or pred is not None)]
    feat_total = sum(cfg.weights[k] for k in ("stat", "dist", "time") if k in views)
    wtot = sum(cfg.weights[k] for k in avail)
    src = "profile" if use_profile else "analytic" if cfg.noise_mode == "analytic" else "none"
    if wtot <= 0:
        return Distances(comps, 1.0, 0.0, 0.0, groups, noise, src)
    if cfg.weight_mode == "fixed":                       # constant total weight; a missing view counts as `missing_prior`
        missing = [k for k in req if k not in avail]
        D = (sum(cfg.weights[k] * v for k, v in avail.items()) + sum(cfg.weights[k] * cfg.missing_prior for k in missing)) / \
            (wtot + sum(cfg.weights[k] for k in missing))
    else:                                                # availability-normalised (schema 2.0) and calibrated weights
        D = sum(cfg.weights[k] * v for k, v in avail.items()) / wtot
    D = min(1.0, max(0.0, D))
    sim = 1.0 - D if cfg.similarity == "linear" else math.exp(-cfg.exp_scale * D)
    comp_ok = sum(cfg.weights[k] for k in ("stat", "dist", "time") if k in avail)
    return Distances(comps, D, sim, comp_ok / feat_total if feat_total > 0 else 0.0, groups, noise, src)
