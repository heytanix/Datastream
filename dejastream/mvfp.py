"""Multi-view concept fingerprints (schema 2.0).

A fingerprint is *evidence about* a concept, not an identity. It is split into views:

  feature fingerprint  F_X = (statistics, distributions, temporal)   computed from X only
  predictive fingerprint F_XY                                         needs labels, optional

Entries that cannot be computed (too few observations, constant feature, irregular
timestamps for lag statistics, ...) are NaN and mean "unavailable", never "zero".
Nothing in F_X reads labels.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

SCHEMA_VERSION = "2.0"
QGRID = np.linspace(0.0, 1.0, 21)                 # quantile sketch: 21 points
ACF_LAGS = (1, 5, 10)
STAT_KEYS = ("mean", "std", "median", "min", "max", "q01", "q10", "q25", "q75", "q90", "q99", "skew", "missing")
TIME_KEYS = ("trend", "acf1", "acf5", "acf10", "volatility", "roc", "season", "abrupt")
RESID_Q = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
_EPS = 1e-12


@dataclass(frozen=True)
class FPConfig:
    irregular_cv: float = 0.5        # dt coefficient of variation above which lag statistics are unavailable
    max_missing_for_time: float = 0.2  # above this missing ratio a column gets no temporal statistics
    min_time_obs: int = 10


def _nan_list(a: np.ndarray) -> list:
    return [None if not math.isfinite(v) else float(v) for v in np.asarray(a, float).ravel()]


def _arr(v: Iterable, shape: tuple | None = None) -> np.ndarray:
    a = np.array([np.nan if x is None else x for x in v], float)
    return a.reshape(shape) if shape else a


def _rank(v: np.ndarray) -> np.ndarray:
    """Average ranks (ties share a rank)."""
    _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    cs = np.cumsum(cnt)
    return (cs - (cnt - 1) / 2.0)[inv]


@dataclass
class FeatureFingerprint:
    """Feature-only fingerprint F_X = (statistics, distributions, temporal)."""
    n: int
    num_idx: tuple[int, ...]
    cat_idx: tuple[int, ...]
    statistics: dict[str, np.ndarray]                 # key -> (n_num,)
    categorical: dict[str, Any]                       # freq: list[dict[int,float]], n_unique, missing
    distributions: dict[str, np.ndarray]              # quantiles (n_num, 21), corr (n_pairs,)
    temporal: dict[str, np.ndarray]                   # key -> (n_num,)
    temporal_meta: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    meta: dict[str, Any] = field(default_factory=dict)

    # -- views ------------------------------------------------------------
    def view_available(self, view: str) -> bool:
        if view == "stat":
            return bool(len(self.cat_idx)) or bool(np.isfinite(self.statistics["median"]).any())
        if view == "dist":
            return bool(np.isfinite(self.distributions["quantiles"]).any()) or bool(np.isfinite(self.distributions["corr"]).any())
        if view == "time":
            return any(bool(np.isfinite(v).any()) for v in self.temporal.values())
        raise ValueError(f"unknown view {view!r}")

    def completeness(self) -> float:
        """Fraction of numeric statistic/temporal cells that could be computed (0..1)."""
        cells = [np.isfinite(self.statistics[k]) for k in ("median", "q10", "q90", "skew")]
        cells += [np.isfinite(self.temporal[k]) for k in TIME_KEYS]
        if not cells or not len(self.num_idx):
            return 1.0 if self.cat_idx else 0.0
        return float(np.mean(np.concatenate(cells)))

    def effective_n(self) -> float:
        """Sample size discounted for serial correlation: n (1 - rho) / (1 + rho), rho = median lag-1 autocorrelation.

        Noise floors in the distance use this, because an autocorrelated window holds less independent
        information than ``n`` suggests. Unavailable autocorrelation means no discount.
        """
        rho = self.temporal.get("acf1")
        r = float(np.nanmedian(rho)) if rho is not None and np.isfinite(rho).any() else 0.0
        r = min(max(r, 0.0), 0.95)
        return max(5.0, self.n * (1 - r) / (1 + r))

    def l1_unit(self, kind: str = "shape") -> tuple[np.ndarray, np.ndarray]:
        """Flat, unit-free Level-1 retrieval vector and its per-entry (scale, weight).

        Per numeric feature: log sigma, standardised quantiles (q01..q99 relative to the median), skew; for
        ``shape_corr``/``full`` also mean |rank correlation| with the other features, lag-1 autocorrelation and
        log volatility; for ``full`` also the top-3 category frequencies of each categorical feature.
        Median location is compared separately (scaled by the pooled sigma), see ``ConceptRepository.level1``.
        """
        if kind not in ("shape", "shape_corr", "full"):
            raise ValueError(f"unknown Level-1 representation {kind!r}")
        st = self.statistics
        k = len(self.num_idx)
        sig = (st["q75"] - st["q25"]) / 1.349
        sig = np.where(np.isfinite(sig) & (sig > _EPS), sig, np.where(st["std"] > _EPS, st["std"], np.nan))
        with np.errstate(invalid="ignore", divide="ignore"):
            cols = [np.log(sig)] + [(st[q] - st["median"]) / sig for q in ("q01", "q10", "q25", "q75", "q90", "q99")] + [st["skew"]]
        scale = [1.0] + [0.5] * 6 + [1.7]
        weight = [2.0] + [1.0] * 6 + [0.5]
        if kind in ("shape_corr", "full"):
            C = np.full((k, k), np.nan)
            pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]
            for p_, (a, b) in enumerate(pairs):
                C[a, b] = C[b, a] = abs(self.distributions["corr"][p_])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mac = np.nanmean(C, axis=1) if k > 1 else np.full(k, np.nan)
                lv = np.log(np.maximum(self.temporal["volatility"], _EPS))
            cols += [mac, self.temporal["acf1"], lv]
            scale += [0.3, 0.3, 1.0]
            weight += [1.0, 1.0, 1.0]
        block = np.stack(cols, axis=1)                    # (k, m), feature-major when flattened
        flat = block.ravel()
        sc = np.tile(np.array(scale), k)
        wt = np.tile(np.array(weight), k)
        if kind == "full" and len(self.cat_idx):
            tops = []
            for f in self.categorical["freq"]:
                top = sorted(f.values(), reverse=True)[:3]
                tops.extend(top + [0.0] * (3 - len(top)))
            flat = np.concatenate([flat, np.array(tops)])
            sc = np.concatenate([sc, np.full(len(tops), 0.15)])
            wt = np.concatenate([wt, np.ones(len(tops))])
        return flat, np.stack([sc, wt])

    def global_vector(self) -> np.ndarray:
        """Level-1 compact signature: (n_num, 4) = median, robust sigma, q10, q90."""
        st = self.statistics
        sig = (st["q75"] - st["q25"]) / 1.349
        sig = np.where(np.isfinite(sig) & (sig > _EPS), sig, st["std"])
        return np.stack([st["median"], sig, st["q10"], st["q90"]], axis=1)

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version, "type": "feature", "n": self.n,
            "num_idx": list(self.num_idx), "cat_idx": list(self.cat_idx),
            "statistics": {k: _nan_list(v) for k, v in self.statistics.items()},
            "categorical": {
                "freq": [{str(c): float(p) for c, p in f.items()} for f in self.categorical["freq"]],
                "n_unique": _nan_list(self.categorical["n_unique"]),
                "missing": _nan_list(self.categorical["missing"])},
            "distributions": {"quantiles": {"shape": list(self.distributions["quantiles"].shape),
                                            "data": _nan_list(self.distributions["quantiles"])},
                              "corr": _nan_list(self.distributions["corr"])},
            "temporal": {k: _nan_list(v) for k, v in self.temporal.items()},
            "temporal_meta": dict(self.temporal_meta), "meta": dict(self.meta)}

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureFingerprint":
        ver = str(d.get("schema_version", "1.0"))
        if not ver.startswith("2."):
            raise ValueError(f"unsupported fingerprint schema {ver!r}; use migrate_v1() for legacy vectors")
        q = d["distributions"]["quantiles"]
        return cls(
            n=int(d["n"]), num_idx=tuple(d["num_idx"]), cat_idx=tuple(d["cat_idx"]),
            statistics={k: _arr(v) for k, v in d["statistics"].items()},
            categorical={"freq": [{int(c): float(p) for c, p in f.items()} for f in d["categorical"]["freq"]],
                         "n_unique": _arr(d["categorical"]["n_unique"]), "missing": _arr(d["categorical"]["missing"])},
            distributions={"quantiles": _arr(q["data"], tuple(q["shape"])), "corr": _arr(d["distributions"]["corr"])},
            temporal={k: _arr(v) for k, v in d["temporal"].items()},
            temporal_meta=dict(d.get("temporal_meta", {})), schema_version=ver, meta=dict(d.get("meta", {})))

    def nbytes(self) -> int:
        arrays = list(self.statistics.values()) + list(self.distributions.values()) + list(self.temporal.values())
        return int(sum(a.nbytes for a in arrays) + 12 * sum(len(f) for f in self.categorical["freq"]))


def _temporal_for_column(v: np.ndarray, tt: np.ndarray | None, cfg: FPConfig) -> dict[str, float]:
    """Temporal statistics of one ordered column. NaN = not computable."""
    out = {k: math.nan for k in TIME_KEYS}
    n = len(v)
    ok = np.isfinite(v)
    if n < cfg.min_time_obs or (1 - ok.mean()) > cfg.max_missing_for_time or ok.sum() < cfg.min_time_obs:
        return out
    pos = np.arange(n, dtype=float) if tt is None else tt
    if not ok.all():
        v = np.interp(pos, pos[ok], v[ok])
    mu, sd = v.mean(), v.std()
    if sd <= _EPS * (abs(mu) + 1):
        return out                                                   # constant series: nothing to say
    dt = np.ones(n - 1) if tt is None else np.diff(tt)
    pdt = dt[dt > 0]
    med_dt = float(np.median(pdt)) if len(pdt) else 1.0
    dt = np.where(dt > 0, dt, med_dt)
    regular = tt is None or (np.std(dt) / (np.mean(dt) + _EPS)) <= cfg.irregular_cv
    if pos.std() > _EPS:
        out["trend"] = float(np.corrcoef(v, pos)[0, 1])
    dz = np.diff(v) / np.sqrt(dt / med_dt)                          # Brownian scaling for irregular gaps
    out["volatility"] = float(dz.std() / (sd * math.sqrt(2)))
    out["roc"] = float(np.mean(np.abs(np.diff(v)) / (dt / med_dt)) / sd)
    if n >= 20:
        mad = 1.4826 * np.median(np.abs(dz - np.median(dz)))
        thr = 3 * mad if mad > _EPS else _EPS
        out["abrupt"] = float(np.mean(np.abs(dz - np.median(dz)) > thr))
    if regular:                                                       # lag statistics need evenly spaced observations
        xc = v - mu
        den = float(xc @ xc)
        for lag in ACF_LAGS:
            if n >= 3 * lag + 10:
                out[f"acf{lag}"] = float(xc[:-lag] @ xc[lag:] / den)
        if n >= 30:
            L = min(n // 3, 40)
            out["season"] = float(max(0.0, max(xc[:-l] @ xc[l:] / den for l in range(2, L + 1))))
    return out


def extract_feature_fp(X: np.ndarray, t: np.ndarray | None = None, categorical: Sequence[int] = (),
                       config: FPConfig | None = None) -> FeatureFingerprint:
    """Build F_X from an (n, d) window. Categorical columns are integer-coded; NaN/inf = missing.

    ``t`` (optional) are timestamps; rows are put in timestamp order (stable) before temporal
    statistics are computed. Uses no labels.
    """
    cfg = config or FPConfig()
    X = np.asarray(X, float)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("X must be a non-empty 2-D array")
    n, d = X.shape
    cat = tuple(sorted({int(c) for c in categorical}))
    if any(c < 0 or c >= d for c in cat):
        raise ValueError("categorical index out of range")
    num = tuple(j for j in range(d) if j not in cat)
    X = np.where(np.isfinite(X), X, np.nan)
    tt = None
    if t is not None:
        t = np.asarray(t, float)
        if t.shape != (n,) or not np.isfinite(t).all():
            raise ValueError("t must be finite and have one entry per row")
        order = np.argsort(t, kind="stable")
        X, tt = X[order], t[order]

    k = len(num)
    stats = {key: np.full(k, np.nan) for key in STAT_KEYS}
    quant = np.full((k, len(QGRID)), np.nan)
    temporal = {key: np.full(k, np.nan) for key in TIME_KEYS}
    for i, j in enumerate(num):
        col = X[:, j]
        okm = np.isfinite(col)
        stats["missing"][i] = 1.0 - okm.mean()
        v = col[okm]
        if v.size:
            q = np.quantile(v, [0, .01, .1, .25, .5, .75, .9, .99, 1])
            stats["min"][i], stats["q01"][i], stats["q10"][i], stats["q25"][i], stats["median"][i] = q[:5]
            stats["q75"][i], stats["q90"][i], stats["q99"][i], stats["max"][i] = q[5:]
            stats["mean"][i], stats["std"][i] = v.mean(), v.std()
            if v.size >= 3 and stats["std"][i] > _EPS * (abs(stats["mean"][i]) + 1):
                stats["skew"][i] = float(np.mean(((v - stats["mean"][i]) / stats["std"][i]) ** 3))
            if v.size >= 2:
                quant[i] = np.quantile(v, QGRID)
        for key, val in _temporal_for_column(col, tt, cfg).items():
            temporal[key][i] = val

    # Spearman correlations (rank based, robust to outliers); constant columns -> NaN
    pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]
    corr = np.full(len(pairs), np.nan)
    if pairs and n >= 5:
        R = np.full((n, k), np.nan)
        for i, j in enumerate(num):
            col = X[:, j]
            ok = np.isfinite(col)
            if ok.any() and np.ptp(col[ok]) > _EPS:
                filled = np.where(ok, col, np.median(col[ok]))
                R[:, i] = _rank(filled)
        for p, (a, b) in enumerate(pairs):
            if np.isfinite(R[:, a]).all() and np.isfinite(R[:, b]).all():
                corr[p] = float(np.corrcoef(R[:, a], R[:, b])[0, 1])

    freqs, nuniq, miss = [], np.zeros(len(cat)), np.zeros(len(cat))
    for i, j in enumerate(cat):
        col = X[:, j]
        okm = np.isfinite(col)
        miss[i] = 1.0 - okm.mean()
        codes, cnt = np.unique(col[okm].astype(int), return_counts=True)
        freqs.append({int(c): float(m) / max(1, cnt.sum()) for c, m in zip(codes, cnt)})
        nuniq[i] = len(codes)
    tmeta = {"n_obs": int(n), "timestamps": t is not None}
    return FeatureFingerprint(n, num, cat, stats, {"freq": freqs, "n_unique": nuniq, "missing": miss},
                              {"quantiles": quant, "corr": corr}, temporal, tmeta)


def _nan_avg(a: np.ndarray, b: np.ndarray, wa: float, wb: float) -> np.ndarray:
    out = np.where(np.isfinite(a), a, b).astype(float)
    both = np.isfinite(a) & np.isfinite(b)
    out[both] = (wa * a[both] + wb * b[both]) / (wa + wb)
    return out


def merge_feature_fp(a: FeatureFingerprint, b: FeatureFingerprint, wa: float, wb: float, n_cap: int = 2000) -> FeatureFingerprint:
    """Weighted merge (a: stored concept, b: new window). Quantile sketches are averaged in
    quantile space, which is exact for location-scale families and an approximation otherwise."""
    if (a.num_idx, a.cat_idx) != (b.num_idx, b.cat_idx):
        raise ValueError("incompatible fingerprints")
    stats = {k: _nan_avg(a.statistics[k], b.statistics[k], wa, wb) for k in STAT_KEYS}
    stats["min"] = np.fmin(a.statistics["min"], b.statistics["min"])
    stats["max"] = np.fmax(a.statistics["max"], b.statistics["max"])
    freqs = []
    for fa, fb in zip(a.categorical["freq"], b.categorical["freq"]):
        keys = set(fa) | set(fb)
        freqs.append({c: (wa * fa.get(c, 0.0) + wb * fb.get(c, 0.0)) / (wa + wb) for c in keys})
    cat = {"freq": freqs, "n_unique": np.array([len(f) for f in freqs], float),
           "missing": _nan_avg(a.categorical["missing"], b.categorical["missing"], wa, wb)}
    dist = {"quantiles": _nan_avg(a.distributions["quantiles"], b.distributions["quantiles"], wa, wb),
            "corr": _nan_avg(a.distributions["corr"], b.distributions["corr"], wa, wb)}
    temporal = {k: _nan_avg(a.temporal[k], b.temporal[k], wa, wb) for k in TIME_KEYS}
    return FeatureFingerprint(min(a.n + b.n, n_cap), a.num_idx, a.cat_idx, stats, cat, dist, temporal,
                              dict(a.temporal_meta), meta=dict(a.meta))


# ---------------------------------------------------------------------------------------------
# predictive fingerprint (label aware, optional)
# ---------------------------------------------------------------------------------------------
@dataclass
class PredictiveFingerprint:
    """How one model behaves on one stretch of labelled data (F_XY). Never built from X alone."""
    n_labeled: int
    error_rate: float
    calibration_error: float | None            # expected calibration error, None without probabilities
    residual_quantiles: np.ndarray | None      # quantiles of 1 - P(true class), None without probabilities
    classes: tuple[int, ...]
    class_recall: np.ndarray                   # per class, NaN when the class was not observed
    confusion: np.ndarray                      # row-normalised, NaN rows for unobserved classes
    schema_version: str = SCHEMA_VERSION

    @property
    def accuracy(self) -> float:
        return 1.0 - self.error_rate

    def to_dict(self) -> dict:
        return {"schema_version": self.schema_version, "type": "predictive", "n_labeled": self.n_labeled,
                "error_rate": self.error_rate, "calibration_error": self.calibration_error,
                "residual_quantiles": None if self.residual_quantiles is None else _nan_list(self.residual_quantiles),
                "classes": list(self.classes), "class_recall": _nan_list(self.class_recall),
                "confusion": _nan_list(self.confusion)}

    @classmethod
    def from_dict(cls, d: dict) -> "PredictiveFingerprint":
        k = len(d["classes"])
        rq = d["residual_quantiles"]
        return cls(int(d["n_labeled"]), float(d["error_rate"]), d["calibration_error"],
                   None if rq is None else _arr(rq), tuple(d["classes"]), _arr(d["class_recall"]),
                   _arr(d["confusion"], (k, k)), str(d.get("schema_version", SCHEMA_VERSION)))


def predictive_fp(y_true: Sequence, y_pred: Sequence, proba: Sequence | None = None,
                  classes: Sequence[int] = (0, 1), bins: int = 10) -> PredictiveFingerprint | None:
    """Build F_XY from labelled predictions. Rows with a missing label (None/NaN) are *ignored*,
    never counted as correct. Returns None when no labelled row exists. ``proba`` is P(class 1)
    for binary tasks or an (n, K) matrix."""
    yt = np.array([np.nan if v is None else v for v in y_true], float)
    keep = np.isfinite(yt)
    if not keep.any():
        return None
    yt = yt[keep].astype(int)
    yp = np.asarray(y_pred)[keep].astype(int)
    cls = tuple(int(c) for c in classes)
    K = len(cls)
    idx = {c: i for i, c in enumerate(cls)}
    err = float(np.mean(yt != yp))
    conf = np.full((K, K), np.nan)
    recall = np.full(K, np.nan)
    for c in cls:
        m = yt == c
        if m.any():
            row = np.array([np.mean(yp[m] == c2) for c2 in cls])
            conf[idx[c]] = row
            recall[idx[c]] = row[idx[c]]
    ece, rq = None, None
    if proba is not None:
        pr = np.asarray(proba, float)[keep]
        if pr.ndim == 1:
            pr = np.stack([1 - pr, pr], axis=1)
        ptrue = pr[np.arange(len(yt)), [idx.get(int(c), 0) for c in yt]]
        rq = np.quantile(1.0 - ptrue, RESID_Q)
        confidence = pr.max(axis=1)
        correct = (pr.argmax(axis=1) == [idx.get(int(c), 0) for c in yt]).astype(float)
        edges = np.linspace(0.5 if K == 2 else 1.0 / K, 1.0, bins + 1)
        which = np.clip(np.digitize(confidence, edges[1:-1]), 0, bins - 1)
        ece = float(sum(abs(correct[which == b].mean() - confidence[which == b].mean()) * (which == b).mean()
                        for b in range(bins) if (which == b).any()))
    return PredictiveFingerprint(int(keep.sum()), err, ece, rq, cls, recall, conf)


def merge_predictive_fp(a: PredictiveFingerprint, b: PredictiveFingerprint, wa: float, wb: float,
                        n_cap: int = 2000) -> PredictiveFingerprint:
    if a.classes != b.classes:
        raise ValueError("incompatible predictive fingerprints")
    ece = a.calibration_error if b.calibration_error is None else b.calibration_error if a.calibration_error is None \
        else (wa * a.calibration_error + wb * b.calibration_error) / (wa + wb)
    rq = a.residual_quantiles if b.residual_quantiles is None else b.residual_quantiles if a.residual_quantiles is None \
        else _nan_avg(a.residual_quantiles, b.residual_quantiles, wa, wb)
    return PredictiveFingerprint(min(a.n_labeled + b.n_labeled, n_cap), (wa * a.error_rate + wb * b.error_rate) / (wa + wb),
                                 ece, rq, a.classes, _nan_avg(a.class_recall, b.class_recall, wa, wb),
                                 _nan_avg(a.confusion, b.confusion, wa, wb))


class PredictiveAccumulator:
    """Collects (prediction, probability) as they are made and attaches labels when they arrive.

    Supports delayed labels: ``record`` stores a prediction under a sample id; ``reveal`` attaches
    the label later. Unlabelled predictions are excluded from the fingerprint (not scored as correct).
    """

    def __init__(self, classes: Sequence[int] = (0, 1), maxlen: int | None = None):
        self.classes = tuple(classes)
        self.maxlen = maxlen
        self._rows: dict[Any, list] = {}               # sample id -> [y_true|None, y_pred, proba]
        self._seq = 0

    def record(self, sample_id: Any, y_pred: int, proba: float | None = None, y_true: int | None = None) -> None:
        self._rows[sample_id] = [y_true, int(y_pred), proba]
        if self.maxlen and len(self._rows) > self.maxlen:
            self._rows.pop(next(iter(self._rows)))

    def reveal(self, sample_id: Any, y_true: int) -> bool:
        row = self._rows.get(sample_id)
        if row is None:
            return False
        row[0] = int(y_true)
        return True

    def add(self, y_true: int, y_pred: int, proba: float | None = None) -> None:
        """Record an already-labelled prediction under an auto-generated id."""
        self._seq += 1
        self.record(("auto", self._seq), y_pred, proba, y_true)

    @property
    def n_labeled(self) -> int:
        return sum(1 for r in self._rows.values() if r[0] is not None)

    @property
    def n_unlabeled(self) -> int:
        return len(self._rows) - self.n_labeled

    def clear(self) -> None:
        self._rows.clear()

    def fingerprint(self) -> PredictiveFingerprint | None:
        rows = list(self._rows.values())
        if not rows:
            return None
        have_p = all(r[2] is not None for r in rows)
        return predictive_fp([r[0] for r in rows], [r[1] for r in rows],
                             [r[2] for r in rows] if have_p else None, self.classes)


def migrate_v1(vec: Sequence[float], n: int, n_features: int = 4) -> FeatureFingerprint:
    """Migrate a legacy 17-number fingerprint (raw, not z-scored) to schema 2.0.

    Only the label-free entries (per-feature mean and std) survive. The class gaps,
    feature/label correlations and label rate are dropped on purpose, and every other entry is
    left NaN (unavailable), so a migrated concept reports low completeness instead of pretending
    to know its distribution.
    """
    v = np.asarray(vec, float)
    if v.shape != (4 * n_features + 1,):
        raise ValueError(f"expected {4 * n_features + 1} legacy entries, got {v.shape}")
    k = n_features
    stats = {key: np.full(k, np.nan) for key in STAT_KEYS}
    stats["mean"], stats["std"] = v[:k].copy(), v[k:2 * k].copy()
    stats["median"] = v[:k].copy()      # assumes symmetry: flagged below
    return FeatureFingerprint(
        n, tuple(range(k)), (), stats, {"freq": [], "n_unique": np.zeros(0), "missing": np.zeros(0)},
        {"quantiles": np.full((k, len(QGRID)), np.nan), "corr": np.full(k * (k - 1) // 2, np.nan)},
        {key: np.full(k, np.nan) for key in TIME_KEYS}, {"n_obs": n, "timestamps": False},
        meta={"migrated_from": "1.0", "dropped_label_dims": ["class_gap", "corr_xy", "label_rate"],
              "median_is_mean_proxy": True})
