"""Mathematically valid concept memory: mergeable summaries from which a concept fingerprint is *recomputed*.

Averaging fingerprints entry by entry is wrong for several statistics (measured in results/research/audit.json:
merging N(0,1) with N(5,1) gave std 0.99 instead of 2.36 and q90 2.48 instead of 5.24). Instead a concept keeps

  moments      count, mean, M2, M3 per feature   merged EXACTLY with the Chan/Pebay parallel formulas -> mean, std, skew
  sketches     a t-digest-style quantile sketch  per feature; merging = union + compress -> all quantiles, min, max
  cat counts   exact category counts             -> category frequencies
  history      per-window temporal and correlation vectors (bounded); the concept's value is the median over windows.
               Autocorrelation, trend, periodicity and rank correlation are not mergeable across windows (they need the
               chronology / the joint ranks), so they are NOT averaged; a robust location of the window values is used.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .mvfp import QGRID, STAT_KEYS, TIME_KEYS, FeatureFingerprint

_EPS = 1e-12


def _nl(a) -> list:
    return [None if not math.isfinite(v) else float(v) for v in np.asarray(a, float).ravel()]


class QuantileSketch:
    """Mergeable quantile summary (Dunning's merging t-digest, scale function k1).

    At most ~compression centroids, finer near the tails. Exact min and max are kept. ``merge`` is associative up to
    compression error, so a concept can absorb any number of windows in bounded memory.
    """

    def __init__(self, compression: int = 50):
        self.delta = float(compression)
        self.means = np.empty(0)
        self.weights = np.empty(0)
        self.n = 0.0
        self.min = math.inf
        self.max = -math.inf

    @classmethod
    def from_values(cls, v: np.ndarray, compression: int = 50) -> "QuantileSketch":
        s = cls(compression)
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        if v.size:
            s.means, s.weights = np.sort(v), np.ones(v.size)
            s.n, s.min, s.max = float(v.size), float(v.min()), float(v.max())
            s._compress()
        return s

    @classmethod
    def from_quantiles(cls, q: np.ndarray, n: float, compression: int = 50) -> "QuantileSketch":
        """Approximate sketch rebuilt from a stored 21-point quantile summary (used when raw data is unavailable)."""
        s = cls(compression)
        q = np.asarray(q, float)
        if not np.isfinite(q).all() or n <= 0:
            return s
        s.means = 0.5 * (q[:-1] + q[1:])
        s.weights = np.full(len(s.means), n / len(s.means))
        s.n, s.min, s.max = float(n), float(q[0]), float(q[-1])
        s._compress()
        return s

    def _compress(self) -> None:
        if len(self.means) <= 1:
            return
        order = np.argsort(self.means, kind="stable")
        m, w = self.means[order].tolist(), self.weights[order].tolist()
        N = float(sum(w))
        c = self.delta / (2 * math.pi)
        asin = math.asin
        k_of = lambda q: c * asin(2.0 * (0.0 if q < 0.0 else 1.0 if q > 1.0 else q) - 1.0)
        om, ow = [m[0]], [w[0]]
        left, k_left = 0.0, k_of(0.0)
        for i in range(1, len(m)):
            prop = ow[-1] + w[i]
            if k_of((left + prop) / N) - k_left <= 1.0:
                om[-1] = (om[-1] * ow[-1] + m[i] * w[i]) / prop
                ow[-1] = prop
            else:
                left += ow[-1]
                k_left = k_of(left / N)
                om.append(m[i])
                ow.append(w[i])
        self.means, self.weights = np.array(om), np.array(ow)

    def merge(self, other: "QuantileSketch") -> "QuantileSketch":
        if other.n == 0:
            return self
        self.means = np.concatenate([self.means, other.means])
        self.weights = np.concatenate([self.weights, other.weights])
        self.n += other.n
        self.min, self.max = min(self.min, other.min), max(self.max, other.max)
        self._compress()
        return self

    def quantiles(self, ps: np.ndarray) -> np.ndarray:
        ps = np.asarray(ps, float)
        if self.n == 0:
            return np.full(ps.shape, np.nan)
        cw = np.cumsum(self.weights) - self.weights / 2
        xs = np.concatenate([[0.0], cw, [self.n]])
        ys = np.concatenate([[self.min], self.means, [self.max]])
        return np.interp(ps * self.n, xs, ys)

    def to_dict(self) -> dict:
        return {"delta": self.delta, "n": self.n, "min": None if not math.isfinite(self.min) else self.min,
                "max": None if not math.isfinite(self.max) else self.max, "means": _nl(self.means), "weights": _nl(self.weights)}

    @classmethod
    def from_dict(cls, d: dict) -> "QuantileSketch":
        s = cls(int(d["delta"]))
        s.means, s.weights = np.array(d["means"], float), np.array(d["weights"], float)
        s.n = float(d["n"])
        s.min = math.inf if d["min"] is None else float(d["min"])
        s.max = -math.inf if d["max"] is None else float(d["max"])
        return s

    def nbytes(self) -> int:
        return int(self.means.nbytes + self.weights.nbytes + 24)


@dataclass
class Moments:
    """Exactly mergeable count, mean, M2, M3 (Chan et al.; Pebay) per feature, plus row and missing counts."""
    n: np.ndarray
    mean: np.ndarray
    m2: np.ndarray
    m3: np.ndarray
    rows: float = 0.0

    @classmethod
    def from_window(cls, X: np.ndarray) -> "Moments":
        X = np.asarray(X, float)
        k = X.shape[1]
        n, mean, m2, m3 = np.zeros(k), np.zeros(k), np.zeros(k), np.zeros(k)
        for j in range(k):
            v = X[:, j]
            v = v[np.isfinite(v)]
            if v.size:
                n[j], mean[j] = v.size, v.mean()
                d = v - mean[j]
                m2[j], m3[j] = np.sum(d ** 2), np.sum(d ** 3)
        return cls(n, mean, m2, m3, float(len(X)))

    @classmethod
    def from_fp(cls, fp: FeatureFingerprint) -> "Moments":
        """Approximate moments from a stored fingerprint (population std, skew); exact when the fingerprint came from raw data."""
        st = fp.statistics
        miss = np.nan_to_num(st["missing"], nan=0.0)
        n = fp.n * (1 - miss)
        std, skew = np.nan_to_num(st["std"]), np.nan_to_num(st["skew"])
        return cls(n, np.nan_to_num(st["mean"]), n * std ** 2, n * skew * std ** 3, float(fp.n))

    def merge(self, o: "Moments") -> "Moments":
        n = self.n + o.n
        with np.errstate(invalid="ignore", divide="ignore"):
            delta = o.mean - self.mean
            mean = np.where(n > 0, self.mean + delta * o.n / n, 0.0)
            m2 = self.m2 + o.m2 + delta ** 2 * self.n * o.n / n
            m3 = (self.m3 + o.m3 + delta ** 3 * self.n * o.n * (self.n - o.n) / n ** 2
                  + 3 * delta * (self.n * o.m2 - o.n * self.m2) / n)
        ok = n > 0
        self.mean, self.m2, self.m3 = np.where(ok, mean, 0.0), np.where(ok, m2, 0.0), np.where(ok, m3, 0.0)
        self.n, self.rows = n, self.rows + o.rows
        return self

    def to_dict(self) -> dict:
        return {"n": _nl(self.n), "mean": _nl(self.mean), "m2": _nl(self.m2), "m3": _nl(self.m3), "rows": self.rows}

    @classmethod
    def from_dict(cls, d: dict) -> "Moments":
        a = lambda k: np.array([0.0 if v is None else v for v in d[k]], float)
        return cls(a("n"), a("mean"), a("m2"), a("m3"), float(d["rows"]))


@dataclass
class ConceptSummaries:
    """Everything needed to recompute a concept's fingerprint after absorbing any number of windows."""
    num_idx: tuple
    cat_idx: tuple
    moments: Moments
    sketches: list
    cat_counts: list                                   # per categorical feature: dict code -> count
    cat_rows: np.ndarray                               # rows seen per categorical feature (incl. missing)
    history: deque = field(default_factory=lambda: deque(maxlen=20))     # (temporal (k, T), corr (pairs,)) per window
    n_windows: int = 0
    max_history: int = 20

    @classmethod
    def from_window(cls, X: np.ndarray, fp: FeatureFingerprint, compression: int = 50, max_history: int = 20) -> "ConceptSummaries":
        """Exact summaries of one raw window ``X`` (the same rows ``fp`` was computed from)."""
        X = np.asarray(X, float)
        mom = Moments.from_window(X[:, list(fp.num_idx)]) if fp.num_idx else Moments(*(np.zeros(0),) * 4)
        sk = [QuantileSketch.from_values(X[:, j], compression) for j in fp.num_idx]
        counts, rows = [], np.zeros(len(fp.cat_idx))
        for i, j in enumerate(fp.cat_idx):
            col = X[:, j]
            ok = np.isfinite(col)
            codes, cnt = np.unique(col[ok].astype(int), return_counts=True)
            counts.append({int(c): float(m) for c, m in zip(codes, cnt)})
            rows[i] = len(col)
        s = cls(fp.num_idx, fp.cat_idx, mom, sk, counts, rows, deque(maxlen=max_history), 0, max_history)
        s._push_history(fp)
        return s

    @classmethod
    def from_fp(cls, fp: FeatureFingerprint, compression: int = 50, max_history: int = 20) -> "ConceptSummaries":
        """APPROXIMATE summaries rebuilt from a stored fingerprint (loading schema-2.0 concepts, or no raw data)."""
        sk = [QuantileSketch.from_quantiles(fp.distributions["quantiles"][i], fp.n * (1 - float(np.nan_to_num(fp.statistics["missing"][i]))), compression)
              for i in range(len(fp.num_idx))]
        counts = [{c: p * fp.n for c, p in f.items()} for f in fp.categorical["freq"]]
        rows = np.full(len(fp.cat_idx), float(fp.n))
        s = cls(fp.num_idx, fp.cat_idx, Moments.from_fp(fp), sk, counts, rows, deque(maxlen=max_history), 0, max_history)
        s._push_history(fp)
        return s

    def _push_history(self, fp: FeatureFingerprint) -> None:
        self.history.append((np.stack([fp.temporal[k] for k in TIME_KEYS], axis=1), np.asarray(fp.distributions["corr"], float)))
        self.n_windows += 1

    def merge_window(self, other: "ConceptSummaries") -> None:
        if (self.num_idx, self.cat_idx) != (other.num_idx, other.cat_idx):
            raise ValueError("incompatible summaries")
        self.moments.merge(other.moments)
        for a, b in zip(self.sketches, other.sketches):
            a.merge(b)
        for a, b in zip(self.cat_counts, other.cat_counts):
            for c, m in b.items():
                a[c] = a.get(c, 0.0) + m
        self.cat_rows = self.cat_rows + other.cat_rows
        for item in other.history:
            self.history.append(item)
        self.n_windows += other.n_windows

    def compose(self, n_cap: int = 2000, template: FeatureFingerprint | None = None) -> FeatureFingerprint:
        """Recompute the concept fingerprint from the summaries (nothing is averaged)."""
        k = len(self.num_idx)
        m = self.moments
        stats = {key: np.full(k, np.nan) for key in STAT_KEYS}
        with np.errstate(invalid="ignore", divide="ignore"):
            valid = m.n > 0
            std = np.where(valid, np.sqrt(m.m2 / np.where(valid, m.n, 1)), np.nan)
            stats["mean"], stats["std"] = np.where(valid, m.mean, np.nan), std
            stats["skew"] = np.where(valid & (std > _EPS * (np.abs(m.mean) + 1)), np.sqrt(m.n) * m.m3 / np.maximum(m.m2, _EPS) ** 1.5, np.nan)
            stats["missing"] = np.where(m.rows > 0, 1 - m.n / np.maximum(m.rows, 1), np.nan)
        quant = np.full((k, len(QGRID)), np.nan)
        probs = {"q01": .01, "q10": .1, "q25": .25, "median": .5, "q75": .75, "q90": .9, "q99": .99}
        for i, sk in enumerate(self.sketches):
            if sk.n == 0:
                continue
            qs = sk.quantiles(np.concatenate([QGRID, list(probs.values())]))
            quant[i] = qs[:len(QGRID)]
            for key, v in zip(probs, qs[len(QGRID):]):
                stats[key][i] = float(v)
            stats["min"][i], stats["max"][i] = sk.min, sk.max
        temporal = {key: np.full(k, np.nan) for key in TIME_KEYS}
        corr = np.full(k * (k - 1) // 2, np.nan)
        if self.history:
            T = np.stack([h[0] for h in self.history])           # (W, k, T)
            C = np.stack([h[1] for h in self.history])           # (W, pairs)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slice: statistic unavailable
                med = np.nanmedian(T, axis=0)
                corr = np.nanmedian(C, axis=0) if C.shape[1] else corr
            for t_i, key in enumerate(TIME_KEYS):
                temporal[key] = med[:, t_i]
        freqs, nuniq, miss = [], np.zeros(len(self.cat_idx)), np.zeros(len(self.cat_idx))
        for i, c in enumerate(self.cat_counts):
            tot = sum(c.values())
            freqs.append({code: v / tot for code, v in c.items()} if tot else {})
            nuniq[i] = len(c)
            miss[i] = 1 - tot / max(self.cat_rows[i], 1)
        n = int(min(m.rows if k else (self.cat_rows.max() if len(self.cat_rows) else 0), n_cap))
        return FeatureFingerprint(max(n, 1), self.num_idx, self.cat_idx, stats, {"freq": freqs, "n_unique": nuniq, "missing": miss},
                                  {"quantiles": quant, "corr": corr}, temporal, {"n_obs": n, "timestamps": False, "windows": self.n_windows},
                                  meta={"composed_from": "mergeable summaries", "windows": self.n_windows})

    def nbytes(self) -> int:
        h = sum(t.nbytes + c.nbytes for t, c in self.history)
        return int(sum(s.nbytes() for s in self.sketches) + 4 * self.moments.mean.nbytes + h + 16 * sum(len(c) for c in self.cat_counts))

    def to_dict(self) -> dict:
        return {"num_idx": list(self.num_idx), "cat_idx": list(self.cat_idx), "moments": self.moments.to_dict(),
                "sketches": [s.to_dict() for s in self.sketches],
                "cat_counts": [{str(c): v for c, v in d.items()} for d in self.cat_counts], "cat_rows": _nl(self.cat_rows),
                "history": [{"temporal": _nl(t), "shape": list(t.shape), "corr": _nl(c)} for t, c in self.history],
                "n_windows": self.n_windows, "max_history": self.max_history}

    @classmethod
    def from_dict(cls, d: dict) -> "ConceptSummaries":
        hist = deque(maxlen=int(d["max_history"]))
        for h in d["history"]:
            hist.append((np.array([np.nan if v is None else v for v in h["temporal"]], float).reshape(h["shape"]),
                         np.array([np.nan if v is None else v for v in h["corr"]], float)))
        return cls(tuple(d["num_idx"]), tuple(d["cat_idx"]), Moments.from_dict(d["moments"]), [QuantileSketch.from_dict(s) for s in d["sketches"]],
                   [{int(c): float(v) for c, v in x.items()} for x in d["cat_counts"]], np.array([0.0 if v is None else v for v in d["cat_rows"]], float),
                   hist, int(d["n_windows"]), int(d["max_history"]))
