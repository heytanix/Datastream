"""Controlled window generators with known concept identity, for calibration and retrieval experiments."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..mvfp import FeatureFingerprint, extract_feature_fp


@dataclass(frozen=True)
class Spec:
    """A synthetic concept: per-feature location/scale, optional heavy tails, AR(1) dependence, cross-correlation."""
    name: str
    mu: tuple
    sigma: tuple
    rho: tuple | float = 0.0          # AR(1) coefficient, scalar or per feature
    corr: float = 0.0                 # common cross-feature correlation (equicorrelated)
    tail: float = 0.0                 # fraction of observations replaced by x4 outliers (shape change, same middle quantiles)
    bimodal: float = 0.0              # mixture separation in sigma units (0 = unimodal)
    noise: float = 0.0                # additive measurement noise sd

    @property
    def d(self) -> int:
        return len(self.mu)


def sample(spec: Spec, n: int, rng: np.random.Generator) -> np.ndarray:
    d = spec.d
    rho = np.broadcast_to(np.asarray(spec.rho, float), (d,))
    base = rng.normal(0, 1, (n, d))
    if spec.corr:
        L = np.linalg.cholesky((1 - spec.corr) * np.eye(d) + spec.corr * np.ones((d, d)))
        base = base @ L.T
    Z = np.empty((n, d))
    Z[0] = base[0]
    for i in range(1, n):
        Z[i] = rho * Z[i - 1] + np.sqrt(1 - rho ** 2) * base[i]
    if spec.bimodal:
        Z = Z * 0.6 + rng.choice([-1, 1], (n, d)) * spec.bimodal * 0.5
    if spec.tail:
        m = rng.random((n, d)) < spec.tail
        Z = np.where(m, Z * 4, Z)
    X = np.asarray(spec.mu) + np.asarray(spec.sigma) * Z
    if spec.noise:
        X = X + rng.normal(0, spec.noise, X.shape)
    return X


def window(spec: Spec, n: int, rng: np.random.Generator, **fp_kw) -> FeatureFingerprint:
    return extract_feature_fp(sample(spec, n, rng), t=np.arange(n), **fp_kw)


def base_spec(d: int = 4, rho: float = 0.0, noise: float = 0.0, name: str = "A") -> Spec:
    return Spec(name, tuple(0.0 for _ in range(d)), tuple(1.0 for _ in range(d)), rho, noise=noise)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score_pos > score_neg) (Mann-Whitney), ties count half. Use with scores where larger = 'different'."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="stable")
    ranks = np.empty(len(allv))
    # average ranks for ties
    sv = allv[order]
    i = 0
    r = np.empty(len(allv))
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    ranks[order] = r
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@dataclass
class Library:
    """A set of concepts with known identity; ``families`` documents why each pair is hard."""
    specs: dict[str, Spec]
    families: dict[str, str] = field(default_factory=dict)


def lookalike_library(d: int = 4) -> Library:
    """Concepts that agree on median/spread/q10/q90 but differ in shape, correlation or temporal behaviour."""
    z = tuple(0.0 for _ in range(d))
    o = tuple(1.0 for _ in range(d))
    specs = {
        "base": Spec("base", z, o),
        "heavy_tail": Spec("heavy_tail", z, o, tail=0.06),                    # same middle quantiles, heavy tails
        "bimodal": Spec("bimodal", z, o, bimodal=2.2),                        # different shape
        "correlated": Spec("correlated", z, o, corr=0.7),                     # same marginals, different dependence
        "autocorr": Spec("autocorr", z, o, rho=0.9),                          # same marginals, different temporal
        "shifted": Spec("shifted", tuple(0.35 for _ in range(d)), o),         # small location shift
        "wider": Spec("wider", z, tuple(1.4 for _ in range(d))),
        "far": Spec("far", tuple(3.0 for _ in range(d)), o),
    }
    fam = {"heavy_tail": "shape (tails)", "bimodal": "shape", "correlated": "dependence", "autocorr": "temporal",
           "shifted": "small location shift", "wider": "scale", "far": "easy"}
    return Library(specs, fam)
