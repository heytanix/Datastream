"""Synthetic benchmark scenarios with ground-truth concept labels.

Each scenario is a ``ScenarioData``: arrays X, y, a per-sample ``truth`` concept name, and the
true change points. ``truth`` is the concept of the *segment* a sample belongs to, even inside a
gradual/incremental transition, so retrieval metrics compare against the target concept.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .research.gen import Spec, sample
from .stream import CONCEPTS, Concept

RECURRING = "A B C A B D C A D B A".split()


@dataclass
class ScenarioData:
    name: str
    X: np.ndarray
    y: np.ndarray
    truth: list
    change_points: list[tuple[int, str]]
    has_recurrence: bool = True
    notes: str = ""

    def __len__(self) -> int:
        return len(self.y)

    def __iter__(self):
        for t in range(len(self.y)):
            yield t, self.X[t], int(self.y[t]), self.truth[t]


def _gen(c: Concept, n: int, rng, phi: float = 0.0, meas_noise: float = 0.0, label_noise: float = 0.04, mu_shift=None):
    mu = np.array(c.mu, float) + (0 if mu_shift is None else mu_shift)
    sg, w = np.array(c.sigma, float), np.array(c.w, float)
    if phi == 0.0:
        Z = rng.normal(0, 1, (n, 4))
    else:                                              # AR(1) with unit marginal variance
        Z = np.empty((n, 4))
        Z[0] = rng.normal(0, 1, 4)
        e = rng.normal(0, 1, (n, 4)) * np.sqrt(1 - phi ** 2)
        for i in range(1, n):
            Z[i] = phi * Z[i - 1] + e[i]
    X = mu + sg * Z
    y = ((X - mu) @ w > 0).astype(int)
    y = np.where(rng.random(n) < label_noise, 1 - y, y)
    if meas_noise:
        X = X + rng.normal(0, meas_noise, X.shape)
    return X, y


def _segments(name, schedule, seg_len, seed, concepts=None, transition=0, mode="abrupt", **kw) -> ScenarioData:
    rng = np.random.default_rng(seed)
    concepts = concepts or CONCEPTS
    Xs, ys, truth, cps = [], [], [], []
    prev = None
    for k, cname in enumerate(schedule):
        c = concepts[cname]
        phi = getattr(c, "phi", 0.0)
        jitter = None if kw.get("mu_jitter") is None else rng.normal(0, kw["mu_jitter"], 4)
        X, y = _gen(c, seg_len, rng, phi, kw.get("meas_noise", 0.0), kw.get("label_noise", 0.04), jitter)
        if k > 0 and transition and mode != "abrupt":
            L = min(transition, seg_len)
            Xp, yp = _gen(concepts[prev], L, rng, getattr(concepts[prev], "phi", 0.0), kw.get("meas_noise", 0.0), kw.get("label_noise", 0.04))
            if mode == "gradual":                       # each sample comes from old or new concept, p(new) ramps up
                pick_new = rng.random(L) < (np.arange(L) + 1) / (L + 1)
                X[:L] = np.where(pick_new[:, None], X[:L], Xp)
                y[:L] = np.where(pick_new, y[:L], yp)
            else:                                       # incremental: parameters interpolate from old to new
                a = (np.arange(L) / L)[:, None]
                cn, cp = concepts[cname], concepts[prev]
                mu = (1 - a) * np.array(cp.mu) + a * np.array(cn.mu)
                sg = (1 - a) * np.array(cp.sigma) + a * np.array(cn.sigma)
                w = (1 - a) * np.array(cp.w) + a * np.array(cn.w)
                Xi = mu + sg * rng.normal(0, 1, (L, 4))
                yi = (np.sum((Xi - mu) * w, axis=1) > 0).astype(int)
                X[:L], y[:L] = Xi, np.where(rng.random(L) < kw.get("label_noise", 0.04), 1 - yi, yi)
        Xs.append(X), ys.append(y)
        truth += [cname] * seg_len
        cps.append((k * seg_len, cname))
        prev = cname
    return ScenarioData(name, np.vstack(Xs), np.concatenate(ys), truth, cps)


@dataclass(frozen=True)
class _P(Concept):
    phi: float = 0.0


def abrupt_recurring(seed=7, seg_len=1500):
    return _segments("abrupt_recurring", RECURRING, seg_len, seed)


def noisy_recurring(seed=7, seg_len=1500):
    return _segments("noisy_recurring", RECURRING, seg_len, seed, label_noise=0.12, meas_noise=0.5, mu_jitter=0.3)


def gradual(seed=7, seg_len=1500):
    return _segments("gradual", RECURRING, seg_len, seed, transition=300, mode="gradual")


def incremental(seed=7, seg_len=1500):
    return _segments("incremental", RECURRING, seg_len, seed, transition=400, mode="incremental")


def similar_x_different_y(seed=7, seg_len=1500):
    """A and A' have identical p(x) and opposite labelling; B differs in p(x)."""
    a = CONCEPTS["A"]
    cs = {"A": a, "Ap": Concept("Ap", a.mu, a.sigma, tuple(-v for v in a.w)), "B": CONCEPTS["B"]}
    s = _segments("similar_x_diff_y", "A Ap A Ap B A Ap A Ap B".split(), seg_len, seed, concepts=cs)
    s.notes = "A and Ap are indistinguishable from X alone"
    return s


def temporal_only(seed=7, seg_len=1500):
    """Same marginal and same labelling rule; only the temporal behaviour differs (iid vs AR(1) phi=0.9)."""
    a = CONCEPTS["A"]
    cs = {"P": _P("P", a.mu, a.sigma, a.w, 0.0), "Q": _P("Q", a.mu, a.sigma, a.w, 0.9)}
    s = _segments("temporal_only", "P Q P Q P Q P Q".split(), seg_len, seed, concepts=cs)
    s.notes = "a static model is already optimal; this measures concept identification only"
    return s


def sea(seed=7, seg_len=1500):
    """SEA: 3 uniform features, y = (x0 + x1 <= theta), 10% label noise. p(x) is identical in all concepts."""
    rng = np.random.default_rng(seed)
    thetas = {"T8": 8.0, "T9": 9.0, "T7": 7.0, "T95": 9.5}
    order = ["T8", "T9", "T7", "T95"] * 2
    Xs, ys, truth, cps = [], [], [], []
    for k, name in enumerate(order):
        X = rng.uniform(0, 10, (seg_len, 3))
        y = (X[:, 0] + X[:, 1] <= thetas[name]).astype(int)
        y = np.where(rng.random(seg_len) < 0.10, 1 - y, y)
        Xs.append(X), ys.append(y), truth.extend([name] * seg_len), cps.append((k * seg_len, name))
    s = ScenarioData("sea", np.vstack(Xs), np.concatenate(ys), truth, cps)
    s.notes = "feature-only fingerprints cannot separate SEA concepts"
    return s


def hyperplane(seed=7, n=15000, d=10, drift=0.0005):
    """Rotating hyperplane: weights drift continuously, no recurrence."""
    rng = np.random.default_rng(seed)
    w = rng.uniform(0, 1, d)
    direction = rng.choice([-1, 1], d)
    X = rng.uniform(0, 1, (n, d))
    y = np.empty(n, int)
    for i in range(n):
        w = np.clip(w + drift * direction, 0, 1)
        flip = rng.random(d) < 0.001
        direction = np.where(flip, -direction, direction)
        y[i] = int(X[i] @ w > w.sum() / 2)
    y = np.where(rng.random(n) < 0.05, 1 - y, y)
    return ScenarioData("hyperplane", X, y, ["H"] * n, [], False, "incremental drift, no recurring concepts")


SCENARIOS = {f.__name__: f for f in (abrupt_recurring, noisy_recurring, gradual, incremental, similar_x_different_y,
                                      temporal_only, sea, hyperplane)}


# -------------------------------------------------------------------------------------------------
# additional controlled streams (M8): exact concept identities and transition points
# -------------------------------------------------------------------------------------------------
from statistics import NormalDist

_ND = NormalDist()


@dataclass(frozen=True)
class GSpec:
    """Generator for one concept with explicit control of p(x), p(y|x), dependence, class prior and feature availability."""
    mu: tuple
    sigma: tuple = (1.0, 1.0, 1.0, 1.0)
    w: tuple = (1.0, 1.0, 0.0, 0.0)
    corr: float = 0.0
    rho: float = 0.0
    prior: float | None = None            # P(y=1); None = balanced boundary through the concept mean
    absolute: bool = False                # decision boundary w.x = 0 in input space (shared across concepts) instead of through mu
    drop: tuple = ()                      # features that are entirely missing (sensor dropout)
    miss: float = 0.0                     # share of randomly missing values
    noise: float = 0.0                    # measurement noise on X


def custom_stream(name: str, concepts: dict[str, GSpec], schedule: list[tuple[str, int]], seed: int = 7, label_noise: float = 0.04,
                  notes: str = "", has_recurrence: bool = True) -> ScenarioData:
    rng = np.random.default_rng(seed)
    Xs, ys, truth, cps, t0 = [], [], [], [], 0
    for cname, n in schedule:
        g = concepts[cname]
        sp = Spec(cname, g.mu, g.sigma, g.rho, g.corr, noise=g.noise)
        X = sample(sp, n, rng)
        w, mu, sg = np.array(g.w, float), np.array(g.mu, float), np.array(g.sigma, float)
        proj = X @ w if g.absolute else (X - mu) @ w
        thr = 0.0 if g.prior is None else _ND.inv_cdf(1 - g.prior) * math.sqrt(float(np.sum((w * sg) ** 2)))
        y = (proj > thr).astype(int)
        y = np.where(rng.random(n) < label_noise, 1 - y, y)
        if g.drop:
            X[:, list(g.drop)] = np.nan
        if g.miss:
            X = np.where(rng.random(X.shape) < g.miss, np.nan, X)
        Xs.append(X), ys.append(y), truth.extend([cname] * n), cps.append((t0, cname))
        t0 += n
    return ScenarioData(name, np.vstack(Xs), np.concatenate(ys), truth, cps, has_recurrence, notes)


_A = GSpec(mu=(0, 0, 0, 0))
_B = GSpec(mu=(2.5, -1.5, 1, 0), sigma=(0.8, 1.2, 1.0, 1.0), w=(1, -1, 0.5, 0))
_C = GSpec(mu=(-2, 2, -1, 1.5), sigma=(1.5, 0.7, 1.0, 0.8), w=(-1, -0.5, 1, 1))


def aba(seed=7, seg_len=1500):
    return custom_stream("aba", {"A": _A, "B": _B}, [("A", seg_len), ("B", seg_len), ("A", seg_len)], seed, notes="A -> B -> A")


def abca(seed=7, seg_len=1500):
    return custom_stream("abca", {"A": _A, "B": _B, "C": _C}, [(c, seg_len) for c in "ABCA"], seed, notes="A -> B -> C -> A")


def aba_noisy(seed=7, seg_len=1500):
    n = lambda g: GSpec(**{**g.__dict__, "noise": 0.5})
    return custom_stream("aba_noisy", {"A": n(_A), "B": n(_B)}, [("A", seg_len), ("B", seg_len), ("A", seg_len)], seed, label_noise=0.12, notes="A -> B -> A with measurement and label noise")


def ab_gradual(seed=7, seg_len=1500):
    s = _segments("ab_gradual", ["A", "B", "A"], seg_len, seed, transition=400, mode="gradual")
    s.notes = "A -> B -> A, gradual 400-sample transitions"
    return s


def diff_x_same_y(seed=7, seg_len=1500):
    """p(x) differs, p(y|x) is the same function (shared absolute boundary): one model serves every concept."""
    c1, c2 = GSpec(mu=(0, 0, 0, 0), absolute=True), GSpec(mu=(1.5, -1.5, 0, 0), absolute=True)
    return custom_stream("diff_x_same_y", {"X1": c1, "X2": c2}, [(c, seg_len) for c in ("X1", "X2", "X1", "X2", "X1")], seed,
                         notes="different P(X), identical P(Y|X)")


def same_marginal_diff_corr(seed=7, seg_len=1500):
    """Identical marginals and labelling rule; only the dependence between features differs."""
    c0, c1 = GSpec(mu=(0, 0, 0, 0), absolute=True), GSpec(mu=(0, 0, 0, 0), corr=0.75, absolute=True)
    return custom_stream("same_marginal_diff_corr", {"C0": c0, "C1": c1}, [(c, seg_len) for c in ("C0", "C1", "C0", "C1", "C0")], seed,
                         notes="identical marginals, different dependence; a single model is already optimal")


def varied_sizes(seed=7):
    return custom_stream("varied_sizes", {"A": _A, "B": _B, "C": _C}, [("A", 500), ("B", 3000), ("A", 800), ("C", 400), ("B", 2500), ("A", 1500), ("C", 1000)], seed,
                         notes="recurring concepts with very different segment lengths")


def overlapping(seed=7, seg_len=1500):
    """Fingerprints close together: centres 0.4 sigma apart, different decision boundaries."""
    cs = {"O1": GSpec(mu=(0, 0, 0, 0), w=(1, 1, 0, 0)), "O2": GSpec(mu=(0.4, 0, 0, 0), w=(1, -1, 0, 0)),
          "O3": GSpec(mu=(0, 0.4, 0, 0), w=(0, 1, 1, 0)), "O4": GSpec(mu=(0.4, 0.4, 0, 0), w=(-1, 1, 0, 1))}
    return custom_stream("overlapping", cs, [(c, seg_len) for c in ("O1", "O2", "O3", "O4", "O1", "O3", "O2", "O4")], seed,
                         notes="concepts with overlapping feature distributions")


def imbalanced(seed=7, seg_len=1500):
    cs = {"A": GSpec(mu=(0, 0, 0, 0), prior=0.08), "B": GSpec(mu=(2.5, -1.5, 1, 0), w=(1, -1, 0.5, 0), prior=0.5),
          "C": GSpec(mu=(-2, 2, -1, 1.5), w=(-1, -0.5, 1, 1), prior=0.92)}
    return custom_stream("imbalanced", cs, [(c, seg_len) for c in "ABCABCA"], seed, notes="class priors 8% / 50% / 92%")


def feature_availability(seed=7, seg_len=1500):
    cs = {"A": _A, "B": GSpec(mu=(2.5, -1.5, 1, 0), w=(1, -1, 0.5, 0), drop=(3,)), "C": GSpec(mu=(-2, 2, -1, 1.5), w=(-1, -0.5, 1, 1), miss=0.3)}
    return custom_stream("feature_availability", cs, [(c, seg_len) for c in "ABACABA"], seed,
                         notes="B: feature 3 is missing entirely; C: 30% of values missing at random")


SCENARIOS.update({f.__name__: f for f in (aba, abca, aba_noisy, ab_gradual, diff_x_same_y, same_marginal_diff_corr, varied_sizes, overlapping,
                                          imbalanced, feature_availability)})
