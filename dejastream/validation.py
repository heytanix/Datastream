"""Statistical validation of candidate models against the active (baseline) model on labelled samples.

* task-aware metrics (classification / regression / anomaly detection); every metric is "higher is better"
  (losses are negated), so one comparison routine serves all of them
* paired comparison: candidate and baseline are scored on the SAME observations, and the difference is
  bootstrapped over (block-resampled) observations, which respects serial dependence
* objectives: ``non_inferiority`` (candidate not worse than baseline by more than ``margin``), ``superiority``
  (candidate better by more than ``min_delta``), ``point`` (legacy: point estimates only)
* absolute floor: instead of one fixed accuracy for every dataset, the candidate must beat a trivial predictor
  (majority class / training mean / all-normal) with confidence
* choosing among several candidates on the same samples is optimistically biased; three protocols are supported:
  ``pick_best`` (legacy, biased), ``bonferroni`` (alpha / m), ``split_confirm`` (select on one part, confirm on fresh labels)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Callable, NamedTuple, Sequence

import numpy as np

# --------------------------------------------------------------------------------------------------
# metrics (higher is better)
# --------------------------------------------------------------------------------------------------


def _acc(y, p, s=None):
    return float(np.mean(np.asarray(y) == np.asarray(p)))


def _balanced_acc(y, p, s=None):
    y, p = np.asarray(y), np.asarray(p)
    cls = np.unique(y)
    return float(np.mean([np.mean(p[y == c] == c) for c in cls]))


def _macro_f1(y, p, s=None):
    y, p = np.asarray(y), np.asarray(p)
    f = []
    for c in np.unique(np.concatenate([y, p])):
        tp = np.sum((p == c) & (y == c))
        prec = tp / max(np.sum(p == c), 1)
        rec = tp / max(np.sum(y == c), 1)
        f.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f))


def _neg_logloss(y, p, s):
    if s is None:
        return math.nan
    s = np.clip(np.asarray(s, float), 1e-9, 1 - 1e-9)
    y = np.asarray(y)
    return float(-np.mean(-(y * np.log(s) + (1 - y) * np.log(1 - s))))


def _neg_ece(y, p, s, bins=10):
    if s is None:
        return math.nan
    s, y = np.asarray(s, float), np.asarray(y)
    conf, correct = np.maximum(s, 1 - s), ((s > 0.5).astype(int) == y).astype(float)
    idx = np.clip(np.digitize(conf, np.linspace(0.5, 1, bins + 1)[1:-1]), 0, bins - 1)
    return float(-sum(abs(correct[idx == b].mean() - conf[idx == b].mean()) * np.mean(idx == b) for b in range(bins) if (idx == b).any()))


def _neg_mae(y, p, s=None):
    return float(-np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def _neg_rmse(y, p, s=None):
    return float(-math.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def _neg_rel_err(y, p, s=None):
    y = np.asarray(y, float)
    den = np.mean(np.abs(y))
    return math.nan if den == 0 else float(-np.mean(np.abs(y - np.asarray(p, float))) / den)


def _r2(y, p, s=None):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss = np.sum((y - y.mean()) ** 2)
    return math.nan if ss == 0 else float(1 - np.sum((y - p) ** 2) / ss)


def _precision(y, p, s=None):
    y, p = np.asarray(y), np.asarray(p)
    return math.nan if np.sum(p == 1) == 0 else float(np.sum((p == 1) & (y == 1)) / np.sum(p == 1))


def _recall(y, p, s=None):
    y, p = np.asarray(y), np.asarray(p)
    return math.nan if np.sum(y == 1) == 0 else float(np.sum((p == 1) & (y == 1)) / np.sum(y == 1))


def _f1(y, p, s=None):
    pr, rc = _precision(y, p), _recall(y, p)
    if not (np.isfinite(pr) and np.isfinite(rc)):
        return 0.0 if np.sum(np.asarray(y) == 1) > 0 else math.nan
    return 0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc)


def _avg_precision(y, p, s):
    if s is None or np.sum(np.asarray(y) == 1) == 0:
        return math.nan
    order = np.argsort(-np.asarray(s, float), kind="stable")
    yy = np.asarray(y)[order]
    tp = np.cumsum(yy == 1)
    prec = tp / (np.arange(len(yy)) + 1)
    return float(np.sum(prec[yy == 1]) / np.sum(yy == 1))


METRICS: dict[str, dict[str, Callable]] = {
    "classification": {"accuracy": _acc, "balanced_accuracy": _balanced_acc, "macro_f1": _macro_f1, "neg_log_loss": _neg_logloss, "neg_ece": _neg_ece},
    "regression": {"neg_mae": _neg_mae, "neg_rmse": _neg_rmse, "neg_relative_error": _neg_rel_err, "r2": _r2},
    "anomaly": {"precision": _precision, "recall": _recall, "f1": _f1, "average_precision": _avg_precision},
}


def trivial_prediction(task: str, y: np.ndarray) -> np.ndarray:
    """What a model that learned nothing would predict on these samples."""
    y = np.asarray(y)
    if task == "classification":
        vals, cnt = np.unique(y, return_counts=True)
        return np.full(len(y), vals[np.argmax(cnt)])
    if task == "regression":
        return np.full(len(y), float(np.mean(y)))
    return np.zeros(len(y), int)                      # anomaly: everything is normal


# --------------------------------------------------------------------------------------------------
# configuration and results
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ValidationConfig:
    task: str = "classification"
    metric: str = "accuracy"
    objective: str = "non_inferiority"       # "non_inferiority" | "superiority" | "point"
    alpha: float = 0.05
    margin: float = 0.02                     # non-inferiority margin, in metric units
    min_delta: float = 0.0                   # superiority: required improvement
    n_boot: int = 400
    block: int = 5                           # block length of the bootstrap (serial dependence)
    seed: int = 0
    min_samples: int = 30
    floor: str = "majority_point"            # "majority_point": point estimate above the trivial predictor (+ floor_margin) |
                                             # "majority": significantly above it (conservative at n~40) | "abs" | "none"
    floor_margin: float = 0.0
    abs_floor: float | None = None
    selection: str = "bonferroni"            # "pick_best" | "bonferroni" | "split_confirm"
    confirm_min: int = 30                    # fresh labels needed in the confirmation stage (split_confirm)
    ci_method: str = "t"                     # "t": bootstrap SE x t critical value | "percentile": bootstrap percentiles

    def __post_init__(self):
        if self.task not in METRICS or self.metric not in METRICS[self.task]:
            raise ValueError(f"metric {self.metric!r} is not defined for task {self.task!r}: {list(METRICS.get(self.task, {}))}")
        if self.objective not in ("non_inferiority", "superiority", "point"):
            raise ValueError("bad objective")
        if self.selection not in ("pick_best", "bonferroni", "split_confirm") or self.floor not in ("majority", "majority_point", "abs", "none") \
                or self.ci_method not in ("t", "percentile"):
            raise ValueError("bad selection / floor / ci_method")
        if not 0 < self.alpha < 1 or self.margin < 0 or self.n_boot < 50 or self.min_samples < 5 or self.block < 1:
            raise ValueError("bad alpha / margin / n_boot / min_samples / block")
        if self.floor == "abs" and self.abs_floor is None:
            raise ValueError("floor='abs' needs abs_floor")


class Pred(NamedTuple):
    pred: np.ndarray
    score: np.ndarray | None = None            # P(class 1) / anomaly score, for metrics that need it


@dataclass
class Comparison:
    verdict: str                               # "accept" | "reject" | "insufficient"
    metric: str
    n: int
    cand: float
    base: float
    diff: float
    ci_low: float                              # one-sided lower bound of cand - base at the (adjusted) alpha
    ci_high: float
    alpha: float
    p_value: float | None
    floor_ok: bool | None
    trivial: float | None
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        r = lambda v: None if v is None or not math.isfinite(v) else round(float(v), 4)
        return {"verdict": self.verdict, "metric": self.metric, "n": self.n, "candidate": r(self.cand), "baseline": r(self.base), "diff": r(self.diff),
                "ci_low": r(self.ci_low), "ci_high": r(self.ci_high), "alpha": r(self.alpha), "p_value": r(self.p_value),
                "floor_ok": self.floor_ok, "trivial": r(self.trivial), "reasons": list(self.reasons)}


@dataclass
class Selection:
    chosen: str | None
    comparisons: dict[str, Comparison]
    protocol: str
    selected_metric: float | None = None       # metric of the chosen model on the SELECTION samples (optimistic if reused)
    confirmed_metric: float | None = None      # ... on samples not used for selection (split_confirm only)


# --------------------------------------------------------------------------------------------------
# paired bootstrap
# --------------------------------------------------------------------------------------------------
_ND = NormalDist()


def _t_crit(p: float, df: float) -> float:
    """Student-t quantile by the Cornish-Fisher expansion (accurate to ~1e-3 for df >= 5)."""
    z = _ND.inv_cdf(p)
    return z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)


def _block_indices(n: int, block: int, rng: np.random.Generator, B: int) -> np.ndarray:
    nb = int(math.ceil(n / block))
    starts = rng.integers(0, n, (B, nb))
    return ((starts[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(B, -1)[:, :n]


def _subset(p: Pred, idx) -> Pred:
    return Pred(np.asarray(p.pred)[idx], None if p.score is None else np.asarray(p.score)[idx])


def compare_paired(y: Sequence, cand: Pred, base: Pred, cfg: ValidationConfig, alpha: float | None = None,
                   rng: np.random.Generator | None = None) -> Comparison:
    """Paired comparison of a candidate with the baseline on the same labelled observations."""
    y = np.asarray(y)
    n = len(y)
    a = cfg.alpha if alpha is None else alpha
    f = METRICS[cfg.task][cfg.metric]
    mc, mb = f(y, cand.pred, cand.score), f(y, base.pred, base.score)
    if n < cfg.min_samples:
        return Comparison("insufficient", cfg.metric, n, mc, mb, mc - mb, math.nan, math.nan, a, None, None, None,
                          [f"only {n} labelled samples, need {cfg.min_samples}"])
    if not (np.isfinite(mc) and np.isfinite(mb)):
        return Comparison("insufficient", cfg.metric, n, mc, mb, math.nan, math.nan, math.nan, a, None, None, None,
                          [f"metric {cfg.metric} undefined on these samples"])
    rng = rng or np.random.default_rng(cfg.seed)
    idx = _block_indices(n, cfg.block, rng, cfg.n_boot)
    diffs = np.empty(cfg.n_boot)
    for b in range(cfg.n_boot):
        ii = idx[b]
        diffs[b] = f(y[ii], cand.pred[ii], None if cand.score is None else cand.score[ii]) - f(y[ii], base.pred[ii], None if base.score is None else base.score[ii])
    diffs = diffs[np.isfinite(diffs)]
    d_obs = mc - mb
    if cfg.ci_method == "t":
        se, df = float(diffs.std(ddof=1)), max(n / cfg.block - 1, 5.0)       # blocks, not observations, carry the information
        lo, hi = d_obs - _t_crit(1 - a, df) * se, d_obs + _t_crit(1 - a, df) * se
    else:
        lo, hi = float(np.quantile(diffs, a)), float(np.quantile(diffs, 1 - a))
    reasons: list[str] = []
    if cfg.objective == "point":
        ok_main = d_obs >= 0.0
        p = None
        reasons.append(f"point estimate difference {d_obs:+.3f}")
    else:
        d0 = -cfg.margin if cfg.objective == "non_inferiority" else cfg.min_delta
        ok_main = lo > d0
        p = float((1 + np.sum((diffs - d_obs) >= (d_obs - d0))) / (len(diffs) + 1))
        reasons.append(f"{1 - a:.0%} lower bound of the difference {lo:+.3f} vs required {d0:+.3f} ({cfg.objective.replace('_', '-')})")
    floor_ok, trivial = None, None
    if cfg.floor == "majority":
        tp = trivial_prediction(cfg.task, y)
        trivial = f(y, tp, None)
        if np.isfinite(trivial):
            tdiff = np.empty(cfg.n_boot)
            for b in range(cfg.n_boot):
                ii = idx[b]
                tdiff[b] = f(y[ii], cand.pred[ii], None if cand.score is None else cand.score[ii]) - f(y[ii], tp[ii], None)
            tdiff = tdiff[np.isfinite(tdiff)]
            if cfg.ci_method == "t":
                tlo = (mc - f(y, tp, None)) - _t_crit(1 - a, max(n / cfg.block - 1, 5.0)) * float(tdiff.std(ddof=1))
            else:
                tlo = float(np.quantile(tdiff, a))
            floor_ok = bool(tlo > 0)
            reasons.append("beats the trivial predictor" if floor_ok else f"does not clearly beat the trivial predictor ({trivial:.3f})")
    elif cfg.floor == "majority_point":
        tp = trivial_prediction(cfg.task, y)
        trivial = f(y, tp, None)
        if np.isfinite(trivial):
            floor_ok = bool(mc > trivial + cfg.floor_margin)
            reasons.append(f"metric {mc:.3f} {'>' if floor_ok else '<='} trivial predictor {trivial:.3f} + {cfg.floor_margin}")
    elif cfg.floor == "abs":
        floor_ok = bool(mc >= cfg.abs_floor)
        reasons.append(f"metric {mc:.3f} {'>=' if floor_ok else '<'} absolute floor {cfg.abs_floor}")
    verdict = "accept" if ok_main and floor_ok is not False else "reject"
    return Comparison(verdict, cfg.metric, n, mc, mb, d_obs, lo, hi, a, p, floor_ok, trivial, reasons)


def select_candidate(y: Sequence, candidates: dict[str, Pred], base: Pred, cfg: ValidationConfig,
                     rng: np.random.Generator | None = None) -> Selection:
    """Choose among candidate models with the configured multiple-candidate protocol (offline, all labels given).

    pick_best      choose the best by metric, test at alpha on the SAME samples (optimistic; the legacy behaviour)
    bonferroni     test every candidate at alpha / m, choose the best of the accepted
    split_confirm  choose on the first half, confirm the choice on the second half
    """
    y = np.asarray(y)
    rng = rng or np.random.default_rng(cfg.seed)
    f = METRICS[cfg.task][cfg.metric]
    names = list(candidates)
    if not names:
        return Selection(None, {}, cfg.selection)
    if cfg.selection == "split_confirm":
        h = len(y) // 2
        sel, conf = slice(0, h), slice(h, len(y))
        sc = {k: f(y[sel], candidates[k].pred[sel], None if candidates[k].score is None else candidates[k].score[sel]) for k in names}
        best = max((k for k in names if np.isfinite(sc[k])), key=lambda k: sc[k], default=None)
        if best is None:
            return Selection(None, {}, "split_confirm")
        cmp = compare_paired(y[conf], _subset(candidates[best], conf), _subset(base, conf), cfg, rng=rng)
        return Selection(best if cmp.verdict == "accept" else None, {best: cmp}, "split_confirm", sc[best], cmp.cand)
    m = len(names)
    a = cfg.alpha / m if cfg.selection == "bonferroni" else cfg.alpha
    if cfg.selection == "pick_best":
        best = max(names, key=lambda k: f(y, candidates[k].pred, candidates[k].score) if np.isfinite(f(y, candidates[k].pred, candidates[k].score)) else -np.inf)
        cmp = compare_paired(y, candidates[best], base, cfg, alpha=a, rng=rng)
        return Selection(best if cmp.verdict == "accept" else None, {best: cmp}, "pick_best", cmp.cand)
    cmps = {k: compare_paired(y, candidates[k], base, cfg, alpha=a, rng=rng) for k in names}
    ok = [k for k in names if cmps[k].verdict == "accept"]
    best = max(ok, key=lambda k: cmps[k].cand) if ok else None
    return Selection(best, cmps, "bonferroni", None if best is None else cmps[best].cand)
