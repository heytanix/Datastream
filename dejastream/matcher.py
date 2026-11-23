"""Concept repository and confidence-aware, hierarchical matching.

Level 1  global vector (median, robust sigma, q10, q90 per feature): vectorised retrieval of top-K
Level 2  full feature views (statistics, distributions, temporal) with feature-group aggregation
Level 3  predictive evidence (observed behaviour of the candidate's model vs the stored one), if any

The repository stores *fingerprints and metadata only*. Trained models live elsewhere and are
referenced by ``model_id``; concept identity and model identity are separate.
"""
from __future__ import annotations

import enum
import math
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .mvfp import (FeatureFingerprint, PredictiveFingerprint, SCHEMA_VERSION, merge_feature_fp,
                   merge_predictive_fp, migrate_v1)
from .memory import ConceptSummaries
from .similarity import Distances, SimilarityConfig, compare

RECORD_SCHEMA = "2.1"                          # records that carry mergeable summaries; plain fingerprints stay "2.0"


class Decision(str, enum.Enum):
    HIGH_CONFIDENCE_MATCH = "HIGH_CONFIDENCE_MATCH"
    CANDIDATE_MATCH = "CANDIDATE_MATCH"
    NEW_CONCEPT = "NEW_CONCEPT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class MatchConfig:
    top_k: int = 5                    # candidates passed from level 1 to level 2
    min_samples: int = 30             # below this: INSUFFICIENT_EVIDENCE
    min_completeness: float = 0.6     # share of feature-view weight that could be computed
    t_high: float = 0.18              # D at or below this (and well separated) -> HIGH_CONFIDENCE_MATCH
    t_cand: float = 0.32              # D at or below this -> CANDIDATE_MATCH, above -> NEW_CONCEPT
    k_intra: float = 2.5              # thresholds widen to k_intra * the concept's own window-to-window D
    min_margin: float = 0.06          # best vs second-best separation needed for HIGH
    margin_scale: float = 0.15
    n_full: int = 100                 # samples at which the evidence term saturates
    conf_high: float = 0.35
    pred_missing_factor: float = 0.85  # confidence discount when no predictive evidence exists
    max_concepts: int = 50
    # -- Level 1 retrieval
    l1_repr: str = "basic"            # "basic" (median, sigma, q10, q90) | "shape" | "shape_corr" | "full"
    top_k_mode: str = "fixed"         # "fixed": top_k candidates | "adaptive": between k_min and k_max by score gap
    k_min: int = 3
    k_max: int = 10
    l1_ratio: float = 1.5             # adaptive: keep candidates scoring within l1_ratio x best (+ l1_delta)
    l1_delta: float = 0.1
    expand_on_new: bool = False       # if the match says NEW_CONCEPT, re-rank every stored concept before concluding it
    max_full_scan: int = 100          # ... but only when the repository holds at most this many concepts
    # -- thresholds
    threshold_mode: str = "legacy"    # "legacy": max(t, k_intra x intra) | "empirical": conformal quantiles of same-concept distances
    alpha_cand: float = 0.02          # empirical: share of true same-concept windows allowed to fall outside the candidate range
    alpha_high: float = 0.10          # ... and outside the high-confidence range
    min_obs: int = 5                  # same-concept distances needed before a per-concept estimate is trusted
    pool_size: int = 500              # global pool of same-concept distances (fallback for young concepts)
    var_window: int = 50              # per-concept distances kept
    t_min: float = 0.05
    t_max: float = 0.9
    size_ratio: float = 2.0           # compare only calibration windows within this factor of the query size
    # -- concept memory
    merge_mode: str = "legacy"        # "legacy": weighted averaging of fingerprints | "mergeable": exact moments + sketches + history
    sketch_compression: int = 50
    history_windows: int = 20
    eviction: str = "legacy"          # "legacy"/"least_used": (uses, last_used) | "lru" | "score": recurrence, recency and model performance
    evict_weights: tuple = (1.0, 1.0, 1.0)    # score = w0 log(1+uses) + w1 recency-rank + w2 historical model accuracy
    grace_updates: int = 0            # a concept needs this many updates before it can be evicted (unless nothing else can be)

    def __post_init__(self):
        if self.threshold_mode not in ("legacy", "empirical"):
            raise ValueError("threshold_mode must be legacy or empirical")
        if self.l1_repr not in ("basic", "shape", "shape_corr", "full") or self.top_k_mode not in ("fixed", "adaptive"):
            raise ValueError("bad l1_repr / top_k_mode")
        if not (0 < self.alpha_cand < 1 and 0 < self.alpha_high < 1):
            raise ValueError("alpha must be in (0, 1)")
        if self.merge_mode not in ("legacy", "mergeable") or self.eviction not in ("legacy", "least_used", "lru", "score"):
            raise ValueError("bad merge_mode / eviction")
        if self.max_concepts < 1 or len(self.evict_weights) != 3 or self.grace_updates < 0:
            raise ValueError("bad max_concepts / evict_weights / grace_updates")


@dataclass
class Candidate:
    concept_id: str
    distances: Distances
    confidence: float = 0.0
    thr_cand: float | None = None

    @property
    def D(self) -> float:
        return self.distances.combined

    def as_dict(self) -> dict:
        return {"concept_id": self.concept_id, "confidence": round(self.confidence, 3), **self.distances.as_dict()}


@dataclass
class MatchResult:
    decision: Decision
    best_id: str | None
    similarity: float | None          # 1 - D: how close the fingerprints are
    D: float | None
    components: dict[str, float | None]
    confidence: float                 # heuristic match-confidence score in [0, 1]; NOT a calibrated probability
    reason: str
    ranking: list[Candidate] = field(default_factory=list)
    n: int = 0
    completeness: float = 0.0
    # -- the three quantities, kept separate (M5)
    evidence_quality: float = 0.0     # how reliable the comparison is (sample size, views present, resolvability)
    confidence_terms: dict[str, float] = field(default_factory=dict)   # factors of the heuristic score
    match_probability: float | None = None     # calibrated P(best candidate is the true concept), only with a fitted calibrator
    margin: float | None = None       # D_2 - D_1 (None with a single stored concept)
    margin_norm: float | None = None  # margin / D_2, or distance-to-new-concept boundary when only one concept exists
    thresholds: dict[str, Any] = field(default_factory=dict)
    noise: dict[str, str] = field(default_factory=dict)
    n_eff: float | None = None
    n_scored: int = 0
    n_total: int = 0
    has_pred: bool = False

    def as_dict(self, top: int = 3) -> dict:
        r = lambda v, nd=4: None if v is None else round(float(v), nd)
        return {"decision": self.decision.value, "best": self.best_id, "similarity": r(self.similarity), "D": r(self.D),
                "components": {k: r(v) for k, v in self.components.items()}, "confidence": r(self.confidence, 3),
                "evidence_quality": r(self.evidence_quality, 3), "match_probability": r(self.match_probability, 3),
                "margin": r(self.margin), "reason": self.reason, "n": self.n, "completeness": r(self.completeness, 3),
                "ranking": [c.as_dict() for c in self.ranking[:top]]}

    def explanation(self, top: int = 3) -> dict:
        """Structured account of the decision: what was chosen, why alternatives lost, which views contributed,
        and what extra evidence would improve it."""
        r = lambda v, nd=3: None if v is None else round(float(v), nd)
        avail = [k for k, v in self.components.items() if v is not None]
        missing = [k for k in ("stat", "dist", "time", "pred") if k not in avail]
        rejected = []
        for c in self.ranking[1:top + 1]:
            why = []
            if self.D is not None and c.D - self.D < 0.02:
                why.append("almost as close as the best candidate (ambiguous)")
            elif self.D is not None:
                why.append(f"farther than the best candidate by {c.D - self.D:.3f}")
            if c.thr_cand is not None and c.D > c.thr_cand:
                why.append(f"outside its own match range ({c.thr_cand:.3f})")
            rejected.append({"concept_id": c.concept_id, "D": r(c.D), "reasons": why})
        if self.n_total > self.n_scored:
            rejected.append({"concept_id": f"{self.n_total - self.n_scored} other stored concepts", "D": None,
                             "reasons": ["not scored: outside the Level-1 shortlist"]})
        need = []
        if not self.has_pred:
            need.append("labelled samples, so the candidate model can be validated (predictive evidence)")
        if self.evidence_quality < 0.6 and self.n:
            need.append("more post-drift observations (evidence quality is limited by sample size, missing views or noise)")
        if any(v == "unresolvable" for v in self.noise.values()):
            need.append("a larger window: at this size a real change in " + ", ".join(k for k, v in self.noise.items() if v == "unresolvable") + " could not be detected")
        if self.margin is not None and self.margin < 0.06 and self.decision in (Decision.CANDIDATE_MATCH, Decision.HIGH_CONFIDENCE_MATCH):
            need.append("labelled validation to separate the best candidate from near-duplicates")
        return {"decision": self.decision.value, "concept_id": self.best_id, "distance": r(self.D), "similarity": r(self.similarity),
                "margin": r(self.margin), "margin_normalised": r(self.margin_norm), "evidence_quality": r(self.evidence_quality),
                "confidence_score": r(self.confidence), "confidence_terms": {k: r(v) for k, v in self.confidence_terms.items()},
                "match_probability": r(self.match_probability), "completeness": r(self.completeness),
                "available_views": avail, "missing_views": missing, "noise_assessment": dict(self.noise),
                "thresholds": {k: (r(v) if isinstance(v, float) else v) for k, v in self.thresholds.items()},
                "alternatives_rejected": rejected, "reason": self.reason, "additional_evidence_needed": need}


@dataclass
class VariabilityTracker:
    """Recent window-to-concept distances of one concept (same-concept evidence) with each window's size."""
    obs: deque = field(default_factory=lambda: deque(maxlen=50))

    def add(self, n: int, d: float, maxlen: int | None = None) -> None:
        if maxlen and self.obs.maxlen != maxlen:
            self.obs = deque(self.obs, maxlen=maxlen)
        self.obs.append((int(n), float(d)))

    def values(self, n: int | None = None, ratio: float = 2.0) -> np.ndarray:
        if n is None:
            return np.array([d for _, d in self.obs])
        return np.array([d for m, d in self.obs if n / ratio <= m <= n * ratio])

    def summary(self) -> dict:
        v = self.values()
        if not len(v):
            return {"n_obs": 0}
        med = float(np.median(v))
        return {"n_obs": int(len(v)), "median": med, "mean": float(v.mean()), "mad": float(np.median(np.abs(v - med))),
                "q90": float(np.quantile(v, 0.9)), "q95": float(np.quantile(v, 0.95))}


def conformal_quantile(vals: np.ndarray, p: float) -> float:
    """Split-conformal style upper quantile: the ceil((n+1)p)-th smallest value. With too few values to certify level p
    it returns 1.25 x the maximum (deliberately conservative). Valid for exchangeable windows; serially dependent windows
    make the guarantee approximate."""
    v = np.sort(np.asarray(vals, float))
    n = len(v)
    if n == 0:
        raise ValueError("no calibration values")
    k = int(math.ceil((n + 1) * p))
    return float(v[k - 1]) if k <= n else float(v[-1] * 1.25)


FEATURE_NAMES = ("D", "margin_norm", "log_n_eff", "completeness", "var_ratio", "has_pred")


def match_features(res: "MatchResult") -> np.ndarray | None:
    """Feature vector of a match result for the calibrator (None when there is no best candidate)."""
    if res.D is None or res.margin_norm is None or not res.n_eff:
        return None
    thr = res.thresholds.get("thr_cand") or 1.0
    var = res.thresholds.get("var_ref") or 0.0
    return np.array([res.D, res.margin_norm, math.log(max(res.n_eff, 1.0)), res.completeness, min(var / thr, 2.0), float(res.has_pred)])


class MatchCalibrator:
    """Logistic regression  P(best candidate is the true concept | match features), fitted on held-out labelled matches.

    Deliberately small: six features, ridge penalty, Newton iterations. A calibration is only as good as the
    calibration data: it is not claimed to transfer to streams unlike those it was fitted on.
    """

    def __init__(self, l2: float = 1e-2):
        self.l2 = l2
        self.w: np.ndarray | None = None
        self.mu = self.sd = None

    def fit(self, X: np.ndarray, y: np.ndarray, iters: int = 50) -> "MatchCalibrator":
        X, y = np.asarray(X, float), np.asarray(y, float)
        if len(X) < 20 or y.min() == y.max():
            raise ValueError("need at least 20 labelled matches containing both outcomes")
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = np.hstack([(X - self.mu) / self.sd, np.ones((len(X), 1))])
        w = np.zeros(Z.shape[1])
        reg = self.l2 * np.eye(len(w)); reg[-1, -1] = 0
        for _ in range(iters):
            p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
            g = Z.T @ (p - y) + reg @ w
            H = (Z * (p * (1 - p))[:, None]).T @ Z + reg + 1e-9 * np.eye(len(w))
            step = np.linalg.solve(H, g)
            w -= step
            if np.max(np.abs(step)) < 1e-8:
                break
        self.w = w
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("calibrator is not fitted")
        X = np.atleast_2d(np.asarray(X, float))
        Z = np.hstack([(X - self.mu) / self.sd, np.ones((len(X), 1))])
        return 1 / (1 + np.exp(-np.clip(Z @ self.w, -30, 30)))


@dataclass
class ConceptRecord:
    concept_id: str
    fp: FeatureFingerprint
    model_id: str | None
    sample_count: int
    created_t: int = 0
    last_used_t: int = 0
    uses: int = 1
    intra: float | None = None                      # mean window-to-centroid D: how variable the concept is
    pred: PredictiveFingerprint | None = None
    model_history: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    var: VariabilityTracker = field(default_factory=VariabilityTracker)
    summaries: ConceptSummaries | None = None       # mergeable summaries (merge_mode="mergeable")
    update_count: int = 0                           # windows merged into this concept
    model_versions: list[dict] = field(default_factory=list)   # {"model_id", "since_t", "reason"}

    @property
    def index(self) -> int:
        return int(self.concept_id.rsplit("_", 1)[1])

    def to_dict(self) -> dict:
        return {"schema_version": RECORD_SCHEMA if self.summaries is not None else self.schema_version, "concept_id": self.concept_id,
                "model_id": self.model_id, "update_count": self.update_count, "model_versions": list(self.model_versions),
                "summaries": None if self.summaries is None else self.summaries.to_dict(),
                "sample_count": self.sample_count, "created_t": self.created_t, "last_used_t": self.last_used_t,
                "uses": self.uses, "intra": self.intra, "model_history": list(self.model_history), "meta": dict(self.meta),
                "feature_fingerprint": self.fp.to_dict(), "variability": [[n, d] for n, d in self.var.obs],
                "predictive_fingerprint": None if self.pred is None else self.pred.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "ConceptRecord":
        ver = str(d.get("schema_version", "1.0"))
        if ver.startswith("1"):                       # legacy: {"id", "fingerprint": [17 raw numbers], "n"?}
            fp = migrate_v1(d["fingerprint"], int(d.get("n", 0)) or 1)
            return cls(f"concept_{int(d['id']):03d}", fp, d.get("model_id"), fp.n, int(d.get("created_t", 0)),
                       int(d.get("created_t", 0)), int(d.get("uses", 1)), d.get("d_in") or None,
                       meta={"migrated_from": "1.0"}, schema_version=SCHEMA_VERSION)
        pf = d.get("predictive_fingerprint")
        rec = cls(d["concept_id"], FeatureFingerprint.from_dict(d["feature_fingerprint"]), d.get("model_id"),
                  int(d["sample_count"]), int(d.get("created_t", 0)), int(d.get("last_used_t", 0)),
                  int(d.get("uses", 1)), d.get("intra"), None if pf is None else PredictiveFingerprint.from_dict(pf),
                  list(d.get("model_history", [])), dict(d.get("meta", {})), ver)
        for n, dist in d.get("variability", []):
            rec.var.add(n, dist)
        rec.update_count = int(d.get("update_count", 0))
        rec.model_versions = list(d.get("model_versions", []))
        if d.get("summaries"):
            rec.summaries = ConceptSummaries.from_dict(d["summaries"])
        return rec

    def nbytes(self) -> int:
        return self.fp.nbytes() + 8 * 8 + (0 if self.summaries is None else self.summaries.nbytes())


class ConceptRepository:
    """Bounded in-process store of concept fingerprints with hierarchical retrieval."""

    def __init__(self, match: MatchConfig | None = None, sim: SimilarityConfig | None = None):
        self.match_cfg = match or MatchConfig()
        self.sim_cfg = sim or SimilarityConfig()
        self.records: dict[str, ConceptRecord] = {}
        self._next = 1
        self._dirty = True                     # Level-1 index must be rebuilt
        self._idx: dict | None = None
        self.last_l1_scores: np.ndarray | None = None
        self.pool: deque = deque(maxlen=self.match_cfg.pool_size)     # (n, D) of every same-concept observation
        self.calibrator: MatchCalibrator | None = None

    def __len__(self) -> int:
        return len(self.records)

    def __contains__(self, cid: str) -> bool:
        return cid in self.records

    # -- mutation -----------------------------------------------------------------------------
    def add(self, fp: FeatureFingerprint, model_id: str | None, t: int = 0, pred: PredictiveFingerprint | None = None,
            protect: str | None = None, meta: dict | None = None, X: np.ndarray | None = None) -> ConceptRecord:
        """Store a new concept. With ``merge_mode="mergeable"`` the raw window ``X`` (if given) makes the summaries exact."""
        cid = f"concept_{self._next:03d}"
        self._next += 1
        rec = ConceptRecord(cid, fp, model_id, fp.n, t, t, 1, None, pred, [model_id] if model_id else [], dict(meta or {}))
        rec.model_versions = [{"model_id": model_id, "since_t": int(t), "reason": "created"}] if model_id else []
        if self.match_cfg.merge_mode == "mergeable":
            c, h = self.match_cfg.sketch_compression, self.match_cfg.history_windows
            rec.summaries = ConceptSummaries.from_window(X, fp, c, h) if X is not None else ConceptSummaries.from_fp(fp, c, h)
            rec.meta["summaries_exact"] = X is not None
        self.records[cid] = rec
        self._dirty = True
        keep = {cid} | ({protect} if protect else set())      # the new concept and the caller's concept are never evicted
        while len(self.records) > max(self.match_cfg.max_concepts, len(keep)):
            before = len(self.records)
            self._evict(protect=keep)
            if len(self.records) == before:
                break
        return rec

    def _evict(self, protect: "str | set[str]") -> None:
        """Drop one concept. Independent of fingerprint similarity: decided by recurrence, recency and model performance only."""
        cfg = self.match_cfg
        keep = {protect} if isinstance(protect, str) else set(protect)
        victims = [r for r in self.records.values() if r.concept_id not in keep]
        if not victims:
            return
        if cfg.grace_updates:
            grown = [r for r in victims if r.update_count >= cfg.grace_updates]
            victims = grown or victims
        if cfg.eviction == "lru":
            v = min(victims, key=lambda r: r.last_used_t)
        elif cfg.eviction == "score":
            order = sorted(self.records.values(), key=lambda r: r.last_used_t)
            rank = {r.concept_id: i / max(len(order) - 1, 1) for i, r in enumerate(order)}
            w0, w1, w2 = cfg.evict_weights
            v = min(victims, key=lambda r: (w0 * math.log1p(r.uses) + w1 * rank[r.concept_id] + w2 * float(r.meta.get("perf", 0.5)), r.last_used_t))
        else:
            v = min(victims, key=lambda r: (r.uses, r.last_used_t))
        del self.records[v.concept_id]
        self._dirty = True

    def record_performance(self, cid: str, acc: float, alpha: float = 0.2) -> None:
        """Track how well the concept's model has done (used by the ``score`` eviction policy)."""
        rec = self.records.get(cid)
        if rec is not None and math.isfinite(acc):
            rec.meta["perf"] = float(acc) if "perf" not in rec.meta else (1 - alpha) * rec.meta["perf"] + alpha * float(acc)

    def update(self, cid: str, fp: FeatureFingerprint, t: int, count_use: bool = False, X: np.ndarray | None = None) -> float:
        """Merge a new window into a concept; returns the window-to-concept distance D (before merging).

        legacy merge: weighted averaging of fingerprint entries (std and quantiles are then only approximate).
        mergeable:    the window's summaries are merged exactly and the fingerprint is recomputed; ``X`` (the raw rows) makes
                      the window's summaries exact, otherwise they are rebuilt approximately from ``fp``.
        """
        rec = self.records[cid]
        d = compare(rec.fp, fp, self.sim_cfg, views=("stat", "dist", "time")).combined
        rec.intra = d if rec.intra is None else 0.8 * rec.intra + 0.2 * d
        self._record_same(rec, fp.n, d)
        if self.match_cfg.merge_mode == "mergeable":
            c, h = self.match_cfg.sketch_compression, self.match_cfg.history_windows
            if rec.summaries is None:                                   # a concept loaded from a schema-2.0 dump
                rec.summaries = ConceptSummaries.from_fp(rec.fp, c, h)
                rec.meta["summaries_exact"] = False
            rec.summaries.merge_window(ConceptSummaries.from_window(X, fp, c, h) if X is not None else ConceptSummaries.from_fp(fp, c, h))
            rec.fp = rec.summaries.compose()
        else:
            rec.fp = merge_feature_fp(rec.fp, fp, min(rec.sample_count, 1000), fp.n)
        rec.sample_count += fp.n
        rec.update_count += 1
        rec.last_used_t = t
        if count_use:
            rec.uses += 1
        self._dirty = True
        return d

    def _record_same(self, rec: ConceptRecord, n: int, d: float) -> None:
        rec.var.add(n, d, self.match_cfg.var_window)
        self.pool.append((int(n), float(d)))

    def observe_same(self, cid: str, fp: FeatureFingerprint) -> float:
        """Record the distance between a window known to belong to ``cid`` and the stored fingerprint, WITHOUT merging it.
        Used to calibrate thresholds at the window sizes that queries will have."""
        rec = self.records[cid]
        d = compare(rec.fp, fp, self.sim_cfg, views=("stat", "dist", "time")).combined
        self._record_same(rec, fp.n, d)
        return d

    def thresholds(self, cid: str, n: int) -> dict[str, Any]:
        """Match-range limits for concept ``cid`` for a query window of ``n`` samples.

        legacy:    thr_cand = max(t_cand, k_intra x intra), thr_high = max(t_high, 0.6 k_intra x intra)
        empirical: conformal quantiles (1 - alpha) of the concept's own same-concept distances at similar window sizes; the
                   global pool replaces them for concepts with fewer than ``min_obs`` observations; too little data -> legacy.
        """
        cfg, rec = self.match_cfg, self.records[cid]
        intra = rec.intra or 0.0
        legacy = {"thr_high": max(cfg.t_high, cfg.k_intra * intra * 0.6), "thr_cand": max(cfg.t_cand, cfg.k_intra * intra),
                  "source": "legacy", "n_obs": 0 if rec.intra is None else len(rec.var.obs), "var_ref": intra}
        if cfg.threshold_mode == "legacy":
            return legacy
        vals, src = rec.var.values(n, cfg.size_ratio), "concept"
        if len(vals) < cfg.min_obs:
            vals, src = np.array([d for m, d in self.pool if n / cfg.size_ratio <= m <= n * cfg.size_ratio]), "pool"
            if len(vals) < cfg.min_obs:
                vals = np.array([d for _, d in self.pool])
        if len(vals) < cfg.min_obs:
            legacy["source"] = "legacy_fallback"
            return legacy
        thr_c = float(np.clip(conformal_quantile(vals, 1 - cfg.alpha_cand), cfg.t_min, cfg.t_max))
        thr_h = float(np.clip(conformal_quantile(vals, 1 - cfg.alpha_high), cfg.t_min, thr_c))
        return {"thr_high": thr_h, "thr_cand": thr_c, "source": src, "n_obs": int(len(vals)), "var_ref": float(np.quantile(vals, 0.9))}

    def update_pred(self, cid: str, pf: PredictiveFingerprint) -> None:
        rec = self.records[cid]
        rec.pred = pf if rec.pred is None else merge_predictive_fp(rec.pred, pf, min(rec.pred.n_labeled, 1000), pf.n_labeled)

    def replace_model(self, cid: str, new_model_id: str) -> None:
        """Same concept, different model: concept identity is kept, the old model id goes to history."""
        rec = self.records[cid]
        if rec.model_id and rec.model_id not in rec.model_history:
            rec.model_history.append(rec.model_id)
        rec.model_id = new_model_id
        rec.model_history.append(new_model_id)
        rec.model_versions.append({"model_id": new_model_id, "since_t": rec.last_used_t, "reason": "replaced"})
        rec.pred = None                              # behaviour of the old model no longer describes the new one

    # -- retrieval ----------------------------------------------------------------------------
    def _build_index(self) -> dict:
        """Persistent matrices for vectorised Level-1 retrieval (rebuilt only after the repository changes)."""
        cfg = self.match_cfg
        recs = list(self.records.values())
        if not recs:
            return {"ids": [], "layout": None}
        layout = (recs[0].fp.num_idx, recs[0].fp.cat_idx)
        recs = [r for r in recs if (r.fp.num_idx, r.fp.cat_idx) == layout]
        idx: dict = {"ids": [r.concept_id for r in recs], "layout": layout}
        if cfg.l1_repr == "basic":
            idx["G"] = np.stack([r.fp.global_vector() for r in recs]) if layout[0] else np.zeros((len(recs), 0, 4))
        else:
            idx["loc"] = np.stack([r.fp.statistics["median"] for r in recs]) if layout[0] else np.zeros((len(recs), 0))
            rows = [r.fp.l1_unit(cfg.l1_repr) for r in recs]
            idx["U"] = np.stack([u for u, _ in rows])
            idx["sw"] = rows[0][1]
            idx["sig"] = np.stack([np.maximum(r.fp.global_vector()[:, 1], 1e-9) for r in recs]) if layout[0] else np.zeros((len(recs), 0))
        return idx

    def level1_scores(self, q: FeatureFingerprint) -> tuple[list[str], np.ndarray]:
        """(ids, score) for every compatible stored concept; lower = closer. One vectorised pass."""
        if self._dirty or self._idx is None:
            self._idx, self._dirty = self._build_index(), False
        idx = self._idx
        if not idx["ids"] or idx["layout"] != (q.num_idx, q.cat_idx):
            return [], np.zeros(0)
        if not len(q.num_idx):
            return idx["ids"], np.zeros(len(idx["ids"]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN fingerprints -> inf score below
            if self.match_cfg.l1_repr == "basic":
                G, g = idx["G"], q.global_vector()                   # (C, k, 4), (k, 4)
                s = np.maximum(0.5 * (G[:, :, 1] + g[None, :, 1]), 1e-9)
                loc = np.abs(G[:, :, [0, 2, 3]] - g[None, :, [0, 2, 3]]) / s[:, :, None]
                spread = np.abs(np.log((G[:, :, 1] + 1e-9) / (g[None, :, 1] + 1e-9)))
                score = np.nanmean(np.concatenate([np.clip(loc, 0, 5), np.clip(spread, 0, 3)[:, :, None]], axis=2), axis=(1, 2))
            else:
                u, _ = q.l1_unit(self.match_cfg.l1_repr)
                sc, wt = idx["sw"]
                qsig = np.maximum(q.global_vector()[:, 1], 1e-9)
                s = np.maximum(0.5 * (idx["sig"] + qsig[None, :]), 1e-9)
                dm = np.clip(np.abs(idx["loc"] - q.statistics["median"][None, :]) / s, 0, 5)        # (C, k)
                du = np.clip(np.abs(idx["U"] - u[None, :]) / sc[None, :], 0, 5)                      # (C, u)
                num = 2.0 * np.nansum(dm, axis=1) + np.nansum(du * wt[None, :], axis=1)
                den = 2.0 * np.sum(np.isfinite(dm), axis=1) + np.sum(np.isfinite(du) * wt[None, :], axis=1)
                score = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
        return idx["ids"], np.where(np.isfinite(score), score, np.inf)

    def level1(self, q: FeatureFingerprint, exclude: set[str] | None = None) -> list[str]:
        """Fast retrieval: vectorised distance on a compact representation; returns the candidate concept ids.

        ``top_k_mode="adaptive"`` keeps between ``k_min`` and ``k_max`` candidates: those scoring within
        ``l1_ratio x best + l1_delta`` (ambiguous queries get more candidates, distinctive ones fewer).
        """
        cfg = self.match_cfg
        ids, score = self.level1_scores(q)
        self.last_l1_scores = score
        keep = np.array([i not in (exclude or set()) for i in ids], bool) if ids else np.zeros(0, bool)
        if not keep.any():
            return []
        order = [i for i in np.argsort(score, kind="stable") if keep[i]]
        if cfg.top_k_mode == "adaptive":
            best = score[order[0]]
            cut = best * cfg.l1_ratio + cfg.l1_delta if np.isfinite(best) else np.inf
            k = cfg.k_min
            while k < min(cfg.k_max, len(order)) and score[order[k]] <= cut:
                k += 1
            order = order[:k]
        else:
            order = order[:cfg.top_k]
        return [ids[i] for i in order]

    def score(self, q: FeatureFingerprint, cid: str, observed_pred: PredictiveFingerprint | None = None,
              views=("stat", "dist", "time", "pred")) -> Distances:
        rec = self.records[cid]
        pair = (rec.pred, observed_pred) if rec.pred is not None and observed_pred is not None else None
        return compare(rec.fp, q, self.sim_cfg, pred=pair, views=views)

    def match(self, q: FeatureFingerprint, observed_pred: dict[str, PredictiveFingerprint] | None = None,
              exclude: set[str] | None = None) -> MatchResult:
        """Hierarchical, confidence-aware match of a feature fingerprint against the repository."""
        cfg = self.match_cfg
        comp = q.completeness() if len(q.num_idx) else 1.0
        n_eff = q.effective_n()
        total = len(self.records)
        if q.n < cfg.min_samples:
            return MatchResult(Decision.INSUFFICIENT_EVIDENCE, None, None, None, {}, 0.0,
                               f"only {q.n} samples, need {cfg.min_samples}", n=q.n, completeness=comp, n_eff=n_eff, n_total=total)
        if total == 0:
            return MatchResult(Decision.NEW_CONCEPT, None, None, None, {}, 0.0, "repository is empty", n=q.n, completeness=comp,
                               n_eff=n_eff, n_total=0)
        cands = []
        scored: set[str] = set()
        for cid in self.level1(q, exclude):
            cands.append(Candidate(cid, self.score(q, cid, (observed_pred or {}).get(cid))))
            scored.add(cid)
        if not cands:
            return MatchResult(Decision.NEW_CONCEPT, None, None, None, {}, 0.0, "no compatible stored concept", n=q.n,
                               completeness=comp, n_eff=n_eff, n_total=total)
        cands.sort(key=lambda c: c.D)
        if cfg.expand_on_new and total <= cfg.max_full_scan and cands[0].D > self.thresholds(cands[0].concept_id, q.n)["thr_cand"]:
            # the shortlist says "new concept": make sure Level 1 did not hide the right one
            for cid in self.records:
                if cid not in scored and cid not in (exclude or set()) and (self.records[cid].fp.num_idx, self.records[cid].fp.cat_idx) == (q.num_idx, q.cat_idx):
                    cands.append(Candidate(cid, self.score(q, cid, (observed_pred or {}).get(cid))))
                    scored.add(cid)
            cands.sort(key=lambda c: c.D)
        best = cands[0]
        base = dict(n=q.n, completeness=best.distances.completeness, n_eff=n_eff, n_scored=len(scored), n_total=total,
                    noise=dict(best.distances.noise), has_pred=best.distances.components.get("pred") is not None)
        if best.distances.completeness < cfg.min_completeness:
            return MatchResult(Decision.INSUFFICIENT_EVIDENCE, best.concept_id, best.distances.similarity, best.D,
                               best.distances.components, 0.0,
                               f"only {best.distances.completeness:.0%} of the feature views could be computed", cands, **base)
        th = self.thresholds(best.concept_id, q.n)
        for c in cands[:5]:
            c.thr_cand = self.thresholds(c.concept_id, q.n)["thr_cand"]
        thr_high, thr_cand, var_ref = th["thr_high"], th["thr_cand"], th["var_ref"]
        if len(cands) > 1:
            second = cands[1].D
            margin, margin_norm = second - best.D, (second - best.D) / max(second, 1e-9)
        else:                                      # a single stored concept: distance from the "new concept" boundary instead
            second, margin, margin_norm = 1.0, None, max(0.0, (thr_cand - best.D) / thr_cand)
        sim_term = min(1.0, max(0.0, 1.0 - best.D / thr_cand))
        marg_term = min(1.0, max(0.0, (second - best.D) / cfg.margin_scale))
        unresolved = any(v == "unresolvable" for v in best.distances.noise.values())
        evid_term = min(1.0, n_eff / cfg.n_full) if cfg.threshold_mode == "empirical" else min(1.0, q.n / cfg.n_full)
        evid_term *= (0.5 + 0.5 * best.distances.completeness)
        var_term = 1.0 - 0.5 * min(1.0, var_ref / thr_cand)
        pred_term = 1.0 if base["has_pred"] else cfg.pred_missing_factor
        conf = sim_term * (0.5 + 0.5 * marg_term) * evid_term * var_term * pred_term
        evidence_quality = evid_term * (0.6 if unresolved else 1.0) * pred_term
        terms = {"closeness": sim_term, "separation": 0.5 + 0.5 * marg_term, "evidence": evid_term, "variability": var_term, "predictive": pred_term}
        for c in cands:
            c.confidence = conf if c is best else 0.0
        info = f"D={best.D:.3f}, margin={second - best.D:.3f}, var={var_ref:.3f}, n={q.n}"
        extra = dict(evidence_quality=evidence_quality, confidence_terms=terms, margin=margin, margin_norm=margin_norm, thresholds=th, **base)
        if best.D > thr_cand:
            decision, why = Decision.NEW_CONCEPT, f"best candidate too far ({info}, limit {thr_cand:.3f})"
        elif best.D <= thr_high and (second - best.D) >= cfg.min_margin and conf >= cfg.conf_high:
            decision, why = Decision.HIGH_CONFIDENCE_MATCH, f"close and well separated ({info})"
        else:
            decision = Decision.CANDIDATE_MATCH
            why = ("ambiguous: runner-up is almost as close" if (second - best.D) < cfg.min_margin else
                   "within match range but evidence is not strong enough for high confidence") + f" ({info})"
        res = MatchResult(decision, best.concept_id, best.distances.similarity, best.D, best.distances.components, conf, why, cands, **extra)
        if self.calibrator is not None:
            f = match_features(res)
            if f is not None:
                res.match_probability = float(self.calibrator.predict(f)[0])
        return res

    # -- persistence --------------------------------------------------------------------------
    def to_dict(self) -> dict:
        mergeable = any(r.summaries is not None for r in self.records.values())
        return {"schema_version": RECORD_SCHEMA if mergeable else SCHEMA_VERSION, "next_index": self._next,
                "concepts": [r.to_dict() for r in self.records.values()]}

    def load_dict(self, d: dict) -> None:
        self.records = {}
        self._dirty = True
        for entry in d["concepts"]:
            rec = ConceptRecord.from_dict(entry)
            self.records[rec.concept_id] = rec
        self._next = int(d.get("next_index", 1 + max([r.index for r in self.records.values()], default=0)))

    def memory_per_concept(self) -> float:
        return float(np.mean([r.nbytes() for r in self.records.values()])) if self.records else 0.0
