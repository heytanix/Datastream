"""Layers 3+4: Datastream with multi-view fingerprints and safe model recall.

Per sample: predict with the active model, learn online (30 samples late), and watch for drift
with a Page-Hinkley error test (needs labels) and a label-free feature-shift test (needs X only).

On drift:  collect post-drift observations  ->  feature-only fingerprint  ->  hierarchical match
  HIGH / CANDIDATE   shadow-validate the best candidate model(s) on labelled post-drift samples;
                     HIGH confidence may be activated provisionally while labels are still arriving
                     and is rolled back if validation fails
  NEW_CONCEPT        train a model once enough labels exist and store a new concept
  INSUFFICIENT       keep collecting
Labels may arrive late (``label_delay``) or never (``y=None``): X-only evidence is used immediately,
label-dependent steps wait for released labels.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np

from .detector import PageHinkley
from .matcher import ConceptRepository, Decision, MatchConfig, MatchResult
from .models import LogReg
from .mvfp import PredictiveAccumulator, extract_feature_fp
from .similarity import SimilarityConfig
from .validation import METRICS, Pred, ValidationConfig, compare_paired, select_candidate


@dataclass(frozen=True)
class RecallPolicy:
    """Acceptance criteria for reusing a stored model."""
    val_min: int = 30             # labelled post-drift samples needed before a verdict
    min_acc: float = 0.70         # candidate accuracy on the validation window
    min_gain: float = 0.0         # candidate must beat the previously active model by at least this
    max_pred_dist: float = 0.6    # D_pred ceiling when a stored predictive fingerprint exists
    max_shadow: int = 3           # candidates validated in parallel
    provisional: bool = True      # serve a HIGH_CONFIDENCE candidate while validation is pending
    mode: str = "point"           # "point": thresholds on point estimates (schema 2.0) | "paired": statistical comparison (validation.py)
                                  # | "none": accept the fingerprint match without any validation (ablation only; unsafe)
    validation: ValidationConfig | None = None   # used when mode == "paired"; default ValidationConfig(min_samples=val_min)

    def __post_init__(self):
        if self.mode not in ("point", "paired", "none"):
            raise ValueError("mode must be point, paired or none")
        if self.validation is not None and self.validation.task != "classification":
            raise ValueError("the engine's learner is a classifier; regression/anomaly validation is available in validation.py only")


@dataclass
class _Stats:
    n: int = 0
    correct: int = 0
    acc_obj: PredictiveAccumulator = field(default_factory=PredictiveAccumulator)
    ys: list = field(default_factory=list)
    preds: list = field(default_factory=list)
    scores: list = field(default_factory=list)

    @property
    def acc(self) -> float:
        return self.correct / self.n if self.n else float("nan")

    def add(self, y: int, pred: int, proba: float) -> None:
        self.n += 1
        self.correct += int(pred == y)
        self.acc_obj.add(y, pred, proba)
        self.ys.append(y)
        self.preds.append(pred)
        self.scores.append(proba)


@dataclass
class _Recovery:
    prev_id: str
    prev_model: LogReg
    X: list = field(default_factory=list)
    t: list = field(default_factory=list)
    lab: list = field(default_factory=list)            # released (x, y) pairs
    stage: str = "collect"                             # collect | validating | await_train
    last_try: int = 0
    fp: object = None
    cands: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    baseline: _Stats = field(default_factory=_Stats)
    provisional: str | None = None
    result: MatchResult | None = None
    chosen: str | None = None                          # split_confirm: candidate picked on the selection samples
    sel_n: int = 0                                     # ... and how many labelled samples were used for that selection


def _f(v, nd=4):
    return None if v is None else round(float(v), nd)


class MVEngine:
    strategy = "datastream_mv"

    def __init__(self, n_features: int = 4, *, warmup: int = 150, window: int = 100, buffer_min: int = 40,
                 train_min: int = 80, max_wait: int = 200, quarantine: int = 30, label_delay: int = 0,
                 shift_threshold: float = 0.25, categorical: tuple = (), match: MatchConfig | None = None,
                 sim: SimilarityConfig | None = None, recall: RecallPolicy | None = None):
        self.n_features, self.categorical = n_features, tuple(categorical)
        self.warmup, self.window, self.buffer_min, self.train_min = warmup, window, buffer_min, train_min
        self.max_wait, self.quarantine, self.label_delay = max_wait, quarantine, label_delay
        self.shift_threshold = shift_threshold
        self.policy = recall or RecallPolicy()
        self.repo = ConceptRepository(match, sim)
        self.models: dict[str, LogReg] = {}               # fingerprints live in the repo, models here
        self.detector = PageHinkley()
        self.state = "WARMUP"
        self.active_id: str | None = None
        self.model: LogReg | None = None
        self.models_trained = 0
        self._next_model = 1
        self._delay: deque = deque()
        self._wu_X: list = []
        self._wu_lab: list = []
        self.win_X: deque = deque(maxlen=window)
        self.win_t: deque = deque(maxlen=window)
        self._since = 0
        self.pending: deque = deque()
        self.pred_acc = PredictiveAccumulator(maxlen=window)
        self.rec: _Recovery | None = None
        self.log: deque = deque(maxlen=40)                # recent decisions for the dashboard
        self.fp_seconds = 0.0
        self.fp_calls = 0
        self.match_seconds = 0.0
        self.match_calls = 0
        self.t = 0

    # -- helpers --------------------------------------------------------------------------------
    @property
    def active(self):                                      # legacy-dashboard compatibility
        return self.repo.records.get(self.active_id) if self.active_id else None

    def _fp(self, X, t):
        import time
        t0 = time.perf_counter()
        fp = extract_feature_fp(np.asarray(X), t=np.asarray(t), categorical=self.categorical)
        self.fp_seconds += time.perf_counter() - t0
        self.fp_calls += 1
        return fp

    def _new_model(self, X, y) -> tuple[str, LogReg]:
        mid = f"model_{self._next_model:03d}"
        self._next_model += 1
        self.models_trained += 1
        m = LogReg(self.n_features).fit(np.asarray(X), np.asarray(y))
        self.models[mid] = m
        return mid, m

    def _ev(self, kind: str, cid: str | None, dist: float = 0.0, **extra) -> dict:
        idx = int(cid.rsplit("_", 1)[1]) if cid else None
        e = {"t": int(self.t), "strategy": self.strategy, "kind": kind, "concept": idx, "dist": _f(dist, 3), **extra}
        self.log.appendleft(e)
        return e

    def _resume(self, t: int) -> None:
        self.state = "STABLE"
        self.rec = None
        self.detector.reset()
        self.pending.clear()
        self.win_X.clear()
        self.win_t.clear()
        self._since = 0
        self.pred_acc.clear()

    # -- main entry -----------------------------------------------------------------------------
    def process(self, t: int, x: np.ndarray, y: int | None):
        """Prequential step. ``y`` is the true label (None = never labelled). Returns (prediction, events)."""
        self.t = t
        x = np.asarray(x, float)
        if self.model is not None:
            proba = self.model.proba(x)
            pred = int(proba > 0.5)
        else:
            pred, proba = 0, 0.5
        self._delay.append((t, x, y, pred, proba))
        released = []
        while self._delay and t - self._delay[0][0] >= self.label_delay:
            item = self._delay.popleft()
            if item[2] is not None:
                released.append(item)
        events: list[dict] = []
        if self.state == "WARMUP":
            self._wu_X.append((t, x))
            self._wu_lab.extend((r[1], r[2]) for r in released)
            if len(self._wu_lab) >= self.warmup:
                mid, m = self._new_model([a for a, _ in self._wu_lab], [b for _, b in self._wu_lab])
                fp = self._fp([v for _, v in self._wu_X], [k for k, _ in self._wu_X])
                rec = self.repo.add(fp, mid, t, X=np.array([v for _, v in self._wu_X]))
                self.active_id, self.model = rec.concept_id, m
                self._wu_X, self._wu_lab = [], []
                self._resume(t)
                events.append(self._ev("new_concept", rec.concept_id, 0.0, reason="warm-up"))
        elif self.state == "STABLE":
            events.extend(self._stable(t, x, released))
        else:
            events.extend(self._recovering(t, x, released))
        return pred, events

    # -- stable phase ---------------------------------------------------------------------------
    def _stable(self, t, x, released) -> list[dict]:
        self.win_X.append(x)
        self.win_t.append(t)
        self._since += 1
        why = None
        for (_, xx, yy, pp, pr) in released:
            self.pred_acc.add(yy, pp, pr)
            self.pending.append((xx, yy))
            if len(self.pending) > self.quarantine:
                self.model.partial_fit(*self.pending.popleft())
            if self.detector.update(int(pp != yy)) and why is None:
                why = "error"
        if why is None and self._since >= self.window and len(self.win_X) == self.window:
            self._since = 0
            fp = self._fp(self.win_X, self.win_t)
            d = self.repo.score(fp, self.active_id, views=("stat", "dist", "time"))
            if d.combined > self.shift_threshold:
                why = "feature_shift"
            else:
                if self.repo.match_cfg.threshold_mode == "empirical":     # same-concept distances at the query window size
                    sub = list(self.win_X)[: self.buffer_min]
                    self.repo.observe_same(self.active_id, self._fp(sub, list(self.win_t)[: self.buffer_min]))
                self.repo.update(self.active_id, fp, t, X=np.array(self.win_X))
                pf = self.pred_acc.fingerprint()
                if pf is not None and pf.n_labeled >= 20:
                    self.repo.update_pred(self.active_id, pf)
                    self.repo.record_performance(self.active_id, pf.accuracy)
        if why is None:
            return []
        self.pending.clear()
        self.rec = _Recovery(self.active_id, self.model)
        self.state = "RECOVERING"
        return [self._ev("drift", self.active_id, 0.0, why=why)]

    # -- recovery -------------------------------------------------------------------------------
    def _recovering(self, t, x, released) -> list[dict]:
        r = self.rec
        r.X.append(x)
        r.t.append(t)
        r.lab.extend((rr[1], rr[2]) for rr in released)
        ev: list[dict] = []
        if r.stage in ("validating", "confirming"):
            for rr in released:
                self._feed(r, rr[1], rr[2])
        if r.stage == "collect" and len(r.X) >= self.buffer_min and (r.last_try == 0 or len(r.X) - r.last_try >= 10):
            r.last_try = len(r.X)
            ev.extend(self._try_match(t))
        if self.rec is not None and r.stage == "await_train" and len(r.lab) >= self.train_min:
            ev.extend(self._train_new(t, "no stored concept matches"))
        elif self.rec is not None and r.stage == "validating" and r.baseline.n >= self.policy.val_min:
            ev.extend(self._resolve(t))
        elif self.rec is not None and r.stage == "confirming" and r.baseline.n - r.sel_n >= self._vcfg().confirm_min:
            ev.extend(self._resolve(t))
        return ev

    def _try_match(self, t) -> list[dict]:
        r = self.rec
        fp = self._fp(r.X, r.t)
        import time as _time
        t0 = _time.perf_counter()
        res = self.repo.match(fp)
        self.match_seconds += _time.perf_counter() - t0
        self.match_calls += 1
        r.fp, r.result = fp, res
        ev = []
        if res.decision != Decision.INSUFFICIENT_EVIDENCE or len(r.X) == self.buffer_min:
            ev.append(self._ev("match", res.best_id, res.D or 0.0, **res.as_dict(top=5), explanation=res.explanation()))
        d = res.decision
        if d == Decision.INSUFFICIENT_EVIDENCE:
            if len(r.X) >= self.max_wait:
                r.stage = "await_train"
            return ev
        if d == Decision.NEW_CONCEPT:
            r.stage = "await_train"
            return ev
        # the active concept is a candidate like any other: "nothing changed" must pass validation too
        cands = [c.concept_id for i, c in enumerate(res.ranking)
                 if i == 0 or c.D <= self.repo.thresholds(c.concept_id, res.n)["thr_cand"]]
        r.cands = cands[: self.policy.max_shadow]                 # the best match is always validated
        r.stats = {c: _Stats() for c in r.cands}
        r.stage = "validating"
        for xx, yy in r.lab:                            # labelled post-drift samples that already exist
            self._feed(r, xx, yy)
        if (d == Decision.HIGH_CONFIDENCE_MATCH and self.policy.provisional and r.cands[0] != r.prev_id
                and r.baseline.n < self.policy.val_min):
            r.provisional = r.cands[0]
            self.model = self.models[self.repo.records[r.provisional].model_id]
            ev.append(self._ev("provisional", r.provisional, res.D or 0.0, confidence=_f(res.confidence, 3)))
        if self.policy.mode == "none" or r.baseline.n >= self.policy.val_min:
            ev.extend(self._resolve(t))
        return ev

    def _feed(self, r: _Recovery, x, y) -> None:
        for c in r.cands:
            m = self.models[self.repo.records[c].model_id]
            r.stats[c].add(y, m.predict(x), m.proba(x))
        r.baseline.add(y, r.prev_model.predict(x), r.prev_model.proba(x))

    def _vcfg(self) -> ValidationConfig:
        return self.policy.validation or ValidationConfig(min_samples=self.policy.val_min)

    def _resolve(self, t) -> list[dict]:
        r, pol = self.rec, self.policy
        if not r.cands:                                      # defensive: nothing to validate -> learn from scratch
            r.stage = "await_train"
            return []
        if pol.mode == "none":                               # no validation at all: trust the fingerprint
            c = r.cands[0]
            d = self.repo.score(r.fp, c, None, views=("stat", "dist", "time"))
            return self._finish(t, (c, float("nan"), d), [{"concept": int(c.rsplit("_", 1)[1]), "accuracy": None, "n": 0, "accepted": True}], {"mode": "none"})
        if pol.mode == "paired":
            return self._resolve_paired(t)
        best, verdicts = None, []
        for c in r.cands:
            st = r.stats[c]
            d = self.repo.score(r.fp, c, st.acc_obj.fingerprint())
            ok = (st.acc >= pol.min_acc and st.acc >= r.baseline.acc + pol.min_gain and
                  (d.components.get("pred") is None or d.components["pred"] <= pol.max_pred_dist) and
                  d.combined <= self.repo.thresholds(c, len(r.X))["thr_cand"])
            verdicts.append({"concept": int(c.rsplit("_", 1)[1]), "accuracy": _f(st.acc), "n": st.n, "accepted": bool(ok),
                             "D_with_pred": _f(d.combined), "D_pred": _f(d.components.get("pred"))})
            if ok and (best is None or st.acc > best[1]):
                best = (c, st.acc, d)
        return self._finish(t, best, verdicts, {"min_acc": pol.min_acc, "min_gain": pol.min_gain, "val_min": pol.val_min})

    def _resolve_paired(self, t) -> list[dict]:
        """Paired, uncertainty-aware validation (see validation.py). The active model is protected: when it still
        validates, a replacement must be statistically superior, not merely non-inferior."""
        r, pol, vc = self.rec, self.policy, self._vcfg()
        ys = np.array(r.baseline.ys)
        base = Pred(np.array(r.baseline.preds), np.array(r.baseline.scores))
        cand = {c: Pred(np.array(r.stats[c].preds), np.array(r.stats[c].scores)) for c in r.cands}
        f = METRICS[vc.task][vc.metric]
        sub = lambda p, sl: Pred(p.pred[sl], None if p.score is None else p.score[sl])
        comps: dict = {}
        if vc.selection == "split_confirm":
            if r.stage != "confirming":                       # stage 1: choose on the samples collected so far
                sc = {c: f(ys, cand[c].pred, cand[c].score) for c in r.cands}
                ok = [c for c in r.cands if np.isfinite(sc[c])]
                if ok:
                    r.chosen, r.sel_n, r.stage = max(ok, key=sc.get), len(ys), "confirming"
                    return [self._ev("selection", r.chosen, 0.0, metric=vc.metric, scores={int(c.rsplit('_', 1)[1]): _f(v) for c, v in sc.items()},
                                     fresh_labels_needed=vc.confirm_min, n_selection=len(ys))]
            else:                                              # stage 2: confirm on labels the selection never saw
                sl = slice(r.sel_n, None)
                comps = {r.chosen: compare_paired(ys[sl], sub(cand[r.chosen], sl), sub(base, sl), vc)}
        else:
            others = {c: p for c, p in cand.items() if c != r.prev_id}
            comps = dict(select_candidate(ys, others, base, vc).comparisons) if others else {}
            if r.prev_id in cand:                              # the unchanged model is judged for non-inferiority only
                comps[r.prev_id] = compare_paired(ys, cand[r.prev_id], base, replace(vc, objective="non_inferiority"), rng=np.random.default_rng(vc.seed))
        verdicts, accepted = [], []
        for c, cmp in comps.items():
            d = self.repo.score(r.fp, c, r.stats[c].acc_obj.fingerprint())
            gate = (d.components.get("pred") is None or d.components["pred"] <= self.policy.max_pred_dist) and d.combined <= self.repo.thresholds(c, len(r.X))["thr_cand"]
            ok = cmp.verdict == "accept" and gate
            verdicts.append({"concept": int(c.rsplit("_", 1)[1]), "accuracy": _f(cmp.cand), "n": cmp.n, "accepted": bool(ok),
                             "D_with_pred": _f(d.combined), "D_pred": _f(d.components.get("pred")), "comparison": cmp.as_dict()})
            if ok:
                accepted.append((c, cmp.cand, d, cmp))
        best = None
        if accepted:
            prev_ok = next((a for a in accepted if a[0] == r.prev_id), None)
            if prev_ok is not None:                            # protect the active model unless a replacement is superior
                better = [a for a in accepted if a[0] != r.prev_id and a[3].ci_low > 0.0]
                best = max(better, key=lambda a: a[1]) if better else prev_ok
            else:
                best = max(accepted, key=lambda a: a[1])
            best = best[:3]
        crit = {"mode": "paired", "metric": vc.metric, "objective": vc.objective, "alpha": vc.alpha, "margin": vc.margin,
                "selection": vc.selection, "floor": vc.floor, "val_min": vc.min_samples}
        return self._finish(t, best, verdicts, crit)

    def _finish(self, t, best, verdicts, criteria) -> list[dict]:
        r = self.rec
        ev = [self._ev("validation", best[0] if best else r.cands[0], 0.0, baseline_acc=_f(r.baseline.acc),
                       baseline_n=r.baseline.n, candidates=verdicts, accepted=bool(best), criteria=criteria)]
        if best and best[0] == r.prev_id:               # the stale model still works: it was a false alarm
            self.repo.update(r.prev_id, r.fp, t, X=np.array(r.X))
            self._resume(t)
            ev.append(self._ev("false_alarm", r.prev_id, best[2].combined, confidence=_f(r.result.confidence, 3),
                               val_acc=_f(best[1])))
            return ev
        if best:
            cid, acc, d = best
            rec = self.repo.records[cid]
            self.repo.update(cid, r.fp, t, count_use=True, X=np.array(r.X))
            self.active_id, self.model = cid, self.models[rec.model_id]
            self._resume(t)
            ev.append(self._ev("recall", cid, d.combined, confidence=_f(r.result.confidence, 3), val_acc=_f(acc),
                               baseline_acc=_f(r.baseline.acc), provisional=r.provisional == cid))
            return ev
        if r.provisional is not None:
            self.model = r.prev_model
            ev.append(self._ev("rollback", r.prev_id, 0.0, rejected=int(r.provisional.rsplit("_", 1)[1]),
                               reason="shadow validation failed"))
            r.provisional = None
        r.stage = "await_train"
        if len(r.lab) >= self.train_min:
            ev.extend(self._train_new(t, "validation failed"))
        return ev

    def _train_new(self, t, why) -> list[dict]:
        r = self.rec
        X = np.array([a for a, _ in r.lab][-max(self.train_min, 1):]) if len(r.lab) >= self.train_min else np.array([a for a, _ in r.lab])
        y = np.array([b for _, b in r.lab][-len(X):])
        fp = self._fp(r.X, r.t)
        mid, m = self._new_model(X, y)
        rec = self.repo.add(fp, mid, t, protect=r.prev_id, meta={"created_because": why}, X=np.array(r.X))
        self.active_id, self.model = rec.concept_id, m
        self._resume(t)
        return [self._ev("new_concept", rec.concept_id, r.result.D if r.result and r.result.D else 0.0, reason=why)]

    # -- dashboard ------------------------------------------------------------------------------
    def snapshot(self) -> list[dict]:
        recs = list(self.repo.records.values())
        if not recs:
            return []
        G = np.stack([np.nan_to_num(r.fp.global_vector().ravel()) for r in recs])
        z = (G - G.mean(0)) / (G.std(0) + 0.05) if len(recs) > 1 else np.zeros_like(G)
        return [{"id": r.index, "created_t": r.created_t, "uses": r.uses, "d_in": _f(r.intra or 0.0, 3),
                 "active": r.concept_id == self.active_id, "z": [round(float(v), 2) for v in z[i]],
                 "n": r.sample_count, "model": r.model_id, "pred": None if r.pred is None else _f(r.pred.error_rate, 3)}
                for i, r in enumerate(recs)]

    def dash_info(self) -> dict:
        return {"log": list(self.log)[:12], "fp_ms": 1000 * self.fp_seconds / max(1, self.fp_calls),
                "state": self.state, "concepts": len(self.repo), "bytes_per_concept": self.repo.memory_per_concept()}
