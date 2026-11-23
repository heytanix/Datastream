"""Layers 3+4: stream processor logic. One engine = one adaptation strategy.

strategies
  static       train once on the first window, never adapt
  retrain      on drift, train a fresh model on the post-drift buffer (no memory)
  dejastream   on drift, fingerprint the post-drift buffer and look it up in the concept
               repository: recall the stored model on a match, train and store a new
               concept otherwise
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .detector import PageHinkley
from .fingerprint import Scaler, fingerprint
from .models import LogReg
from .stream import D

STRATEGIES = ("static", "retrain", "dejastream")


@dataclass
class StoredConcept:
    cid: int
    fp: np.ndarray
    model: LogReg
    created_t: int
    uses: int = 1
    d_in: float = 0.0
    history: list = field(default_factory=list)


class Engine:
    def __init__(self, strategy: str, warmup: int = 150, buffer: int = 80, window: int = 100,
                 match_ratio: float = 3.0, min_match: float = 0.35, fp_shift: float = 1.6,
                 min_model_acc: float = 0.75, probe: int = 30, probe_acc: float = 0.9,
                 quarantine: int = 30, n_features: int = D):
        assert strategy in STRATEGIES
        self.strategy = strategy
        self.warmup, self.buffer_n, self.window = warmup, buffer, window
        self.match_ratio, self.min_match, self.fp_shift = match_ratio, min_match, fp_shift
        self.min_model_acc = min_model_acc
        self.probe_n, self.probe_acc = probe, probe_acc
        self.provisional: StoredConcept | None = None
        # online updates wait here so a change that is not yet detected cannot pollute a stored model
        self.pending: deque = deque()
        self.quarantine = quarantine
        self.detector = PageHinkley()
        self.n_features = n_features
        self.scaler = Scaler(4 * n_features + 1)
        self.repo: dict[int, StoredConcept] = {}
        self.active: StoredConcept | None = None
        self.model: LogReg | None = None
        self.models_trained = 0
        self.state = "WARMUP"
        self.buf_X: list = []
        self.buf_y: list = []
        self.win_X: deque = deque(maxlen=window)
        self.win_y: deque = deque(maxlen=window)
        self._since_win = 0
        self._fp_streak = 0
        self.t = 0

    # -- helpers ------------------------------------------------------------
    def _train_concept(self, X, y, t) -> StoredConcept:
        m = LogReg(self.n_features).fit(X, y)
        self.models_trained += 1
        fp = fingerprint(X, y)
        self.scaler.update(fp)
        c = StoredConcept(len(self.repo), fp, m, t)
        self.repo[c.cid] = c
        return c

    def _activate(self, c: StoredConcept):
        self.active, self.model = c, c.model
        self.detector.reset()
        self.pending.clear()
        self.win_X.clear()
        self.win_y.clear()
        self._since_win = 0
        self._fp_streak = 0

    def _match(self, fp):
        best, bd = None, 1e9
        for c in self.repo.values():
            d = self.scaler.distance(fp, c.fp)
            if d < bd:
                best, bd = c, d
        return best, bd

    def _threshold(self, c: StoredConcept) -> float:
        return max(self.min_match, self.match_ratio * c.d_in)

    # -- main entry ---------------------------------------------------------
    def process(self, t: int, x: np.ndarray, y: int):
        """Prequential step: predict, then learn. Returns (prediction, [events])."""
        self.t = t
        events: list[dict] = []

        if self.state == "WARMUP":
            self.buf_X.append(x), self.buf_y.append(y)
            pred = self.model.predict(x) if self.model else 0
            if len(self.buf_X) >= self.warmup:
                c = self._train_concept(np.array(self.buf_X), np.array(self.buf_y), t)
                self._activate(c)
                self.state = "STABLE"
                self.buf_X, self.buf_y = [], []
                events.append(self._ev("new_concept", c, 0.0))
            return pred, events

        pred = self.model.predict(x)
        err = int(pred != y)

        if self.state == "STABLE":
            if self.strategy != "static":
                self.pending.append((x, y))
                if len(self.pending) > self.quarantine:
                    self.model.partial_fit(*self.pending.popleft())
                self.win_X.append(x), self.win_y.append(y)
                self._since_win += 1
                drift, why = self.detector.update(err), "error"
                if not drift and self.strategy == "dejastream" and self._since_win >= self.window \
                        and len(self.win_X) == self.window:
                    self._since_win = 0
                    fp = fingerprint(np.array(self.win_X), np.array(self.win_y))
                    d = self.scaler.distance(fp, self.active.fp)
                    if d > max(self.fp_shift, 4 * self.active.d_in):
                        self._fp_streak += 1
                    else:
                        self._fp_streak = 0
                        self._refine(fp, d)
                    if self._fp_streak >= 1:
                        drift, why = True, "fingerprint"
                if drift:
                    self.state = "RECOVERING"
                    self.pending.clear()
                    self.buf_X, self.buf_y = [], []
                    events.append(self._ev("drift", self.active, 0.0, why=why))
        else:  # RECOVERING: keep predicting with the stale model while the buffer fills
            self.buf_X.append(x), self.buf_y.append(y)
            if self.strategy == "dejastream" and len(self.buf_X) == self.probe_n:
                events.extend(self._probe())
            if len(self.buf_X) >= self.buffer_n:
                events.extend(self._recover(t))
        return pred, events

    def _refine(self, fp, d):
        """Stable window: slowly pull the stored fingerprint towards the live one."""
        c = self.active
        c.d_in = d if c.d_in == 0.0 else 0.8 * c.d_in + 0.2 * d
        c.fp = 0.9 * c.fp + 0.1 * fp
        self.scaler.update(fp)

    def _probe(self):
        """Early pattern match: does a stored model already explain the first few post-drift samples?"""
        X, y = np.array(self.buf_X), np.array(self.buf_y)
        best, acc = None, 0.0
        for c in self.repo.values():
            if c is self.active:
                continue
            a = c.model.accuracy(X, y)
            if a > acc:
                best, acc = c, a
        if best is not None and acc >= self.probe_acc:
            self.provisional, self.model = best, best.model
            return [self._ev("probe", best, 1.0 - acc)]
        return []

    def _recover(self, t):
        X, y = np.array(self.buf_X), np.array(self.buf_y)
        self.buf_X, self.buf_y = [], []
        self.state = "STABLE"
        ev = []
        prov, self.provisional = self.provisional, None
        if self.strategy == "retrain":
            c = self._train_concept(X, y, t)
            self._activate(c)
            ev.append(self._ev("new_concept", c, 0.0))
            return ev
        fp = fingerprint(X, y)
        best, d = self._match(fp)
        if prov is not None and best is not prov and self._threshold(prov) >= self.scaler.distance(fp, prov.fp):
            best, d = prov, self.scaler.distance(fp, prov.fp)
        ok = best is not None and d <= self._threshold(best) and best.model.accuracy(X, y) >= self.min_model_acc
        if ok and best is self.active:
            self._activate(best)
            ev.append(self._ev("false_alarm", best, d))
        elif ok:
            best.uses += 1
            best.history.append(t)
            self._activate(best)
            ev.append(self._ev("recall", best, d))
        else:
            c = self._train_concept(X, y, t)
            self._activate(c)
            ev.append(self._ev("new_concept", c, d))
        return ev

    def _ev(self, kind, c, dist, **extra):
        return {"t": self.t, "strategy": self.strategy, "kind": kind, "concept": c.cid,
                "dist": round(float(dist), 3), **extra}

    def snapshot(self) -> list[dict]:
        names = None
        out = []
        for c in self.repo.values():
            z = (c.fp - self.scaler.mean) / self.scaler.std
            out.append({"id": c.cid, "created_t": c.created_t, "uses": c.uses, "d_in": round(c.d_in, 3),
                        "active": c is self.active, "z": [round(float(v), 2) for v in z]})
        return out
