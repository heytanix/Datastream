"""Wires the six layers together around the bus.

  L1 source    -> producer thread publishes ds.raw through a Sender
  L2 bus       -> topics ds.raw, ds.events, ds.metrics
  L3/L4        -> one consumer group per strategy (proc-<name>) runs an Engine
  L5 storage   -> consumer group `store` writes events and metrics to SQLite
  L6 dashboard -> consumer group `dashboard` keeps the live state the web UI polls
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from .bus import Bus
from .strategies import STRATEGIES, make_engine
from .store import Store
from .stream import ConceptStream

METRIC_EVERY = 50
ROLL = 200


class Pipeline:
    def __init__(self, rate: int = 600, seed: int = 7, loops: int = 0, seg_len: int = 1500,
                 db_path: str = "results/datastream.db", stream=None, n_features: int = 4):
        self.rate, self.loops = rate, loops
        self.stream = stream if stream is not None else ConceptStream(seed=seed, seg_len=seg_len, loops=loops)
        self.n_features = n_features
        self.bus = Bus()
        for t in ("ds.raw", "ds.events", "ds.metrics"):
            self.bus.create_topic(t, retention=100_000 if t == "ds.raw" else 20_000)
        self.store = Store(db_path)
        self.engines = {s: make_engine(s, n_features=n_features) for s in STRATEGIES}
        self.stop = threading.Event()
        self.threads: list[threading.Thread] = []
        self.lock = threading.Lock()
        self.reset_state()

    def reset_state(self):
        self.state = {
            "t": 0, "rate": 0.0,
            "acc": {s: deque(maxlen=800) for s in STRATEGIES},
            "total": {s: [0, 0] for s in STRATEGIES},          # [correct, seen]
            "models": {s: 0 for s in STRATEGIES},
            "segments": deque(maxlen=60),                      # true concept timeline [(t, name)]
            "active": {s: deque(maxlen=60) for s in STRATEGIES},  # active concept timeline per strategy
            "events": deque(maxlen=40),
            "counts": {s: defaultdict(int) for s in STRATEGIES},
            "repo": [], "repo_mv": [], "mv": {},
        }

    # -- threads ------------------------------------------------------------
    def start(self):
        for target, name in [(self._produce, "source"), (self._store, "store"), (self._dash, "dashboard")]:
            self._spawn(target, name)
        for s in STRATEGIES:
            self._spawn(lambda s=s: self._process(s), f"proc-{s}")

    def _spawn(self, fn, name):
        th = threading.Thread(target=fn, daemon=True, name=name)
        th.start()
        self.threads.append(th)

    def close(self):
        self.stop.set()
        self.store.commit()

    def _produce(self):
        send = self.bus.sender(capacity=20_000)
        tick, batch = time.time(), max(1, self.rate // 100)
        n = 0
        for t, x, y, c in self.stream:
            if self.stop.is_set():
                break
            send.send("ds.raw", {"t": t, "x": x, "y": y, "concept": c})
            n += 1
            if n % batch == 0:
                tick += batch / self.rate
                delay = tick - time.time()
                if delay > 0:
                    time.sleep(delay)
                else:
                    tick = time.time()
        send.close()

    def _process(self, strategy: str):
        eng = self.engines[strategy]
        cons = self.bus.consumer(f"proc-{strategy}", ["ds.raw"])
        send = self.bus.sender()
        hits, k = 0, 0
        while not self.stop.is_set():
            for rec in cons.poll(0.2):
                r = rec.value
                pred, events = eng.process(r["t"], r["x"], r["y"])
                hits += int(pred == r["y"])
                k += 1
                for e in events:
                    send.send("ds.events", e)
                if k == METRIC_EVERY:
                    act = eng.active
                    send.send("ds.metrics", {"t": r["t"], "strategy": strategy, "acc": hits / k,
                                             "models": eng.models_trained, "state": eng.state,
                                             "active": None if act is None else getattr(act, "cid", getattr(act, "index", None)),
                                             "repo": eng.snapshot(),
                                             "mv": eng.dash_info() if hasattr(eng, "dash_info") else None})
                    hits, k = 0, 0

    def _store(self):
        cons = self.bus.consumer("store", ["ds.events", "ds.metrics"])
        last = time.time()
        while not self.stop.is_set():
            for rec in cons.poll(0.2):
                if rec.topic == "ds.events":
                    self.store.add_event(rec.value)
                else:
                    self.store.add_metric(rec.value)
                    if rec.value["strategy"] == "datastream" and rec.value["t"] % 500 < METRIC_EVERY:
                        self.store.save_concepts("datastream", rec.value["repo"])
            if time.time() - last > 1:
                self.store.commit()
                last = time.time()

    def _dash(self):
        cons = self.bus.consumer("dashboard", ["ds.raw", "ds.events", "ds.metrics"])
        s = self.state
        mark, last_c, mt = time.time(), None, 0
        while not self.stop.is_set():
            recs = cons.poll(0.2, 1000)
            with self.lock:
                for rec in recs:
                    v = rec.value
                    if rec.topic == "ds.raw":
                        s["t"] = v["t"]
                        mt += 1
                        if v["concept"] != last_c:
                            s["segments"].append((v["t"], v["concept"]))
                            last_c = v["concept"]
                    elif rec.topic == "ds.events":
                        s["counts"][v["strategy"]][v["kind"]] += 1
                        s["events"].appendleft(v)
                        if v["kind"] in ("new_concept", "recall", "probe", "false_alarm", "provisional", "rollback"):
                            s["active"][v["strategy"]].append((v["t"], v["concept"], v["kind"]))
                    else:
                        st = v["strategy"]
                        s["acc"][st].append((v["t"], round(v["acc"], 3)))
                        tot = s["total"][st]
                        tot[0] += v["acc"] * METRIC_EVERY
                        tot[1] += METRIC_EVERY
                        s["models"][st] = v["models"]
                        if st == "datastream":
                            s["repo"] = v["repo"]
                        elif st == "datastream_mv":
                            s["repo_mv"], s["mv"] = v["repo"], v["mv"]
                now = time.time()
                if now - mark >= 1:
                    s["rate"] = mt / (now - mark)
                    mark, mt = now, 0

    # -- read side ----------------------------------------------------------
    def snapshot(self) -> dict:
        with self.lock:
            s = self.state
            summary = {k: {"acc": (c / n if n else 0), "seen": n, "models": s["models"][k],
                           "counts": dict(s["counts"][k])} for k, (c, n) in s["total"].items()}
            out = {"t": s["t"], "rate": round(s["rate"]), "acc": {k: list(v) for k, v in s["acc"].items()},
                   "summary": summary, "segments": list(s["segments"]),
                   "active": {k: list(v) for k, v in s["active"].items()},
                   "events": [e for e in list(s["events"]) if e["kind"] != "match"][:14],
                   "repo": s["repo"], "repo_mv": s["repo_mv"], "mv": s["mv"]}
        out["bus"] = self.bus.stats()
        return out
