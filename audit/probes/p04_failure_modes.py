"""Engine / pipeline failure-mode probes. All faults are injected by subclassing/monkeypatching IN MEMORY; source files are never touched."""
import os, sys, time, threading, tempfile, hashlib, io, contextlib
sys.path.insert(0, os.getcwd())
import numpy as np
from dejastream.mv_engine import MVEngine, RecallPolicy
from dejastream.matcher import MatchConfig
from dejastream.scenarios import abrupt_recurring
from dejastream.research.gen import Spec, sample
sc = abrupt_recurring(seed=1)

def run(e, data, mask_after=None, limit=None):
    ev = []
    for t, x, y, _ in data:
        if limit and t >= limit: break
        _, es = e.process(t, x, None if (mask_after is not None and t >= mask_after) else y); ev += es
    return ev

print("== A. labels stop arriving after t=3200 (y=None): can the engine get stuck in RECOVERING?")
e = MVEngine(); ev = run(e, sc, mask_after=3200)
kinds = [x["kind"] for x in ev if x["t"] > 3200]
print(f"   state at end={e.state} stage={e.rec.stage if e.rec else None}; events after t=3200: {sorted(set(kinds))}")
print(f"   recovery buffers at t=16500: len(rec.X)={len(e.rec.X)}, len(rec.t)={len(e.rec.t)}, len(rec.lab)={len(e.rec.lab)}  (unbounded growth; no max-duration or exit)")
e2 = MVEngine(); run(e2, sc, mask_after=3200, limit=6000)
print(f"   same engine at t=6000 (a recurrence of A started at 4500): len(rec.X)={len(e2.rec.X)} -> grows linearly with stream length")

print("== B. label_delay larger than the stream: warm-up never completes")
e = MVEngine(label_delay=10**9); run(e, sc, limit=2000)
print(f"   state={e.state}, models_trained={e.models_trained}, len(_wu_X)={len(e._wu_X)}  (predicts constant 0; unbounded warm-up buffer)")

print("== C. out-of-order timestamps (t decreasing): labels are released by 't - t_oldest >= delay'")
e = MVEngine(); n_rel = 0
for i, (t, x, y, _) in enumerate(sc):
    if i >= 400: break
    _, es = e.process(1000 - i, x, y)
print(f"   after 400 samples with t=1000..601: state={e.state}, models_trained={e.models_trained}, len(_delay)={len(e._delay)}  (with delay 0, 'released' still works since t-t0<0 is False)")
e = MVEngine(label_delay=5)
for i, (t, x, y, _) in enumerate(sc):
    if i >= 400: break
    e.process(1000 - i, x, y)
print(f"   with label_delay=5 and decreasing t: state={e.state}, models_trained={e.models_trained}, len(_delay)={len(e._delay)}  (no label is ever released)")

print("== D. the same observation processed twice (no de-duplication / sequence check)")
e = MVEngine(); data = list(sc)[:300]
for t, x, y, _ in data: e.process(t, x, y)
n1 = len(e.win_X)
for t, x, y, _ in data[:50]: e.process(t, x, y)       # replayed
print(f"   after replaying 50 already-seen observations: state={e.state}; the engine accepted them (win_X={len(e.win_X)}, detector.n={e.detector.n}, models_trained={e.models_trained})")

print("== E. repeated drift on a STATIONARY stream (single concept, 30,000 samples, 5 seeds)")
rows = []
for seed in range(1, 6):
    rng = np.random.default_rng(seed); X = sample(Spec("a", (0,)*4, (1,)*4), 30000, rng); w = np.array([1, 1, 0, 0.0])
    y = ((X @ w) > 0).astype(int); y = np.where(rng.random(30000) < 0.04, 1 - y, y)
    e = MVEngine(); ev = []
    for t in range(30000): ev += e.process(t, X[t], int(y[t]))[1]
    d = [v for v in ev if v["kind"] == "drift"]
    rows.append((len(d), sum(v["why"] == "error" for v in d), sum(v["why"] == "feature_shift" for v in d), e.models_trained))
print("   (drifts, by_error, by_feature_shift, models_trained) per seed:", rows, " -> false drift alarms per 10k samples ~ %.2f" % (np.mean([r[0] for r in rows]) / 3))

print("== F. an exception inside recovery in a live Pipeline: does the system notice?")
from dejastream.pipeline import Pipeline
import dejastream.matcher as M
orig = M.ConceptRepository.match
state = {"n": 0}
def boom(self, *a, **k):
    state["n"] += 1
    if state["n"] == 2: raise RuntimeError("injected fault in match()")
    return orig(self, *a, **k)
M.ConceptRepository.match = boom
err = io.StringIO()
with contextlib.redirect_stderr(err):
    p = Pipeline(rate=2500, loops=1, db_path=os.path.join(tempfile.mkdtemp(), "a.db")); p.start(); time.sleep(9); snap = p.snapshot(); p.close()
M.ConceptRepository.match = orig
alive = {t.name: t.is_alive() for t in p.threads}
lag = p.bus.stats()["lag"]
print(f"   threads alive: {alive}")
print(f"   consumer lag proc-dejastream_mv on ds.raw: {lag.get('proc-dejastream_mv', {}).get('ds.raw')}   (others: { {k: v.get('ds.raw') for k, v in lag.items() if k.startswith('proc-') and k != 'proc-dejastream_mv'} })")
print(f"   traceback printed to stderr only: {'RuntimeError' in err.getvalue()}; dashboard 'summary' for MV still reports accuracy={snap['summary']['dejastream_mv']['acc']:.3f} with seen={snap['summary']['dejastream_mv']['seen']} (stale, no alert in API: keys={list(snap)[:6]})")

print("== G. model objects vs repository when the repository cap evicts concepts (cap=2)")
e = MVEngine(match=MatchConfig(max_concepts=2)); run(e, sc)
print(f"   concepts in repo={len(e.repo)}  models held by engine={len(e.models)} (models of evicted concepts are never released)")

print("== H. stored model aliasing and rollback state")
e = MVEngine(); ev = run(e, sc, limit=4560)
rec = [x for x in ev if x["kind"] == "recall"]
mid = e.repo.records[e.active_id].model_id
print(f"   active model is the stored object: {e.model is e.models[mid]} (online SGD updates mutate the repository's model in place; no snapshot of the version that was validated)")
