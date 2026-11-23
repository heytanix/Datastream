"""Measured performance (this machine, single run unless stated). Source untouched; outputs only to stdout."""
import os, sys, time, json, tempfile, tracemalloc, warnings
sys.path.insert(0, os.getcwd()); warnings.simplefilter("ignore")
import numpy as np
from datastream.mvfp import extract_feature_fp
from datastream.fingerprint import fingerprint as legacy_fp
from datastream.similarity import compare
from datastream.matcher import ConceptRepository, MatchConfig
from datastream.mv_engine import MVEngine
from datastream.strategies import make_engine
from datastream.scenarios import abrupt_recurring
from datastream.validation import compare_paired, ValidationConfig, Pred
from datastream.research.gen import Spec, sample
rng = np.random.default_rng(0)
ms = lambda f, r=50: (lambda t0: [f() for _ in range(r)] and (time.perf_counter() - t0) * 1000 / r)(time.perf_counter())

print("== fingerprint extraction vs window size and feature count (ms, mean of 50)")
print(f"   {'n':>5} {'d':>4} {'legacy v1':>10} {'multi-view':>11}")
for n, d in ((40, 4), (100, 4), (300, 4), (1000, 4), (100, 10), (100, 30), (100, 60)):
    X, y = rng.normal(0, 1, (n, d)), rng.integers(0, 2, n); t = np.arange(n)
    print(f"   {n:5d} {d:4d} {ms(lambda: legacy_fp(X, y)):10.3f} {ms(lambda: extract_feature_fp(X, t=t), 20):11.3f}")

print("== one Level-2 compare() call (ms): two 100-sample windows")
for d in (4, 10, 30):
    a, b = (extract_feature_fp(rng.normal(0, 1, (100, d)), t=np.arange(100)) for _ in range(2)); print(f"   d={d:3d}: {ms(lambda: compare(a, b), 30):.3f} ms")

print("== repository operations (50 concepts, d=4)")
repo = ConceptRepository(); specs = []
for i in range(50):
    s = Spec(f"c{i}", tuple(rng.normal(0, 5, 4)), (1.0,)*4); specs.append(s); repo.add(extract_feature_fp(sample(s, 150, rng), t=np.arange(150)), f"m{i}")
q = extract_feature_fp(sample(specs[7], 60, rng), t=np.arange(60)); repo.match(q)
print(f"   match() (Level 1 + top-5 Level 2): {ms(lambda: repo.match(q), 50):.2f} ms;  level1 only: {ms(lambda: repo.level1(q), 200)*1:.3f} ms")
w = extract_feature_fp(sample(specs[7], 100, rng), t=np.arange(100)); cid = "concept_008"
print(f"   update() legacy merge: {ms(lambda: repo.update(cid, w, t=1), 50):.3f} ms")
t0 = time.perf_counter(); blob = json.dumps(repo.to_dict()); t1 = time.perf_counter(); r2 = ConceptRepository(); r2.load_dict(json.loads(blob)); t2 = time.perf_counter()
print(f"   persistence (NOT used by the pipeline): to_dict+json {1000*(t1-t0):.1f} ms, {len(blob)/1024:.0f} KB for 50 concepts; load {1000*(t2-t1):.1f} ms (models are not included)")
y = rng.integers(0, 2, 60); p1, p0 = Pred(np.where(rng.random(60) < .8, y, 1-y)), Pred(np.where(rng.random(60) < .5, y, 1-y))
print(f"   paired validation compare_paired(n=60, 400 bootstrap): {ms(lambda: compare_paired(y, p1, p0, ValidationConfig()), 10):.1f} ms (opt-in path)")
from datastream.store import Store
st = Store(os.path.join(tempfile.mkdtemp(), "x.db")); t0 = time.perf_counter()
for i in range(5000): st.add_metric({"t": i, "strategy": "s", "acc": .9, "models": 1})
st.commit(); print(f"   SQLite add_metric: {1e6*(time.perf_counter()-t0)/5000:.0f} us/row (5000 rows, one commit)")

print("== engine throughput vs feature count d (6000 samples, 2 alternating concepts, default configs)")
for d in (4, 10, 30):
    segs = []
    for k in range(4):
        sp = Spec("s", tuple(rng.normal(0, 3, d)), (1.0,)*d); X = sample(sp, 1500, rng); w = rng.normal(0, 1, d); yy = ((X - X.mean(0)) @ w > 0).astype(int); segs.append((X, yy))
    X = np.vstack([s[0] for s in segs]); Y = np.concatenate([s[1] for s in segs])
    row = []
    for strat in ("retrain", "datastream", "datastream_mv"):
        e = make_engine(strat, n_features=d); t0 = time.perf_counter()
        for t in range(len(Y)): e.process(t, X[t], int(Y[t]))
        row.append(f"{strat}={1e6*(time.perf_counter()-t0)/len(Y):.0f}us")
    print(f"   d={d:3d}: " + "  ".join(row))

print("== MVEngine vs stability-window size (abrupt_recurring, 16,500 samples)")
data = list(abrupt_recurring(seed=1))
for win in (50, 100, 300):
    e = MVEngine(window=win); t0 = time.perf_counter()
    for t, x, yv, _ in data: e.process(t, x, yv)
    print(f"   window={win:4d}: {1e6*(time.perf_counter()-t0)/len(data):.0f} us/sample, fingerprint calls={e.fp_calls}, models={e.models_trained}")

print("== memory and time vs stream length (MVEngine default, tracemalloc peak; stream = abrupt_recurring repeated)")
for reps in (1, 3, 6):
    tracemalloc.start(); e = MVEngine(); t0 = time.perf_counter(); n = 0
    for _ in range(reps):
        for t, x, yv, _ in data: e.process(n, x, yv); n += 1
    cur, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    print(f"   {n:7d} samples: {1e6*(time.perf_counter()-t0)/n:.0f} us/sample (under tracemalloc), peak {peak/1e6:.1f} MB, current {cur/1e6:.1f} MB, concepts={len(e.repo)}, models={len(e.models)}")
