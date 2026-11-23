import os, sys, json, tempfile, warnings, time
sys.path.insert(0, os.getcwd()); warnings.simplefilter("ignore")
import numpy as np
tmp = tempfile.mkdtemp()
print("== R1. python run.py bench (demo stream) reproduced into a temp file and compared with results/bench.json")
from dejastream.bench import run as bench
bench(out=os.path.join(tmp, "bench.json"), verbose=False)
a, b = json.load(open("results/bench.json")), json.load(open(os.path.join(tmp, "bench.json")))
same = all(a["strategies"][s][k] == b["strategies"][s][k] for s in a["strategies"] for k in ("accuracy", "post_drift_300_acc", "models_trained", "recalls", "false_alarms", "drifts_detected"))
print(f"   all accuracy / post-drift / models / recalls / false alarms identical: {same}")
print("== R2. bench-mv (8 scenarios x 5 seeds x 4 strategies) reproduced and compared with results/bench_mv.json (timing fields ignored)")
from dejastream.bench_mv import run as bmv
t0 = time.time(); bmv(out=os.path.join(tmp, "bench_mv.json"), verbose=False); print(f"   rerun took {time.time()-t0:.0f}s")
A, B = json.load(open("results/bench_mv.json")), json.load(open(os.path.join(tmp, "bench_mv.json")))
bad = []; n = 0
for sc, d in A["scenarios"].items():
    for s, m in d.items():
        if s == "_notes": continue
        for k, v in m.items():
            if v is None or k in ("us_per_sample", "fp_ms", "match_ms"): continue
            n += 1
            if abs(v["mean"] - B["scenarios"][sc][s][k]["mean"]) > 1e-9: bad.append((sc, s, k))
print(f"   compared {n} metric cells; mismatches: {len(bad)} {bad[:3]}")
print("== R3. research audit measurements (audit_checks) reproduced vs results/research/audit.json")
from dejastream.research.audit_checks import main as ac
t0 = time.time(); ac(out=os.path.join(tmp, "audit.json")); print(f"   rerun took {time.time()-t0:.0f}s")
X, Y = json.load(open("results/research/audit.json")), json.load(open(os.path.join(tmp, "audit.json")))
print("   identical:", X == Y)
print("== T1. noisy / unstable concept -> permissive legacy threshold thr_cand = max(0.32, 2.5 * intra)")
from dejastream.mvfp import extract_feature_fp
from dejastream.matcher import ConceptRepository
from dejastream.research.gen import Spec, sample
rng = np.random.default_rng(3); fp = lambda X: extract_feature_fp(X, t=np.arange(len(X)))
Z, O = (0.0,)*4, (1.0,)*4
for label, sp in (("clean", Spec("a", Z, O)), ("heavy-tailed + autocorrelated", Spec("n", Z, O, rho=0.8, tail=0.15, noise=0.8))):
    repo = ConceptRepository(); rec = repo.add(fp(sample(sp, 300, rng)), "m")
    for k in range(15): repo.update(rec.concept_id, fp(sample(sp, 60, rng)), t=k)
    th = repo.thresholds(rec.concept_id, 60)
    far = fp(sample(Spec("f", (1.0,)*4, O), 60, rng))        # a genuinely different concept: 1 sigma shift on every feature
    res = repo.match(far)
    print(f"   {label:30} intra={rec.intra:.3f} -> thr_cand={th['thr_cand']:.3f}, thr_high={th['thr_high']:.3f}; query shifted 1 sigma: D={res.D:.3f} decision={res.decision.value}")
