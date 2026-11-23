"""Mathematical / decision-logic counterexamples against the DEFAULT configuration."""
import os, sys, warnings
sys.path.insert(0, os.getcwd()); warnings.simplefilter("ignore")
import numpy as np
from datastream.mvfp import extract_feature_fp
from datastream.similarity import compare, SimilarityConfig
from datastream.matcher import ConceptRepository, MatchConfig, Decision
from datastream.research.gen import Spec, sample
rng = np.random.default_rng(0)
fp = lambda X: extract_feature_fp(X, t=np.arange(len(X)))
Z4, O4 = (0.0,)*4, (1.0,)*4

print("== 1. effective_n uses ONE rho (median over features) for every statistic")
spec = Spec("m", Z4, O4, rho=(0.9, 0.9, 0.0, 0.0))
f = fp(sample(spec, 200, rng)); rho = f.temporal["acf1"]
print(f"   per-feature lag-1 ACF = {np.round(rho,2)}; median used = {np.nanmedian(rho):.2f}; effective_n() = {f.effective_n():.1f} of n=200")
print(f"   per-feature mean-type n_eff would be {np.round(200*(1-np.clip(rho,0,.95))/(1+np.clip(rho,0,.95)),0)}  -> AR(0.9) features get n_eff~{200*(1-.9)/(1.9):.0f}, but the floor uses {f.effective_n():.0f}")
neg = fp(sample(Spec("n", Z4, O4, rho=-0.8), 200, rng)); print(f"   negative rho is clipped to 0: acf1={np.round(neg.temporal['acf1'],2)} -> effective_n={neg.effective_n():.0f} (n=200): the benefit of negative dependence is discarded")

print("== 2. missing evidence can LOWER the distance (availability-normalised weights)")
iid, ar = sample(Spec("a", Z4, O4), 120, rng), sample(Spec("b", Z4, O4, rho=0.9), 120, rng)
a, b = fp(iid), fp(ar)
full, nt = compare(a, b), compare(a, b, views=("stat", "dist"))
print(f"   iid vs AR(0.9) windows (same marginals): components={ {k: (None if v is None else round(v,3)) for k,v in full.components.items()} }")
print(f"   D with temporal view = {full.combined:.3f}; D when temporal is unavailable = {nt.combined:.3f}  (weights renormalised: D=(0.4*Ds+0.3*Dd+0.1*Dt)/0.8 vs (0.4*Ds+0.3*Dd)/0.7)")
short = compare(fp(iid[:8]), fp(ar[:8])); print(f"   with 8-sample windows temporal is None: {short.components['time'] is None}; completeness={short.completeness:.3f}")
cfgf = SimilarityConfig(weight_mode='fixed'); print(f"   weight_mode='fixed' (opt-in) gives D={compare(fp(iid[:8]), fp(ar[:8]), cfgf).combined:.3f} for the same short windows")

print("== 3. HIGH_CONFIDENCE_MATCH with a single stored concept: margin is hard-wired to 1 - D")
sp = Spec("a", Z4, O4); repo = ConceptRepository(); repo.add(fp(sample(sp, 300, rng)), "m")
for shift in (0.0, 0.15, 0.25, 0.35, 0.45):
    rs = []
    for _ in range(30):
        q = fp(sample(Spec("q", (shift,)*4, O4), 100, rng)); rs.append(repo.match(q))
    dec = {d.value: sum(r.decision == d for r in rs) for d in Decision}
    print(f"   query shifted {shift:.2f} sigma on all 4 features: mean D={np.mean([r.D for r in rs]):.3f}  decisions={ {k:v for k,v in dec.items() if v} }")

print("== 4. Level 1 (top-5) with look-alike concepts: where does the true concept rank?")
for n_conc in (6, 12, 24):
    repo = ConceptRepository(); specs = []
    for i in range(n_conc):
        s = Spec(f"c{i}", Z4, O4, rho=(0.0 if i % 2 else 0.9), corr=(0.0 if i % 3 else 0.7)); specs.append(s)   # same marginals, different dependence/temporal
        repo.add(fp(sample(s, 300, rng)), f"m{i}")
    hit = {1: 0, 3: 0, 5: 0, 10: 0}; N = 120
    for _ in range(N):
        j = int(rng.integers(n_conc)); q = fp(sample(specs[j], 100, rng)); ids, sc = repo.level1_scores(q)
        order = np.argsort(sc, kind="stable"); rank = [ids[i] for i in order].index(f"concept_{j+1:03d}")
        for k in hit: hit[k] += rank < k
    print(f"   {n_conc} concepts with identical marginals: Level-1 Recall@1/3/5/10 = " + "/".join(f"{hit[k]/N:.2f}" for k in hit) + "  (default top_k=5; true concept outside the shortlist is never scored by Level 2)")

print("== 5. feature order / schema is not part of the fingerprint")
X = sample(Spec("a", (0, 3, -2, 5.0), (1, 2, .5, 1), 0.0), 200, rng); A, B = fp(X[:100]), fp(X[100:])
Xs = X[100:][:, [1, 0, 2, 3]]
print(f"   same concept, two windows: D={compare(A, B).combined:.3f}; second window with columns 0 and 1 swapped: D={compare(A, fp(Xs)).combined:.3f} (no error, no warning)")
r = ConceptRepository(); r.add(A, "m"); q5 = fp(np.hstack([X[100:], X[100:, :1]]))
res = r.match(q5); print(f"   query with an extra (5th) feature vs a 4-feature repository: decision={res.decision.value}, reason='{res.reason}'  -> silently a new concept")

print("== 6. non-finite values")
Xi = X[:100].copy(); Xi[3, 0] = np.inf; Xi[7, 1] = -np.inf
fi = fp(Xi); print(f"   +-inf converted to missing: missing ratio col0={fi.statistics['missing'][0]:.2f}, col1={fi.statistics['missing'][1]:.2f}; D(clean,inf)={compare(fp(X[:100]), fi).combined:.3f}")

print("== 7. default merge (legacy averaging) in the default engine path")
r = ConceptRepository(); r.add(fp(rng.normal(0, 1, (300, 1))), "m"); rec = next(iter(r.records.values()))
r.update(rec.concept_id, fp(rng.normal(5, 1, (100, 1))), t=1)
print(f"   after merging N(0,1) (n=300) with N(5,1) (n=100): stored std={rec.fp.statistics['std'][0]:.3f} (pooled truth 2.36), q90={rec.fp.statistics['q90'][0]:.2f} (truth ~5.2)")

print("== 8. confidence contains the similarity term directly")
repo = ConceptRepository(); repo.add(fp(sample(sp, 300, rng)), "m")
res = repo.match(fp(sample(sp, 100, rng)))
print(f"   confidence={res.confidence:.3f} = closeness {res.confidence_terms['closeness']:.3f} (=1-D/thr, a function of similarity) x separation {res.confidence_terms['separation']:.3f} x evidence {res.confidence_terms['evidence']:.3f} x variability {res.confidence_terms['variability']:.3f} x predictive {res.confidence_terms['predictive']:.3f}")
res_nt = None
r2 = ConceptRepository(); r2.add(a, "m"); print(f"   same-concept pair where temporal differs: with temporal view conf={r2.match(b).confidence:.3f} D={r2.match(b).D:.3f}")

print("== 9. legacy temporal distance: heterogeneous statistics averaged with |delta|")
z = fp(rng.normal(0, 1, (120, 2))); z3 = fp(rng.normal(0, 3, (120, 2)))
print(f"   iid N(0,1) vs iid N(0,9) (amplitude only): D_time={compare(z, z3).components['time']:.3f} (ratios are scale-free: ok); D_stat={compare(z, z3).components['stat']:.3f}")
