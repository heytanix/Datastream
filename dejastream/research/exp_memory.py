"""M7: merge correctness, update cost / memory, and eviction policies under skewed recurrence."""
from __future__ import annotations

import json
import time

import numpy as np

from ..matcher import ConceptRepository, Decision, MatchConfig
from ..mvfp import extract_feature_fp, merge_feature_fp
from ..memory import ConceptSummaries
from .gen import Spec, sample
from .presets import improved_similarity


def fpx(X):
    return extract_feature_fp(X, t=np.arange(len(X)))


def exp_merge_accuracy(reps=30, W=10, n=100, seed=1):
    """Windows whose location differs by N(0, spread): error of the merged concept vs the exact pooled data."""
    rng = np.random.default_rng(seed)
    out = {}
    for spread in (0.0, 0.5, 1.0, 2.0):
        errs = {"legacy averaging": {"std": [], "q10": [], "q90": [], "skew": []}, "mergeable": {"std": [], "q10": [], "q90": [], "skew": []}}
        for _ in range(reps):
            wins = [rng.normal(rng.normal(0, spread), 1.0, (n, 1)) for _ in range(W)]
            allx = np.vstack(wins)[:, 0]
            truth = {"std": allx.std(), "q10": np.quantile(allx, .1), "q90": np.quantile(allx, .9), "skew": np.mean(((allx - allx.mean()) / allx.std()) ** 3)}
            fps = [fpx(w) for w in wins]
            leg = fps[0]
            for k, f in enumerate(fps[1:], 1):
                leg = merge_feature_fp(leg, f, n * k, n)
            s = ConceptSummaries.from_window(wins[0], fps[0])
            for w, f in zip(wins[1:], fps[1:]):
                s.merge_window(ConceptSummaries.from_window(w, f))
            mg = s.compose()
            for name, fp in (("legacy averaging", leg), ("mergeable", mg)):
                got = {"std": fp.statistics["std"][0], "q10": fp.statistics["q10"][0], "q90": fp.statistics["q90"][0], "skew": fp.statistics["skew"][0]}
                for k in truth:
                    errs[name][k].append(abs(got[k] - truth[k]) / (allx.std() if k != "skew" else 1.0))
        out[f"location spread {spread}"] = {name: {k: float(np.mean(v)) for k, v in d.items()} for name, d in errs.items()}
    return out


def exp_cost(windows=50, n=100, seed=2):
    rng = np.random.default_rng(seed)
    out = {}
    for d in (4, 10):
        sp = Spec("a", (0.0,) * d, (1.0,) * d)
        for mode in ("legacy", "mergeable"):
            repo = ConceptRepository(MatchConfig(merge_mode=mode))
            X = sample(sp, 200, rng)
            rec = repo.add(fpx(X), "m", X=X)
            ws = [sample(sp, n, rng) for _ in range(windows)]
            fs = [fpx(w) for w in ws]
            t0 = time.perf_counter()
            for w, f in zip(ws, fs):
                repo.update(rec.concept_id, f, t=1, X=w)
            us = 1e6 * (time.perf_counter() - t0) / windows
            out[f"d={d} {mode}"] = {"update_us": us, "bytes_per_concept_after_50_windows": rec.nbytes()}
    return out


def exp_eviction(seeds=(1, 2, 3), events=300, n_concepts=25, cap=8, n=60):
    sim = improved_similarity(weights={"stat": 0.4, "dist": 0.2, "time": 0.4, "pred": 0.2})
    policies = {"legacy (uses, last_used)": dict(eviction="legacy"), "lru": dict(eviction="lru"), "score (recurrence + recency)": dict(eviction="score"),
                "score + grace 2": dict(eviction="score", grace_updates=2)}
    res = {p: {"hit_rate_when_present": [], "evicted_then_needed": [], "adds": []} for p in policies}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        specs = [Spec(f"c{i}", tuple(rng.normal(0, 4, 4)), tuple(rng.uniform(.7, 1.5, 4))) for i in range(n_concepts)]
        zipf = 1 / np.arange(1, n_concepts + 1) ** 1.1
        zipf /= zipf.sum()
        seq = rng.choice(n_concepts, events, p=zipf)
        windows = [sample(specs[c], n, np.random.default_rng(seed * 1000 + i)) for i, c in enumerate(seq)]
        for pname, kw in policies.items():
            repo = ConceptRepository(MatchConfig(max_concepts=cap, top_k_mode="adaptive", expand_on_new=True, **kw), sim)
            where: dict[int, str] = {}                  # concept index -> id of the stored concept that represents it
            seen, present_events, hits, needed_after_evict, adds = set(), 0, 0, 0, 0
            for i, (c, X) in enumerate(zip(seq, windows)):
                present = c in where and where[c] in repo
                if c in seen and not present:
                    needed_after_evict += 1
                res_ = repo.match(fpx(X))
                ok = res_.decision in (Decision.HIGH_CONFIDENCE_MATCH, Decision.CANDIDATE_MATCH) and present and res_.best_id == where[c]
                if present:
                    present_events += 1
                    hits += ok
                if ok:
                    repo.update(res_.best_id, fpx(X), i, count_use=True, X=X)
                else:
                    big = np.vstack([X, sample(specs[c], 100, np.random.default_rng(seed + i))])
                    rec = repo.add(fpx(big), f"m{i}", t=i, X=big)
                    where[c] = rec.concept_id
                    adds += 1
                seen.add(c)
            res[pname]["hit_rate_when_present"].append(hits / max(present_events, 1))
            res[pname]["evicted_then_needed"].append(needed_after_evict)
            res[pname]["adds"].append(adds)
    return {p: {k: (float(np.mean(v)), float(np.std(v))) for k, v in d.items()} for p, d in res.items()}


def to_md(m, c, e) -> str:
    L = ["### Merge accuracy: mean |error| / pooled std (10 windows of 100; lower is better)\n", "| location drift between windows | method | std | q10 | q90 | skew |\n|---|---|---|---|---|---|"]
    for k, d in m.items():
        for name, v in d.items():
            L.append(f"| {k} | {name} | {v['std']:.4f} | {v['q10']:.4f} | {v['q90']:.4f} | {v['skew']:.4f} |")
    L.append("\n### Update cost and memory\n\n| case | µs per update | bytes per concept |\n|---|---|---|")
    for k, v in c.items():
        L.append(f"| {k} | {v['update_us']:.0f} | {v['bytes_per_concept_after_50_windows']} |")
    L.append("\n### Eviction under Zipf recurrence (25 concepts, repository cap 8, 300 events, 3 seeds)\n\n| policy | hit rate when concept is stored | needed after eviction | concepts created |\n|---|---|---|---|")
    for p, v in e.items():
        L.append(f"| {p} | {v['hit_rate_when_present'][0]:.3f}±{v['hit_rate_when_present'][1]:.3f} | {v['evicted_then_needed'][0]:.1f}±{v['evicted_then_needed'][1]:.1f} | {v['adds'][0]:.1f}±{v['adds'][1]:.1f} |")
    return "\n".join(L)


if __name__ == "__main__":
    m, c, e = exp_merge_accuracy(), exp_cost(), exp_eviction()
    json.dump({"merge": m, "cost": c, "eviction": e}, open("results/research/e10_memory.json", "w"), indent=1)
    md = to_md(m, c, e)
    open("results/research/e10_memory.md", "w").write(md)
    print(md)
