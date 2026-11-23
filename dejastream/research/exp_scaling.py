"""M8/M13: retrieval latency, memory and fingerprint-extraction cost as the repository grows (10 .. 1000 concepts)."""
from __future__ import annotations

import json
import time
import tracemalloc

import numpy as np

from ..matcher import ConceptRepository, MatchConfig
from ..mvfp import extract_feature_fp
from .gen import Spec, sample
from .presets import improved_similarity


def _p(a, q):
    return float(np.percentile(a, q))


def build(size, d, rng, cfg, sim):
    repo = ConceptRepository(cfg, sim)
    specs = []
    for i in range(size):
        sp = Spec(f"c{i}", tuple(rng.normal(0, 6, d)), tuple(rng.uniform(0.7, 1.5, d)))
        specs.append(sp)
        repo.add(extract_feature_fp(sample(sp, 150, rng), t=np.arange(150)), f"m{i}", t=i)
    return repo, specs


def exp_scaling(sizes=(10, 50, 100, 500, 1000), queries=150, d=4, seed=0):
    sim = improved_similarity(weights={"stat": 0.4, "dist": 0.2, "time": 0.4, "pred": 0.2}, group_agg="mean")
    configs = {"basic, fixed k=5 (schema 2.0)": dict(), "basic, adaptive k": dict(top_k_mode="adaptive"),
               "shape_corr, adaptive k": dict(l1_repr="shape_corr", top_k_mode="adaptive"),
               "basic adaptive + full-scan fallback": dict(top_k_mode="adaptive", expand_on_new=True, max_full_scan=2000)}
    out = {}
    for size in sizes:
        rng = np.random.default_rng(seed)
        for cname, kw in configs.items():
            tracemalloc.start()
            repo, specs = build(size, d, np.random.default_rng(seed), MatchConfig(max_concepts=size + 5, **kw), sim)
            cur, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            qs = [extract_feature_fp(sample(specs[int(rng.integers(size))], 60, rng), t=np.arange(60)) for _ in range(queries)]
            repo.level1(qs[0])                                    # build the index once, outside the timing (rebuilt only after changes)
            l1, full, ncand = [], [], []
            for q in qs:
                t0 = time.perf_counter_ns(); repo.level1(q); l1.append((time.perf_counter_ns() - t0) / 1e6)
                t0 = time.perf_counter_ns(); r = repo.match(q); full.append((time.perf_counter_ns() - t0) / 1e6); ncand.append(r.n_scored)
            out[f"{size} | {cname}"] = {"size": size, "config": cname, "level1_ms_mean": float(np.mean(l1)), "level1_ms_p95": _p(l1, 95),
                                       "match_ms_mean": float(np.mean(full)), "match_ms_p95": _p(full, 95), "mean_candidates_scored": float(np.mean(ncand)),
                                       "repo_bytes_total": int(sum(r.nbytes() for r in repo.records.values())), "build_peak_mb": peak / 1e6,
                                       "bytes_per_concept": repo.memory_per_concept()}
        # exhaustive Level-2 scan, the cost that Level 1 avoids (few queries: it is slow)
        repo, specs = build(size, d, np.random.default_rng(seed), MatchConfig(max_concepts=size + 5), sim)
        t = []
        for q in qs[:5]:
            t0 = time.perf_counter_ns()
            for cid in repo.records:
                repo.score(q, cid)
            t.append((time.perf_counter_ns() - t0) / 1e6)
        out[f"{size} | exhaustive level-2 scan"] = {"size": size, "config": "exhaustive level-2 scan", "match_ms_mean": float(np.mean(t)), "match_ms_p95": _p(t, 95)}
    return out


def exp_fingerprint_cost(reps=300, seed=0):
    from ..fingerprint import fingerprint as legacy
    rng = np.random.default_rng(seed)
    out = {}
    for n, d in ((40, 4), (100, 4), (300, 4), (100, 10), (100, 30)):
        X, y = rng.normal(0, 1, (n, d)), rng.integers(0, 2, n)
        t = np.arange(n)
        a, b = [], []
        for _ in range(reps):
            t0 = time.perf_counter_ns(); legacy(X, y); a.append((time.perf_counter_ns() - t0) / 1e6)
            t0 = time.perf_counter_ns(); extract_feature_fp(X, t=t); b.append((time.perf_counter_ns() - t0) / 1e6)
        out[f"n={n} d={d}"] = {"legacy_ms_mean": float(np.mean(a)), "multiview_ms_mean": float(np.mean(b)), "multiview_ms_p95": _p(b, 95)}
    return out


def to_md(sc, fc) -> str:
    L = ["### Matching latency vs repository size (d=4, query windows of 60 samples, 150 queries)\n", "| concepts | configuration | Level 1 ms (mean / p95) | full match ms (mean / p95) | mean concepts scored | bytes per concept | build peak MB |", "|---|---|---|---|---|---|---|"]
    for k, v in sc.items():
        if v["config"] == "exhaustive level-2 scan":
            L.append(f"| {v['size']} | {v['config']} | - | {v['match_ms_mean']:.1f} / {v['match_ms_p95']:.1f} | all | - | - |")
        else:
            L.append(f"| {v['size']} | {v['config']} | {v['level1_ms_mean']:.3f} / {v['level1_ms_p95']:.3f} | {v['match_ms_mean']:.2f} / {v['match_ms_p95']:.2f} | {v['mean_candidates_scored']:.1f} | {v['bytes_per_concept']:.0f} | {v['build_peak_mb']:.1f} |")
    L.append("\n### Fingerprint extraction latency\n\n| window | legacy v1 ms | multi-view ms (mean) | multi-view ms (p95) |\n|---|---|---|---|")
    for k, v in fc.items():
        L.append(f"| {k} | {v['legacy_ms_mean']:.3f} | {v['multiview_ms_mean']:.3f} | {v['multiview_ms_p95']:.3f} |")
    return "\n".join(L)


if __name__ == "__main__":
    sc, fc = exp_scaling(), exp_fingerprint_cost()
    json.dump({"scaling": sc, "fingerprint_cost": fc}, open("results/research/e12_scaling.json", "w"), indent=1)
    md = to_md(sc, fc)
    open("results/research/e12_scaling.md", "w").write(md)
    print(md)
