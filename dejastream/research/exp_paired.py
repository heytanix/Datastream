"""M8: paired significance of the improved system against the schema-2.0 baseline, per scenario, with per-seed data kept.

Test: exact paired sign-flip permutation test on the per-seed differences (2^n sign patterns, n = number of seeds with both
values defined), two-sided, Holm-corrected across scenarios for each metric. With 10 seeds the smallest attainable p is 2/1024.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time

import numpy as np

from ..bench_mv import run_one
from ..scenarios import SCENARIOS
from .presets import ablation_variants

METRICS = ["accuracy", "post_drift_acc", "models_trained", "retrieval_precision", "retrieval_recall", "false_match_rate",
           "missed_match_rate", "candidate_recall@1", "us_per_sample"]
# "F" (the improved system with schema-2.0 thresholds) is the variant the seeds-1-5 ablation pointed to; it is confirmed here on
# FRESH seeds (11-20) that were not used to choose it.
PAIRS = [("F = D without adaptive thresholds", "MV schema 2.0 (baseline)"), ("D full multi-view (improved)", "MV schema 2.0 (baseline)"),
         ("MV schema 2.0 (baseline)", "v1 (legacy engine)")]


def sign_flip_p(d: np.ndarray) -> float:
    d = np.asarray(d, float)
    n = len(d)
    if n == 0 or np.all(d == 0):
        return 1.0
    obs = abs(d.mean())
    hits = 0
    for signs in itertools.product((-1.0, 1.0), repeat=n):
        hits += abs(np.dot(signs, d) / n) >= obs - 1e-12
    return hits / 2 ** n


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def collect(out="results/research/e13_paired_raw.json", seeds=tuple(range(11, 21)), log=sys.stdout):
    variants = ablation_variants()
    names = sorted({v for p in PAIRS for v in p})
    raw = json.load(open(out)) if os.path.exists(out) else {}
    t0 = time.time()
    for sc in SCENARIOS:
        for v in names:
            key = f"{sc}||{v}"
            if key in raw:
                continue
            strat, kw = variants[v]
            raw[key] = [run_one(SCENARIOS[sc](seed=s), strat, **kw) for s in seeds]
            json.dump(raw, open(out, "w"))
            print(f"[{time.time() - t0:6.0f}s] {key}", file=log, flush=True)
    return raw


def analyse(raw, seeds_n=10):
    res = {}
    for a, b in PAIRS:
        for m in METRICS:
            rows = []
            for sc in SCENARIOS:
                ra, rb = raw[f"{sc}||{a}"], raw[f"{sc}||{b}"]
                pairs = [(x[m], y[m]) for x, y in zip(ra, rb) if x.get(m) is not None and y.get(m) is not None]
                if len(pairs) < 5:
                    continue
                d = np.array([p[0] - p[1] for p in pairs])
                rows.append({"scenario": sc, "n": len(d), "mean_a": float(np.mean([p[0] for p in pairs])), "mean_b": float(np.mean([p[1] for p in pairs])),
                             "diff": float(d.mean()), "p": sign_flip_p(d)})
            adj = holm([r["p"] for r in rows])
            for r, p in zip(rows, adj):
                r["p_holm"] = p
            res[f"{a} vs {b} | {m}"] = rows
    return res


if __name__ == "__main__":
    raw = collect()
    json.dump(analyse(raw), open("results/research/e13_paired.json", "w"), indent=1)
