"""M2: does the noise model keep same-concept distances stable without collapsing different-concept distances?

Grid: window size n in {20,40,80,160}, AR(1) rho in {-0.5,0,0.5,0.9}, measurement noise in {0,0.5,1}, dimension in {2,4,10}.
Configs: legacy (global AR(1), analytic floor) | per_stat (statistic/feature-specific ESS) | profile (empirical same-concept
profile from a SEPARATE reference series) | none (no floor).
"""
from __future__ import annotations

import itertools
import json
import sys

import numpy as np

from ..mvfp import extract_feature_fp
from ..noise import build_profile
from ..similarity import SimilarityConfig, compare
from .gen import Spec, auc, sample

CONFIGS = {"legacy": SimilarityConfig(), "per_stat": SimilarityConfig(ess_mode="per_stat"),
           "profile": SimilarityConfig(noise_mode="profile"), "none": SimilarityConfig(noise_mode="none")}


def run(reps=40, seed=11, ns=(20, 40, 80, 160), rhos=(-0.5, 0.0, 0.5, 0.9), noises=(0.0, 0.5, 1.0), ds=(2, 4, 10), shift=0.6, out="results/research/e1_noise.json"):
    rng = np.random.default_rng(seed)
    rows = []
    for n, rho, noise, d in itertools.product(ns, rhos, noises, ds):
        sp0 = Spec("a", (0.0,) * d, (1.0,) * d, rho, noise=noise)
        sp1 = Spec("b", (shift,) * d, (1.0,) * d, rho, noise=noise)
        fp = lambda X: extract_feature_fp(X, t=np.arange(len(X)))
        same = [(fp(sample(sp0, n, rng)), fp(sample(sp0, n, rng))) for _ in range(reps)]
        diff = [(fp(sample(sp0, n, rng)), fp(sample(sp1, n, rng))) for _ in range(reps)]
        ref = sample(sp0, 900, rng)                                       # separate reference series for the profile
        cfg0 = SimilarityConfig(noise_mode="none")
        prof = build_profile(ref, n, n, lambda X, t: extract_feature_fp(X, t=t), lambda a, b: compare(a, b, cfg0), B=24, rng=rng)
        cell = {"n": n, "rho": rho, "noise": noise, "d": d, "profile_method": prof.method}
        for name, cfg in CONFIGS.items():
            P = prof if name == "profile" else None
            s = [compare(a, b, cfg, profile=P) for a, b in same]
            df = [compare(a, b, cfg, profile=P) for a, b in diff]
            sc, dc = np.array([x.combined for x in s]), np.array([x.combined for x in df])
            cell[name] = {"same_mean": float(sc.mean()), "same_q95": float(np.quantile(sc, .95)), "diff_median": float(np.median(dc)),
                          "auc": auc(dc, sc), "false_flag_cand": float(np.mean(sc > 0.32)), "false_flag_high": float(np.mean(sc > 0.18)),
                          "unresolvable": float(np.mean([any(v == "unresolvable" for v in x.noise.values()) for x in s])),
                          "view_q95": {v: float(np.quantile([x.components[v] for x in s if x.components[v] is not None], .95))
                                       for v in ("stat", "dist", "time")}}
        rows.append(cell)
    with open(out, "w") as f:
        json.dump(rows, f)
    return rows


def summarise(rows) -> str:
    L = ["| config | mean q95 of D_same | worst-cell q95 | range of q95 (stability) | mean AUC | worst AUC | mean D_diff median | false-flag @0.32 | false-flag @0.18 |", "|---|---|---|---|---|---|---|---|---|"]
    for name in CONFIGS:
        q = np.array([r[name]["same_q95"] for r in rows]); a = np.array([r[name]["auc"] for r in rows])
        L.append(f"| {name} | {q.mean():.3f} | {q.max():.3f} | {q.max() - q.min():.3f} | {a.mean():.3f} | {a.min():.3f} | "
                 f"{np.mean([r[name]['diff_median'] for r in rows]):.3f} | {np.mean([r[name]['false_flag_cand'] for r in rows]):.3f} | "
                 f"{np.mean([r[name]['false_flag_high'] for r in rows]):.3f} |")
    for key, vals in (("rho", (-0.5, 0.0, 0.5, 0.9)), ("n", (20, 40, 80, 160)), ("d", (2, 4, 10)), ("noise", (0.0, 0.5, 1.0))):
        L.append(f"\nmean q95 of D_same by {key}:\n\n| {key} | " + " | ".join(CONFIGS) + " |\n|---|" + "---|" * len(CONFIGS))
        for v in vals:
            sel = [r for r in rows if r[key] == v]
            L.append(f"| {v} | " + " | ".join(f"{np.mean([r[c]['same_q95'] for r in sel]):.3f}" for c in CONFIGS) + " |")
    return "\n".join(L)


if __name__ == "__main__":
    rows = run()
    md = summarise(rows)
    open("results/research/e1_noise.md", "w").write(md)
    print(md)
