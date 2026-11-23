"""Milestone 1: measure each suspected weakness of the schema-2.0 matcher before changing anything."""
from __future__ import annotations

import json
import sys

import numpy as np

from ..matcher import ConceptRepository, MatchConfig
from ..mvfp import extract_feature_fp, merge_feature_fp
from ..similarity import SimilarityConfig, compare
from .gen import Spec, auc, base_spec, lookalike_library, sample, window


def check_merge(rng) -> dict:
    """Weighted averaging of std and quantiles versus the true pooled values."""
    a, b = rng.normal(0, 1, 300), rng.normal(5, 1, 100)
    fa, fb = (extract_feature_fp(x[:, None], t=np.arange(len(x))) for x in (a, b))
    m = merge_feature_fp(fa, fb, 300, 100)
    both = np.concatenate([a, b])
    out = {"pooled_std_true": float(both.std()), "merged_std": float(m.statistics["std"][0]),
           "pooled_mean_true": float(both.mean()), "merged_mean": float(m.statistics["mean"][0])}
    for k, key, p in (("q10", "q10", .1), ("q25", "q25", .25), ("q50", "median", .5), ("q75", "q75", .75), ("q90", "q90", .9)):
        out[f"{k}_true"] = float(np.quantile(both, p))
        out[f"{k}_merged"] = float(m.statistics[key][0])
    return out


def check_ess(rng, reps=400) -> dict:
    """Does the AR(1) factor n(1-rho)/(1+rho) describe the sampling noise of every statistic?"""
    n, res = 100, {}
    for rho in (0.0, 0.5, 0.9, -0.5):
        stats = {k: [] for k in ("mean", "median", "q90", "std_scale", "acf1", "spearman")}
        for _ in range(reps):
            z = np.empty((n, 2))
            z[0] = rng.normal(0, 1, 2)
            e = rng.normal(0, 1, (n, 2)) * np.sqrt(1 - rho ** 2)
            for i in range(1, n):
                z[i] = rho * z[i - 1] + e[i]
            x = z[:, 0]
            stats["mean"].append(x.mean()); stats["median"].append(np.median(x)); stats["q90"].append(np.quantile(x, .9))
            stats["std_scale"].append(np.log(x.std()))
            xc = x - x.mean(); stats["acf1"].append(xc[:-1] @ xc[1:] / (xc @ xc))
            stats["spearman"].append(np.corrcoef(np.argsort(np.argsort(z[:, 0])), np.argsort(np.argsort(z[:, 1])))[0, 1])
        sd = {k: float(np.std(v)) for k, v in stats.items()}
        iid_se = {"mean": 1 / np.sqrt(n), "median": 1.2533 / np.sqrt(n), "q90": 1.7 / np.sqrt(n), "std_scale": 1 / np.sqrt(2 * n),
                  "acf1": 1 / np.sqrt(n), "spearman": 1 / np.sqrt(n)}
        pred_inflation = np.sqrt((1 + rho) / (1 - rho))
        res[f"rho={rho}"] = {"AR1_predicted_se_inflation": float(pred_inflation),
                             **{f"{k}_observed_inflation": float(sd[k] / iid_se[k]) for k in sd}}
    return res


def check_intra_single_window(rng, reps=300) -> dict:
    """matcher.update: intra starts as ONE window distance. How noisy is that estimate versus after 20 windows?"""
    spec = base_spec(4)
    first, later = [], []
    for _ in range(reps):
        repo = ConceptRepository()
        rec = repo.add(window(spec, 150, rng), "m")
        for k in range(20):
            repo.update(rec.concept_id, window(spec, 100, rng), t=k)
            if k == 0:
                first.append(rec.intra)
        later.append(rec.intra)
    return {"intra_after_1_update": {"mean": float(np.mean(first)), "sd": float(np.std(first))},
            "intra_after_20_updates": {"mean": float(np.mean(later)), "sd": float(np.std(later))},
            "relative_sd_1": float(np.std(first) / np.mean(first)), "relative_sd_20": float(np.std(later) / np.mean(later))}


def check_level1_blindspot(rng, reps=60) -> dict:
    """Concepts with the same median, sigma, q10, q90: how far apart are they at Level 1 versus Level 2?"""
    lib = lookalike_library(4)
    repo = ConceptRepository()
    for name in ("base", "heavy_tail", "bimodal", "far"):
        repo.add(window(lib.specs[name], 400, rng), name)
    out = {}
    for name in ("base", "heavy_tail", "bimodal"):
        ranks = []
        for _ in range(reps):
            q = window(lib.specs[name], 80, rng)
            order = repo.level1(q)
            ranks.append(order.index(repo.records and next(r.concept_id for r in repo.records.values() if r.model_id == name)))
        out[name] = {"mean_level1_rank_of_true_concept(0=best)": float(np.mean(ranks))}
    q = window(lib.specs["heavy_tail"], 80, rng)
    base_id = next(r.concept_id for r in repo.records.values() if r.model_id == "base")
    out["heavy_tail_query_vs_base: level2_D"] = float(repo.score(q, base_id).combined)
    return out


def check_group_agg(rng, reps=80) -> dict:
    """One heavy-tailed, non-stationary irrelevant feature among ten: does the worst-group term dominate D_same?"""
    d = 10
    res = {}
    for label, lam in (("legacy 0.7 mean + 0.3 worst", 0.3), ("mean only", 0.0)):
        cfg = SimilarityConfig(group_max_weight=lam)
        same, diff = [], []
        for _ in range(reps):
            def win(shift):
                X = rng.normal(0, 1, (80, d))
                X[:, 0] = rng.standard_cauchy(80) * np.exp(rng.normal(0, 1))      # irrelevant, unstable feature
                X[:, 1:] += shift
                return extract_feature_fp(X, t=np.arange(80))
            same.append(compare(win(0), win(0), cfg).combined)
            diff.append(compare(win(0), win(0.5), cfg).combined)         # a real 0.5-sigma shift of 9 features
        res[label] = {"D_same_mean": float(np.mean(same)), "D_same_q95": float(np.quantile(same, .95)),
                      "D_diff_mean": float(np.mean(diff)), "AUC": auc(diff, same)}
    return res


def check_selection_bias(rng, trials=20000, n=30, m=3, acc=0.80) -> dict:
    """m candidate models of identical true accuracy, best picked on the same n labelled samples."""
    hits = rng.binomial(n, acc, (trials, m)) / n
    best = hits.max(axis=1)
    thr_pass = (best >= 0.70).mean()
    return {"true_accuracy": acc, "mean_accuracy_of_selected_on_selection_set": float(best.mean()),
            "optimism": float(best.mean() - acc), "n": n, "candidates": m}


def check_noise_floor_power(rng, reps=300) -> dict:
    """Does subtracting kappa*nu hide genuine small shifts at n=40?"""
    out = {}
    for shift in (0.2, 0.4, 0.8):
        spec0, spec1 = base_spec(4), Spec("s", tuple([shift] * 4), (1.0,) * 4)
        for kappa in (1.0, 0.0):
            cfg = SimilarityConfig(kappa=kappa)
            same = [compare(window(spec0, 40, rng), window(spec0, 40, rng), cfg).combined for _ in range(reps)]
            diff = [compare(window(spec0, 40, rng), window(spec1, 40, rng), cfg).combined for _ in range(reps)]
            out[f"shift={shift} kappa={kappa}"] = {"AUC": auc(diff, same), "D_same_q95": float(np.quantile(same, .95)),
                                                    "D_diff_median": float(np.median(diff))}
    return out


def check_eviction() -> dict:
    repo = ConceptRepository(match=MatchConfig(max_concepts=3))
    rng = np.random.default_rng(0)
    ids = []
    for i in range(3):
        ids.append(repo.add(window(base_spec(4, name=str(i)), 60, rng), f"m{i}", t=i).concept_id)
    repo.records[ids[0]].uses = 9                                   # a frequently recurring, old concept
    repo.add(window(base_spec(4), 60, rng), "m3", t=10)             # evicts the least used
    repo.add(window(base_spec(4), 60, rng), "m4", t=11)             # a *genuine* new concept created at t=10 is now at risk
    return {"survivors": sorted(r.model_id for r in repo.records.values()),
            "note": "policy = min(uses, last_used_t); ignores model performance and recurrence period"}


def main(out="results/research/audit.json"):
    rng = np.random.default_rng(12345)
    res = {"merge_vs_truth": check_merge(rng), "ess_by_statistic": check_ess(rng), "intra_single_window": check_intra_single_window(rng),
           "level1_blindspot": check_level1_blindspot(rng), "group_aggregation": check_group_agg(rng),
           "selection_bias": check_selection_bias(rng), "noise_floor_power": check_noise_floor_power(rng),
           "eviction": check_eviction()}
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
