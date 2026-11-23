"""M5: thresholds (false-match vs missed-match) and match-confidence calibration, with calibration/evaluation separation."""
from __future__ import annotations

import json
import math
import time

import numpy as np

from ..matcher import ConceptRepository, Decision, MatchCalibrator, MatchConfig, conformal_quantile, match_features
from ..mvfp import extract_feature_fp
from ..similarity import compare
from .exp_retrieval import QUERY_KINDS, build_repo, combine, make_library, make_queries
from .gen import auc, sample
from .presets import improved_similarity

SIM = improved_similarity(weights={"stat": 0.4, "dist": 0.2, "time": 0.4, "pred": 0.2})
W = [0.4, 0.2, 0.4]


def dmatrix(repo, qs):
    cids = list(repo.records)
    M = np.full((len(qs), len(cids), 3), np.nan)
    for i, (_, _, q) in enumerate(qs):
        for j, c in enumerate(cids):
            comps = compare(repo.records[c].fp, q, SIM, views=("stat", "dist", "time")).components
            M[i, j] = [np.nan if comps[v] is None else comps[v] for v in ("stat", "dist", "time")]
    return cids, combine(M, W)


def same_distances(repo, specs, rng, n, K=10):
    """K windows of size n per concept, from fresh RNG draws (never the evaluation queries)."""
    out = {}
    for name, sp in specs.items():
        cid = next(c for c, r in repo.records.items() if r.model_id == name)
        out[cid] = [compare(repo.records[cid].fp, extract_feature_fp(sample(sp, n, rng), t=np.arange(n)), SIM, views=("stat", "dist", "time")).combined for _ in range(K)]
    return out


def policy_eval(Dm, cids, repo, qs, thr_of):
    """thr_of(cid) -> accept limit. Recurring: truth in repo. Novel: truth column masked out (hard negatives remain)."""
    truth = np.array([cids.index(next(c for c, r in repo.records.items() if r.model_id == n)) for n, _, _ in qs])
    rec_ok = rec_wrong = rec_missed = nov_false = 0
    for i, t in enumerate(truth):
        b = int(Dm[i].argmin())
        thr = thr_of(cids[b])
        if Dm[i, b] <= thr:
            rec_ok += b == t
            rec_wrong += b != t
        else:
            rec_missed += 1
        Dn = Dm[i].copy(); Dn[t] = np.inf
        b2 = int(Dn.argmin())
        nov_false += Dn[b2] <= thr_of(cids[b2])
    n = len(truth)
    return {"correct_recall": rec_ok / n, "wrong_concept": rec_wrong / n, "missed_match": rec_missed / n, "false_match_novel": nov_false / n}


def exp_thresholds(seeds=(1, 2), conditions=((4, 40), (4, 80), (10, 40), (10, 80))):
    t0 = time.time()
    res = {}
    for d, n in conditions:
        rows = {}
        for seed in seeds:
            rng = np.random.default_rng(seed * 7 + d)
            specs = make_library(rng, d=d)
            repo = build_repo(specs, np.random.default_rng(seed + 100), MatchConfig(), SIM)
            qs = [(nm, k, q) for nm, k, q in make_queries(specs, rng) if (len(q.num_idx) and q.n == n)]
            cids, Dm = dmatrix(repo, qs)
            same = same_distances(repo, specs, np.random.default_rng(seed + 5000), n)           # separate RNG stream
            pool = np.concatenate([np.array(v) for v in same.values()])
            policies = {}
            for t in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
                policies[f"fixed {t:.2f}"] = lambda c, t=t: t
            intra = {}
            for c, v in same.items():
                e = v[0]
                for x in v[1:]:
                    e = 0.8 * e + 0.2 * x
                intra[c] = e
            policies["legacy 2.5 x intra (floor 0.32)"] = lambda c: max(0.32, 2.5 * intra[c])
            for a in (0.02, 0.05, 0.10):
                policies[f"conformal per-concept alpha={a}"] = lambda c, a=a: float(np.clip(conformal_quantile(np.array(same[c]), 1 - a), 0.05, 0.9))
                policies[f"conformal global pool alpha={a}"] = lambda c, a=a: float(np.clip(conformal_quantile(pool, 1 - a), 0.05, 0.9))
            for name, f in policies.items():
                for k, v in policy_eval(Dm, cids, repo, qs, f).items():
                    rows.setdefault(name, {}).setdefault(k, []).append(v)
        res[f"d={d} n={n}"] = {p: {k: (float(np.mean(v)), float(np.std(v))) for k, v in m.items()} for p, m in rows.items()}
    res["seconds"] = round(time.time() - t0, 1)
    return res


# ---- confidence calibration ----
def collect_matches(seed, ns=(40, 80), d=4):
    rng = np.random.default_rng(seed)
    specs = make_library(rng, d=d)
    mc = MatchConfig(top_k_mode="adaptive", expand_on_new=True, threshold_mode="empirical", min_obs=5)
    repo = build_repo(specs, np.random.default_rng(seed + 100), mc, SIM)
    for n in ns:                                                    # calibrate thresholds at the query sizes
        for name, sp in specs.items():
            cid = next(c for c, r in repo.records.items() if r.model_id == name)
            for _ in range(8):
                repo.observe_same(cid, extract_feature_fp(sample(sp, n, np.random.default_rng(rng.integers(1 << 30))), t=np.arange(n)))
    qs = make_queries(specs, rng)
    rows = []
    for name, kind, q in qs:
        truth = next(c for c, r in repo.records.items() if r.model_id == name)
        for novel in (False, True):
            res = repo.match(q, exclude={truth} if novel else None)
            if res.decision in (Decision.HIGH_CONFIDENCE_MATCH, Decision.CANDIDATE_MATCH):
                f = match_features(res)
                if f is not None:
                    rows.append({"f": f, "heuristic": res.confidence, "correct": int(res.best_id == truth), "novel": novel,
                                 "decision": res.decision.value, "n": q.n})
    return rows


def calib_metrics(p, y, bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    brier = float(np.mean((p - y) ** 2))
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rel, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            rel.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(m.sum()), "mean_pred": float(p[m].mean()), "observed": float(y[m].mean())})
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return {"brier": brier, "ece": float(ece), "auroc": auc(p[y == 1], p[y == 0]) if 0 < y.sum() < len(y) else float("nan"), "reliability": rel}


def exp_confidence(cal_seeds=(31, 32, 33), test_seeds=(1, 2)):
    cal = [r for s in cal_seeds for r in collect_matches(s)]
    test = [r for s in test_seeds for r in collect_matches(s)]
    X, y = np.array([r["f"] for r in cal]), np.array([r["correct"] for r in cal])
    model = MatchCalibrator().fit(X, y)
    pt = model.predict(np.array([r["f"] for r in test]))
    yt = np.array([r["correct"] for r in test])
    heur = np.array([r["heuristic"] for r in test])
    novel = np.array([r["novel"] for r in test])
    out = {"n_calibration": len(cal), "n_test": len(test), "base_rate_cal": float(y.mean()), "base_rate_test": float(yt.mean()),
           "coefficients": dict(zip(("D", "margin_norm", "log_n_eff", "completeness", "var_ratio", "has_pred", "bias"), [float(w) for w in model.w])),
           "constant_baseline_brier": float(np.mean((y.mean() - yt) ** 2)),
           "heuristic_score": calib_metrics(heur, yt), "calibrated_probability": calib_metrics(pt, yt)}
    out["precision_at"] = {}
    for name, score in (("heuristic_score", heur), ("calibrated_probability", pt)):
        rows = {}
        for tau in (0.3, 0.5, 0.7, 0.9):
            m = score >= tau
            rows[str(tau)] = {"n": int(m.sum()), "precision": float(yt[m].mean()) if m.any() else None,
                              "false_match_rate_among_novel_queries": float(np.mean(m[novel])) if novel.any() else None}
        out["precision_at"][name] = rows
    return out


def to_md(thr, conf) -> str:
    L = ["### Threshold policies (improved Level 2; calibration windows from separate RNG streams; 2 seeds; mean)\n"]
    for cond, pols in thr.items():
        if cond == "seconds":
            continue
        L.append(f"\n**{cond}**\n\n| policy | correct recall | wrong concept | missed match | false match (novel) |\n|---|---|---|---|---|")
        for p, m in pols.items():
            L.append(f"| {p} | {m['correct_recall'][0]:.3f} | {m['wrong_concept'][0]:.3f} | {m['missed_match'][0]:.3f} | {m['false_match_novel'][0]:.3f} |")
    L.append("\n### Confidence calibration (held-out seeds)\n")
    L.append(f"calibration matches {conf['n_calibration']}, test matches {conf['n_test']}, base rate of correct best candidate: cal {conf['base_rate_cal']:.3f}, test {conf['base_rate_test']:.3f}")
    L.append(f"\nconstant-base-rate Brier: {conf['constant_baseline_brier']:.4f}\n\n| score | Brier | ECE | AUROC |\n|---|---|---|---|")
    for k in ("heuristic_score", "calibrated_probability"):
        m = conf[k]
        L.append(f"| {k} | {m['brier']:.4f} | {m['ece']:.4f} | {m['auroc']:.3f} |")
    L.append("\nReliability (calibrated probability):\n\n| bin | n | mean predicted | observed |\n|---|---|---|---|")
    for r in conf["calibrated_probability"]["reliability"]:
        L.append(f"| {r['bin']} | {r['n']} | {r['mean_pred']:.3f} | {r['observed']:.3f} |")
    L.append("\nReliability (heuristic score):\n\n| bin | n | mean predicted | observed |\n|---|---|---|---|")
    for r in conf["heuristic_score"]["reliability"]:
        L.append(f"| {r['bin']} | {r['n']} | {r['mean_pred']:.3f} | {r['observed']:.3f} |")
    L.append("\nPrecision and false-match rate above a confidence cut-off:\n\n| score | cut-off | n accepted | precision | share of novel-concept queries accepted |\n|---|---|---|---|---|")
    for k, rows in conf["precision_at"].items():
        for tau, v in rows.items():
            L.append(f"| {k} | {tau} | {v['n']} | {'n/a' if v['precision'] is None else format(v['precision'], '.3f')} | {'n/a' if v['false_match_rate_among_novel_queries'] is None else format(v['false_match_rate_among_novel_queries'], '.3f')} |")
    L.append(f"\nCalibrator coefficients (standardised features): {conf['coefficients']}")
    return "\n".join(L)


if __name__ == "__main__":
    thr = exp_thresholds()
    conf = exp_confidence()
    json.dump({"thresholds": thr, "confidence": conf}, open("results/research/e7_e8_thresholds_confidence.json", "w"), indent=1)
    md = to_md(thr, conf)
    open("results/research/e7_e8_thresholds_confidence.md", "w").write(md)
    print(md)
    print("seconds", thr["seconds"])
