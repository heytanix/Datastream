"""M4: Level-1 candidate recall, adaptive top-k, view ablation, calibrated weights, group aggregation."""
from __future__ import annotations

import itertools
import json
import time

import numpy as np

from ..matcher import ConceptRepository, MatchConfig
from ..mvfp import extract_feature_fp
from ..similarity import SimilarityConfig, compare
from .gen import Spec, auc, sample
from .presets import improved_similarity

D = 4
VARIANTS = ["core", "heavy_tail", "bimodal", "correlated", "autocorr", "shifted", "wider"]
QUERY_KINDS = ["plain", "shifted", "missing_feature", "noisy"]


def make_library(rng, n_bases=6, d=D):
    specs = {}
    for b in range(n_bases):
        mu = rng.normal(0, 2.5, d)
        sg = rng.uniform(0.7, 1.5, d)
        kw = dict(mu=tuple(mu), sigma=tuple(sg))
        mods = {"core": {}, "heavy_tail": {"tail": 0.06}, "bimodal": {"bimodal": 2.2}, "correlated": {"corr": 0.7},
                "autocorr": {"rho": 0.9}, "shifted": {"mu": tuple(mu + 0.35 * sg)}, "wider": {"sigma": tuple(sg * 1.4)}}
        for v, m in mods.items():
            specs[f"b{b}_{v}"] = Spec(f"b{b}_{v}", **{**kw, **m})
    return specs


def make_queries(specs, rng, per_kind=1):
    qs = []
    for name, sp in specs.items():
        for i, kind in enumerate(QUERY_KINDS):
            n = (40, 80)[i % 2]
            if kind == "plain":
                X = sample(sp, n, rng)
            elif kind == "shifted":
                X = sample(Spec(sp.name, tuple(np.asarray(sp.mu) + 0.15 * np.asarray(sp.sigma)), sp.sigma, sp.rho, sp.corr, sp.tail, sp.bimodal), n, rng)
            elif kind == "missing_feature":
                X = sample(sp, n, rng)
                X[rng.random(n) < 0.3, 0] = np.nan
            else:
                X = sample(Spec(sp.name, sp.mu, sp.sigma, sp.rho, sp.corr, sp.tail, sp.bimodal, noise=0.3), n, rng)
            qs.append((name, kind, extract_feature_fp(X, t=np.arange(n))))
    return qs


def build_repo(specs, rng, cfg_match, cfg_sim):
    repo = ConceptRepository(cfg_match, cfg_sim)
    for name, sp in specs.items():
        repo.add(extract_feature_fp(sample(sp, 300, rng), t=np.arange(300)), name)
    return repo


def l1_ranks(repo, qs):
    out = []
    for name, kind, q in qs:
        ids, sc = repo.level1_scores(q)
        order = np.argsort(sc, kind="stable")
        pos = {ids[i]: r for r, i in enumerate(order)}
        out.append(pos[next(c for c, r in repo.records.items() if r.model_id == name)])
    return np.array(out)


def full_matrix(repo, qs, cfg_sim):
    """Components D_stat, D_dist, D_time against every stored concept: (Q, C, 3), concept order = repo order."""
    cids = list(repo.records)
    M = np.full((len(qs), len(cids), 3), np.nan)
    for i, (_, _, q) in enumerate(qs):
        for j, c in enumerate(cids):
            comps = compare(repo.records[c].fp, q, cfg_sim, views=("stat", "dist", "time")).components
            M[i, j] = [np.nan if comps[v] is None else comps[v] for v in ("stat", "dist", "time")]
    return cids, M


def combine(M, w):
    w = np.asarray(w, float)
    ok = np.isfinite(M)
    num = np.nansum(M * w[None, None, :], axis=2)
    den = np.sum(ok * w[None, None, :], axis=2)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 1.0)


def top1_acc(Dm, cids, repo, qs):
    truth = [next(c for c, r in repo.records.items() if r.model_id == n) for n, _, _ in qs]
    pred = [cids[i] for i in Dm.argmin(axis=1)]
    return float(np.mean([p == t for p, t in zip(pred, truth)])), truth


def shortlist_acc(repo, qs, Dm, cids, truth):
    """End-to-end top-1: restrict to the Level-1 shortlist, then rank by Level 2 (D matrix reused)."""
    hit, sizes, rec = 0, [], 0
    for i, (_, _, q) in enumerate(qs):
        sl = repo.level1(q)
        sizes.append(len(sl))
        rec += truth[i] in sl
        idx = [cids.index(c) for c in sl]
        hit += cids[idx[int(np.argmin(Dm[i, idx]))]] == truth[i]
    return hit / len(qs), float(np.mean(sizes)), rec / len(qs)


def exp_level1(seeds=(1, 2, 3), reprs=("basic", "shape", "shape_corr", "full")):
    res = {r: {"recall@1": [], "recall@3": [], "recall@5": [], "recall@10": [], "by_variant@5": {v: [] for v in VARIANTS},
               "by_kind@5": {k: [] for k in QUERY_KINDS}} for r in reprs}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        specs = make_library(rng)
        qs = make_queries(specs, rng)
        for r in reprs:
            repo = build_repo(specs, np.random.default_rng(seed + 100), MatchConfig(l1_repr=r), SimilarityConfig())
            rk = l1_ranks(repo, qs)
            for k in (1, 3, 5, 10):
                res[r][f"recall@{k}"].append(float(np.mean(rk < k)))
            for v in VARIANTS:
                res[r]["by_variant@5"][v].append(float(np.mean([rk[i] < 5 for i, (n, _, _) in enumerate(qs) if n.endswith(v)])))
            for kd in QUERY_KINDS:
                res[r]["by_kind@5"][kd].append(float(np.mean([rk[i] < 5 for i, (_, k_, _) in enumerate(qs) if k_ == kd])))
    ms = lambda v: (float(np.mean(v)), float(np.std(v)))
    return {r: {"recall": {k: ms(v) for k, v in d.items() if k.startswith("recall")},
                "by_variant@5": {v: ms(x) for v, x in d["by_variant@5"].items()}, "by_kind@5": {k: ms(x) for k, x in d["by_kind@5"].items()}}
            for r, d in res.items()}


def exp_pipeline(seeds=(1, 2, 3)):
    """Full-scan vs shortlist end-to-end top-1 for legacy and improved Level 2, fixed vs adaptive vs richer Level 1."""
    sims = {"legacy L2": SimilarityConfig(), "improved L2": improved_similarity()}
    rows = {}
    for sname, sim in sims.items():
        for seed in seeds:
            rng = np.random.default_rng(seed)
            specs = make_library(rng)
            qs = make_queries(specs, rng)
            base = build_repo(specs, np.random.default_rng(seed + 100), MatchConfig(), sim)
            cids, M = full_matrix(base, qs, sim)
            Dm = combine(M, [0.4, 0.3, 0.1])
            full, truth = top1_acc(Dm, cids, base, qs)
            rows.setdefault(f"{sname} | full scan", {"top1": [], "candidates": [], "candidate_recall": []})["top1"].append(full)
            for label, mc in (("basic fixed k=5 (schema 2.0)", MatchConfig()), ("basic adaptive", MatchConfig(top_k_mode="adaptive")),
                              ("shape_corr fixed k=5", MatchConfig(l1_repr="shape_corr")), ("shape_corr adaptive", MatchConfig(l1_repr="shape_corr", top_k_mode="adaptive")),
                              ("full adaptive", MatchConfig(l1_repr="full", top_k_mode="adaptive"))):
                repo = build_repo(specs, np.random.default_rng(seed + 100), mc, sim)
                acc, size, rec = shortlist_acc(repo, qs, Dm, cids, truth)
                r = rows.setdefault(f"{sname} | {label}", {"top1": [], "candidates": [], "candidate_recall": []})
                r["top1"].append(acc); r["candidates"].append(size); r["candidate_recall"].append(rec)
    return {k: {m: (float(np.mean(v)), float(np.std(v))) for m, v in d.items() if v} for k, d in rows.items()}


def exp_views_and_weights(cal_seeds=(11, 12, 13), test_seeds=(1, 2, 3)):
    sim = improved_similarity()
    def collect(seeds):
        out = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            specs = make_library(rng)
            qs = make_queries(specs, rng)
            repo = build_repo(specs, np.random.default_rng(seed + 100), MatchConfig(), sim)
            cids, M = full_matrix(repo, qs, sim)
            truth = [cids.index(next(c for c, r in repo.records.items() if r.model_id == n)) for n, _, _ in qs]
            out.append((M, np.array(truth)))
        return out
    cal, test = collect(cal_seeds), collect(test_seeds)

    def acc(data, w):
        return float(np.mean([np.mean(combine(M, w).argmin(axis=1) == t) for M, t in data]))

    views = {"stat only": [1, 0, 0], "dist only": [0, 1, 0], "time only": [0, 0, 1], "stat+dist": [.5, .5, 0], "stat+time": [.5, 0, .5],
             "dist+time": [0, .5, .5], "stat+dist+time (0.4/0.3/0.1)": [.4, .3, .1], "equal thirds": [1 / 3, 1 / 3, 1 / 3]}
    res = {"views": {k: {"test_top1": acc(test, w)} for k, w in views.items()}}
    # calibrated weights: grid search on the simplex (step 0.1) over CALIBRATION seeds only
    best, bw = -1, None
    for a in range(0, 11):
        for b in range(0, 11 - a):
            w = [a / 10, b / 10, (10 - a - b) / 10]
            s = acc(cal, w)
            if s > best + 1e-9:
                best, bw = s, w
    res["calibrated_weights"] = {"weights(stat,dist,time)": bw, "calibration_top1": best, "heldout_top1": acc(test, bw),
                                 "fixed_0.4/0.3/0.1_heldout_top1": acc(test, [.4, .3, .1])}
    # view dependence: rank correlation of per-pair distances across views (different-concept pairs only)
    allM = np.concatenate([M.reshape(-1, 3) for M, _ in test])
    ok = np.isfinite(allM).all(axis=1)
    rk = lambda v: np.argsort(np.argsort(v))
    res["view_dependence_spearman"] = {f"{a}-{b}": float(np.corrcoef(rk(allM[ok, i]), rk(allM[ok, j]))[0, 1])
                                       for (a, i), (b, j) in itertools.combinations([("stat", 0), ("dist", 1), ("time", 2)], 2)}
    return res


def exp_aggregation(reps=60, seed=5, n=80, d=10):
    """One unstable irrelevant feature vs one genuinely changed feature: no aggregator wins both unless robust."""
    rng = np.random.default_rng(seed)
    aggs = {"mean_worst (0.7/0.3, schema 2.0)": dict(group_agg="mean_worst"), "mean": dict(group_agg="mean"), "median": dict(group_agg="median"),
            "trimmed (20%)": dict(group_agg="trimmed", trim=0.2), "quantile 0.75": dict(group_agg="quantile", group_quantile=0.75),
            "quantile 0.9": dict(group_agg="quantile", group_quantile=0.9)}
    def win(shift_all=0.0, shift_one=0.0, unstable=True):
        X = rng.normal(0, 1, (n, d))
        if unstable:
            X[:, 0] = rng.standard_cauchy(n) * np.exp(rng.normal(0, 1))
        X[:, 1:] += shift_all
        X[:, 3] += shift_one
        return extract_feature_fp(X, t=np.arange(n))
    scen = {"A: unstable noise feature; 9 features shift 0.5": (lambda: win(0, 0), lambda: win(0.5, 0)),
            "B: unstable noise feature; ONE feature shifts 1.5": (lambda: win(0, 0), lambda: win(0, 1.5)),
            "C: no unstable feature; ONE feature shifts 1.5": (lambda: win(0, 0, False), lambda: win(0, 1.5, False)),
            "D: no unstable feature; 9 features shift 0.5": (lambda: win(0, 0, False), lambda: win(0.5, 0, False))}
    res = {}
    for sc, (same_f, diff_f) in scen.items():
        pairs_s = [(same_f(), same_f()) for _ in range(reps)]
        pairs_d = [(same_f(), diff_f()) for _ in range(reps)]
        res[sc] = {}
        for an, kw in aggs.items():
            cfg = SimilarityConfig(**kw)
            s = [compare(a, b, cfg).components["stat"] for a, b in pairs_s]
            df = [compare(a, b, cfg).components["stat"] for a, b in pairs_d]
            res[sc][an] = {"auc": auc(df, s), "same_q95": float(np.quantile(s, .95))}
    return res


def main(out="results/research/e4_retrieval.json"):
    t0 = time.time()
    res = {"level1": exp_level1(), "pipeline": exp_pipeline(), "views_weights": exp_views_and_weights(), "aggregation": exp_aggregation()}
    res["seconds"] = round(time.time() - t0, 1)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    return res


def to_md(res) -> str:
    L = ["### Level 1 candidate recall (42 concepts, 6 look-alike families x 7; 3 seeds, mean±std)\n", "| representation | R@1 | R@3 | R@5 | R@10 |\n|---|---|---|---|---|"]
    for r, d in res["level1"].items():
        L.append(f"| {r} | " + " | ".join(f"{d['recall'][f'recall@{k}'][0]:.3f}±{d['recall'][f'recall@{k}'][1]:.3f}" for k in (1, 3, 5, 10)) + " |")
    L.append("\nRecall@5 by look-alike family:\n\n| representation | " + " | ".join(VARIANTS) + " |\n|---|" + "---|" * len(VARIANTS))
    for r, d in res["level1"].items():
        L.append(f"| {r} | " + " | ".join(f"{d['by_variant@5'][v][0]:.2f}" for v in VARIANTS) + " |")
    L.append("\nRecall@5 by query kind:\n\n| representation | " + " | ".join(QUERY_KINDS) + " |\n|---|" + "---|" * len(QUERY_KINDS))
    for r, d in res["level1"].items():
        L.append(f"| {r} | " + " | ".join(f"{d['by_kind@5'][k][0]:.2f}" for k in QUERY_KINDS) + " |")
    L.append("\n### End-to-end top-1 (shortlist then Level 2)\n\n| configuration | top-1 | mean candidates | candidate recall |\n|---|---|---|---|")
    for k, d in res["pipeline"].items():
        L.append(f"| {k} | {d['top1'][0]:.3f}±{d['top1'][1]:.3f} | " + (f"{d['candidates'][0]:.1f} | {d['candidate_recall'][0]:.3f}" if "candidates" in d else "all | 1.000") + " |")
    L.append("\n### View ablation (improved Level 2, full scan, held-out seeds)\n\n| views | top-1 |\n|---|---|")
    for k, d in res["views_weights"]["views"].items():
        L.append(f"| {k} | {d['test_top1']:.3f} |")
    cw = res["views_weights"]["calibrated_weights"]
    L.append(f"\nCalibrated weights (grid search on separate calibration seeds): {cw['weights(stat,dist,time)']} -> calibration top-1 {cw['calibration_top1']:.3f}, held-out {cw['heldout_top1']:.3f} (fixed 0.4/0.3/0.1 held-out {cw['fixed_0.4/0.3/0.1_heldout_top1']:.3f})")
    L.append(f"\nView dependence (Spearman of distances): {res['views_weights']['view_dependence_spearman']}")
    L.append("\n### Group aggregation (AUC of D_stat; same-concept q95 in brackets)\n")
    aggs = list(next(iter(res["aggregation"].values())).keys())
    L.append("| scenario | " + " | ".join(aggs) + " |\n|---|" + "---|" * len(aggs))
    for sc, row in res["aggregation"].items():
        L.append(f"| {sc} | " + " | ".join(f"{row[a]['auc']:.2f} ({row[a]['same_q95']:.2f})" for a in aggs) + " |")
    return "\n".join(L)


if __name__ == "__main__":
    r = main()
    md = to_md(r)
    open("results/research/e4_retrieval.md", "w").write(md)
    print(md)
    print("seconds", r["seconds"])
