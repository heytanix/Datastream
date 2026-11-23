"""M3: which distribution metrics are defensible, and does a (standardised) temporal view add retrieval power?"""
from __future__ import annotations

import json
import time

import numpy as np

from ..mvfp import extract_feature_fp
from ..similarity import SimilarityConfig, compare
from .gen import auc

D = 3
N = 80


def _std(x):
    return (x - x.mean(0)) / x.std(0)


def shape_family(name, n, rng):
    """Zero-mean, unit-variance samples so that (mostly) only shape differs from the normal reference."""
    if name == "normal":
        return rng.normal(0, 1, (n, D))
    if name == "uniform":
        return rng.uniform(-np.sqrt(3), np.sqrt(3), (n, D))
    if name == "bimodal":
        return _std(rng.choice([-1.0, 1.0], (n, D)) * 1.0 + rng.normal(0, 0.45, (n, D)))
    if name == "laplace":
        return rng.laplace(0, 1 / np.sqrt(2), (n, D))
    if name == "skewed":
        return _std(rng.exponential(1.0, (n, D)))
    if name == "heavy_tail":                 # rescaled: middle quantiles move too
        x = rng.normal(0, 1, (n, D))
        m = rng.random((n, D)) < 0.06
        return np.where(m, x * 4, x) / 1.8
    if name == "heavy_tail_core":            # 6% x4 outliers, middle quantiles UNCHANGED (the audit's blind spot)
        x = rng.normal(0, 1, (n, D))
        m = rng.random((n, D)) < 0.06
        return np.where(m, x * 4, x)
    if name == "shift_0.5":
        return rng.normal(0.5, 1, (n, D))
    if name == "scale_1.4":
        return rng.normal(0, 1.4, (n, D))
    raise KeyError(name)


FAMILIES = ["uniform", "bimodal", "laplace", "skewed", "heavy_tail", "shift_0.5", "scale_1.4"]
METRICS = {
    "A wasserstein": dict(dist_metric="wasserstein"),
    "B psi (a-deciles, schema 2.0)": dict(dist_metric="psi"),
    "B' psi (pooled bins)": dict(dist_metric="psi", psi_binning="pooled"),
    "C wasserstein+psi (schema 2.0)": dict(dist_metric="both"),
    "C' wasserstein+psi (pooled)": dict(dist_metric="both", psi_binning="pooled"),
    "D wasserstein+JS": dict(dist_metric="wasserstein+js"),
    "JS only": dict(dist_metric="js"),
    "E wasserstein+energy": dict(dist_metric="wasserstein+energy"),
    "energy only": dict(dist_metric="energy"),
}


def fpw(X):
    return extract_feature_fp(X, t=np.arange(len(X)))


def exp_distribution(reps=60, seed=3, n=N):
    rng = np.random.default_rng(seed)
    pairs = {"same": [(fpw(shape_family("normal", n, rng)), fpw(shape_family("normal", n, rng))) for _ in range(reps)]}
    for f in FAMILIES:
        pairs[f] = [(fpw(shape_family("normal", n, rng)), fpw(shape_family(f, n, rng))) for _ in range(reps)]
    res = {}
    for name, kw in METRICS.items():
        cfg = SimilarityConfig(**kw)
        t0 = time.perf_counter()
        comp = {k: np.array([compare(a, b, cfg).components["dist"] for a, b in v]) for k, v in pairs.items()}
        us = 1e6 * (time.perf_counter() - t0) / sum(len(v) for v in pairs.values())
        res[name] = {"auc": {f: auc(comp[f], comp["same"]) for f in FAMILIES}, "same_q95": float(np.quantile(comp["same"], .95)),
                     "us_per_compare": us}
        res[name]["mean_auc"] = float(np.mean(list(res[name]["auc"].values())))
    # redundancy: rank correlation of the two ingredients across every pair
    allp = [p for v in pairs.values() for p in v]
    cw, cp = SimilarityConfig(dist_metric="wasserstein"), SimilarityConfig(dist_metric="psi")
    ww = np.array([compare(a, b, SimilarityConfig(dist_metric="wasserstein", noise_mode="none")).components["dist"] for a, b in allp])
    pp = np.array([compare(a, b, SimilarityConfig(dist_metric="psi", noise_mode="none")).components["dist"] for a, b in allp])
    rk = lambda v: np.argsort(np.argsort(v))
    res["_redundancy_spearman_W1_vs_PSI"] = float(np.corrcoef(rk(ww), rk(pp))[0, 1])
    return res


def exp_tail(reps=60, seed=4, n=N):
    rng = np.random.default_rng(seed)
    same = [(fpw(shape_family("normal", n, rng)), fpw(shape_family("normal", n, rng))) for _ in range(reps)]
    out = {}
    for fam in ("heavy_tail_core", "heavy_tail", "laplace", "bimodal"):
        diff = [(fpw(shape_family("normal", n, rng)), fpw(shape_family(fam, n, rng))) for _ in range(reps)]
        for tw in (0.0, 1.0, 2.0):
            cfg = SimilarityConfig(tail_weight=tw)
            s = [compare(a, b, cfg).components["stat"] for a, b in same]
            d = [compare(a, b, cfg).components["stat"] for a, b in diff]
            out[f"{fam} tail_weight={tw}"] = {"auc_stat_view": auc(d, s), "same_q95": float(np.quantile(s, .95))}
    return out


# ---- temporal ----
def ar(n, rho, rng):
    z = np.empty((n, D))
    z[0] = rng.normal(0, 1, D)
    e = rng.normal(0, 1, (n, D)) * np.sqrt(1 - rho ** 2)
    for i in range(1, n):
        z[i] = rho * z[i - 1] + e[i]
    return z


def temporal_family(name, n, rng):
    t = np.arange(n)[:, None]
    noise = rng.normal(0, 0.5, (n, D))
    ph = rng.uniform(0, 2 * np.pi, D)
    if name == "iid":
        return rng.normal(0, 1, (n, D))
    if name == "ar0.9":
        return ar(n, 0.9, rng)
    if name == "sine12":
        return _std(np.sin(2 * np.pi * t / 12 + ph) + noise)
    if name == "sine25":
        return _std(np.sin(2 * np.pi * t / 25 + ph) + noise)
    if name == "trend_up":
        return _std(t / n * 3 + rng.normal(0, 1, (n, D)))
    if name == "trend_down":
        return _std(-t / n * 3 + rng.normal(0, 1, (n, D)))
    if name == "no_trend":
        return rng.normal(0, 1, (n, D))
    if name == "step":
        return _std((t > n // 2) * 2.0 + rng.normal(0, 1, (n, D)))
    if name == "ramp":
        return _std(np.clip((t - n * 0.3) / (n * 0.4), 0, 1) * 2.0 + rng.normal(0, 1, (n, D)))
    raise KeyError(name)


TEMPORAL_TASKS = {  # (reference process, different process)
    "ordering (iid vs AR 0.9)": ("iid", "ar0.9"),
    "periodicity (period 12 vs 25)": ("sine12", "sine25"),
    "trend (up vs down, same quantiles)": ("trend_up", "trend_down"),
    "trend (flat vs up)": ("no_trend", "trend_up"),
    "abrupt vs gradual (step vs ramp)": ("step", "ramp"),
}


def exp_temporal(reps=50, seed=9, n=120):
    rng = np.random.default_rng(seed)
    cfgs = {"legacy time": SimilarityConfig(), "standardised time (legacy ESS)": SimilarityConfig(time_norm="standardised"),
            "standardised time (per-stat ESS)": SimilarityConfig(time_norm="standardised", ess_mode="per_stat")}
    res = {}
    for task, (ref, other) in TEMPORAL_TASKS.items():
        same = [(fpw(temporal_family(ref, n, rng)), fpw(temporal_family(ref, n, rng))) for _ in range(reps)]
        diff = [(fpw(temporal_family(ref, n, rng)), fpw(temporal_family(other, n, rng))) for _ in range(reps)]
        row = {}
        for name, cfg in cfgs.items():
            s = [compare(a, b, cfg).components["time"] for a, b in same]
            d = [compare(a, b, cfg).components["time"] for a, b in diff]
            row[f"AUC {name}"] = auc(d, s)
            row[f"same q95 {name}"] = float(np.quantile(s, .95))
        # retrieval power of the combined distance with and without the temporal view
        for label, w in (("stat+dist only", {"stat": .4, "dist": .3, "time": 0, "pred": 0}),
                         ("stat+dist+time (w=0.1, legacy time)", {"stat": .4, "dist": .3, "time": .1, "pred": 0}),
                         ("stat+dist+time (w=0.3, standardised)", {"stat": .4, "dist": .3, "time": .3, "pred": 0})):
            cfg = SimilarityConfig(weights=w, time_norm="standardised" if "standardised" in label else "legacy",
                                   ess_mode="per_stat" if "standardised" in label else "legacy")
            s = [compare(a, b, cfg).combined for a, b in same]
            d = [compare(a, b, cfg).combined for a, b in diff]
            row[f"combined AUC {label}"] = auc(d, s)
        res[task] = row
    return res


def exp_correlation(reps=50, seed=21):
    """Same marginals, different dependence: |delta r| versus Fisher-z, by dimension and window size, plus missing values."""
    rng = np.random.default_rng(seed)
    out = {}
    for d in (4, 10, 30):
        for n in (40, 80):
            def draw(c, miss=0.0):
                Z = rng.normal(0, 1, (n, d))
                if c:
                    L = np.linalg.cholesky((1 - c) * np.eye(d) + c * np.ones((d, d)))
                    Z = Z @ L.T
                if miss:
                    Z[rng.random(Z.shape) < miss] = np.nan
                return fpw(Z)
            for c in (0.3, 0.6):
                same = [(draw(0), draw(0)) for _ in range(reps)]
                diff = [(draw(0), draw(c)) for _ in range(reps)]
                for tr in ("abs", "fisher"):
                    cfg = SimilarityConfig(corr_transform=tr, ess_mode="per_stat" if tr == "fisher" else "legacy", weights={"stat": 0, "dist": 1, "time": 0, "pred": 0})
                    s = [compare(a, b, cfg).components["dist"] for a, b in same]
                    df = [compare(a, b, cfg).components["dist"] for a, b in diff]
                    out[f"d={d} n={n} corr={c} {tr}"] = {"auc": auc(df, s), "same_q95": float(np.quantile(s, .95))}
    # missing values and a constant feature must not break anything
    X = rng.normal(0, 1, (60, 6)); X[:, 2] = 1.0; X[rng.random(X.shape) < 0.1] = np.nan
    d = compare(fpw(X), fpw(X.copy()), SimilarityConfig(corr_transform="fisher"))
    out["_missing_and_constant_ok"] = bool(np.isfinite(d.combined))
    return out


def main(out="results/research/e2_e3_dist_time.json"):
    res = {"distribution_metrics": exp_distribution(), "tail_term": exp_tail(), "temporal": exp_temporal(), "correlation": exp_correlation()}
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    return res


def to_md(res) -> str:
    L = ["### Distribution metrics (D_dist view, n=80, AUC same-vs-different by family)\n", "| metric | " + " | ".join(FAMILIES) + " | mean AUC | same q95 | µs/compare |", "|---|" + "---|" * (len(FAMILIES) + 3)]
    for k, v in res["distribution_metrics"].items():
        if k.startswith("_"):
            continue
        L.append(f"| {k} | " + " | ".join(f"{v['auc'][f]:.2f}" for f in FAMILIES) + f" | {v['mean_auc']:.3f} | {v['same_q95']:.3f} | {v['us_per_compare']:.0f} |")
    L.append(f"\nSpearman correlation of Wasserstein and PSI distances over all pairs: {res['distribution_metrics']['_redundancy_spearman_W1_vs_PSI']:.3f}\n")
    L.append("### Tail term in D_stat (AUC of the statistics view)\n\n| case | AUC | same q95 |\n|---|---|---|")
    for k, v in res["tail_term"].items():
        L.append(f"| {k} | {v['auc_stat_view']:.2f} | {v['same_q95']:.3f} |")
    L.append("\n### Temporal tasks (n=120)\n")
    cols = list(next(iter(res["temporal"].values())).keys())
    L.append("| task | " + " | ".join(c.replace("|", "/") for c in cols) + " |\n|---|" + "---|" * len(cols))
    for t, row in res["temporal"].items():
        L.append(f"| {t} | " + " | ".join(f"{row[c]:.2f}" for c in cols) + " |")
    L.append("\n### Correlation comparison (AUC of D_dist, same marginals / different dependence)\n\n| case | AUC | same q95 |\n|---|---|---|")
    for k, v in res["correlation"].items():
        if not k.startswith("_"):
            L.append(f"| {k} | {v['auc']:.2f} | {v['same_q95']:.3f} |")
    L.append(f"\nmissing values + a constant feature handled: {res['correlation']['_missing_and_constant_ok']}")
    return "\n".join(L)


if __name__ == "__main__":
    r = main()
    md = to_md(r)
    open("results/research/e2_e3_dist_time.md", "w").write(md)
    print(md)
