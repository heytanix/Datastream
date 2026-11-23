"""Multi-seed, multi-scenario benchmark: static vs retrain vs DejaStream v1 vs DejaStream MV.

Retrieval metrics (computed from engine events and the ground-truth concept of each sample)
  A *resolution* is a ``recall`` or ``new_concept`` event after warm-up. Its true concept is the
  stream's concept at that moment. An engine concept is *owned* by the true concept that was in force
  when it was created. A recall is correct if the recalled concept's owner equals the true concept.
    TP  correct recall                       FP  recall of a concept owned by a different true concept
    FN  new_concept although an owner of the true concept already existed (missed match)
    TN  new_concept for a true concept never seen before
  precision = TP/(TP+FP)   retrieval recall = TP/(TP+FN)
  false-match rate = FP/(all resolutions)   missed-match rate = FN/(resolutions with an existing owner)
Model-recall success: a recall counts as successful if the recalled model's accuracy over the next
200 samples is within 0.08 of that run's own ceiling (90th percentile of its 200-sample rolling accuracy).
The ceiling is relative on purpose: label noise caps accuracy at different levels in different scenarios.
Everything is measured, nothing is hard-coded.
"""
from __future__ import annotations

import json
import time

import numpy as np

from .mvfp import extract_feature_fp
from .scenarios import SCENARIOS
from .strategies import LABELS, STRATEGIES, make_engine

SUCCESS_MARGIN, SUCCESS_WINDOW = 0.08, 200
from .similarity import SimilarityConfig

# extra, explicitly labelled rows: same engine, non-default configuration
VARIANTS = {"temporal_only": {"dejastream_mv[time-weighted]": {
    "sim": SimilarityConfig(weights={"stat": 0.2, "dist": 0.2, "time": 0.5, "pred": 0.1})}}}


def run_one(scenario, strategy: str, **engine_kw) -> dict:
    n_feat = scenario.X.shape[1]
    eng = make_engine(strategy, n_features=n_feat, **engine_kw)
    n = len(scenario)
    correct = np.zeros(n, int)
    events: list[dict] = []
    t0 = time.perf_counter()
    for t, x, y, _ in scenario:
        p, ev = eng.process(t, x, y)
        correct[t] = int(p == y)
        events.extend(ev)
    secs = time.perf_counter() - t0

    cps = scenario.change_points
    res = {"accuracy": float(correct.mean()), "us_per_sample": 1e6 * secs / n,
           "models_trained": int(eng.models_trained)}
    post = [correct[a:a + 300].mean() for a, _ in cps[1:]]
    res["post_drift_acc"] = float(np.mean(post)) if post else None
    drifts = [e["t"] for e in events if e["kind"] == "drift"]
    delays = []
    for a, _ in cps[1:]:
        nxt = [d for d in drifts if d >= a and d - a < 1000]
        if nxt:
            delays.append(nxt[0] - a)
    res["detection_delay"] = float(np.mean(delays)) if delays else None
    res["detection_rate"] = len(delays) / max(1, len(cps) - 1) if cps[1:] else None

    roll = np.convolve(correct, np.ones(SUCCESS_WINDOW) / SUCCESS_WINDOW, mode="valid")
    ceiling = float(np.quantile(roll, 0.9)) if len(roll) else 1.0
    owner: dict = {}
    tp = fp = fn = tn = recalls_ok = recalls = 0
    cand_total = high_n = high_ok = 0
    cand_hit = {1: 0, 3: 0, 5: 0}
    conf_rows: list = []
    for e in events:
        if e["kind"] == "new_concept":
            true_c = scenario.truth[min(e["t"], n - 1)]
            if not owner:                                   # the warm-up concept is not a resolution
                owner[e["concept"]] = true_c
                continue
            existed = any(o == true_c for o in owner.values())
            fn += existed
            tn += not existed
            owner[e["concept"]] = true_c
        elif e["kind"] == "match" and e.get("ranking") is not None and e.get("decision") != "INSUFFICIENT_EVIDENCE":
            true_c = scenario.truth[min(e["t"], n - 1)]
            owners = {cid for cid, o in owner.items() if o == true_c}
            if owners:                                          # the true concept is stored: can it be found?
                ids = [int(r["concept_id"].rsplit("_", 1)[1]) for r in e["ranking"]]
                rk = next((i for i, c in enumerate(ids) if c in owners), None)
                cand_total += 1
                for k in (1, 3, 5):
                    cand_hit[k] += rk is not None and rk < k
                if e["decision"] in ("HIGH_CONFIDENCE_MATCH", "CANDIDATE_MATCH") and ids:
                    conf_rows.append((float(e["confidence"]), int(ids[0] in owners)))
                    if e["decision"] == "HIGH_CONFIDENCE_MATCH":
                        high_n += 1
                        high_ok += ids[0] in owners
        elif e["kind"] == "recall":
            true_c = scenario.truth[min(e["t"], n - 1)]
            recalls += 1
            ok = owner.get(e["concept"]) == true_c
            tp += ok
            fp += not ok
            lo, hi = e["t"], min(n, e["t"] + SUCCESS_WINDOW)
            recalls_ok += correct[lo:hi].mean() >= ceiling - SUCCESS_MARGIN
    resolutions = tp + fp + fn + tn
    res.update({
        "recalls": recalls, "retrieval_precision": tp / (tp + fp) if tp + fp else None,
        "retrieval_recall": tp / (tp + fn) if tp + fn else None,
        "false_match_rate": fp / resolutions if resolutions else None,
        "missed_match_rate": fn / (fn + tp) if fn + tp else None,
        "recall_success_rate": recalls_ok / recalls if recalls else None})
    res.update({f"candidate_recall@{k}": (cand_hit[k] / cand_total if cand_total else None) for k in (1, 3, 5)})
    res["high_conf_precision"] = high_ok / high_n if high_n else None
    if conf_rows:
        c, o = np.array(conf_rows).T
        res["conf_brier"] = float(np.mean((c - o) ** 2))
        idx = np.clip((c * 10).astype(int), 0, 9)
        res["conf_ece"] = float(sum(abs(o[idx == b].mean() - c[idx == b].mean()) * np.mean(idx == b) for b in range(10) if (idx == b).any()))
    else:
        res["conf_brier"] = res["conf_ece"] = None
    if getattr(eng, "match_calls", 0):
        res["match_ms"] = 1000 * eng.match_seconds / eng.match_calls
    if hasattr(eng, "fp_calls") and eng.fp_calls:
        res["fp_ms"] = 1000 * eng.fp_seconds / eng.fp_calls
    if strategy == "dejastream_mv":
        res["bytes_per_concept"] = eng.repo.memory_per_concept()
    elif strategy == "dejastream":
        res["bytes_per_concept"] = float(8 * (4 * n_feat + 1))      # one 17-d float64 vector (model not counted)
    return res


def _agg(rows: list[dict]) -> dict:
    out = {}
    for k in rows[0]:
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)} if vals else None
    return out


def fingerprint_latency(reps: int = 200) -> dict:
    """Wall-clock cost of one fingerprint extraction, legacy (17-d, uses labels) vs multi-view (feature only)."""
    from .fingerprint import fingerprint as legacy
    rng = np.random.default_rng(0)
    out = {}
    for n, d in ((100, 4), (300, 4), (100, 10)):
        X, y = rng.normal(0, 1, (n, d)), rng.integers(0, 2, n)
        t = np.arange(n)
        t0 = time.perf_counter()
        for _ in range(reps):
            legacy(X, y)
        a = (time.perf_counter() - t0) / reps
        t0 = time.perf_counter()
        for _ in range(reps):
            extract_feature_fp(X, t=t)
        b = (time.perf_counter() - t0) / reps
        out[f"n{n}_d{d}"] = {"legacy_ms": 1000 * a, "multiview_ms": 1000 * b}
    return out


def run(seeds=(1, 2, 3, 4, 5), scenarios=None, strategies=STRATEGIES, out="results/bench_mv.json", verbose=True) -> dict:
    scenarios = scenarios or list(SCENARIOS)
    results: dict = {"seeds": list(seeds), "scenarios": {}, "strategies": list(strategies)}
    t_all = time.time()
    for sc in scenarios:
        per = {s: [] for s in strategies}
        for seed in seeds:
            data = SCENARIOS[sc](seed=seed)
            for s in strategies:
                per[s].append(run_one(data, s))
        for vname, kw in VARIANTS.get(sc, {}).items():
            per[vname] = [run_one(SCENARIOS[sc](seed=seed), "dejastream_mv", **kw) for seed in seeds]
        results["scenarios"][sc] = {s: _agg(rows) for s, rows in per.items()}
        results["scenarios"][sc]["_notes"] = data.notes
        if verbose:
            print(f"  finished {sc}  ({time.time() - t_all:.0f}s)", flush=True)
    results["fingerprint_latency"] = fingerprint_latency()
    results["wall_seconds"] = round(time.time() - t_all, 1)
    if out:
        with open(out, "w") as f:
            json.dump(results, f, indent=1)
        with open(out.replace(".json", ".md"), "w") as f:
            f.write(to_markdown(results))
    return results


def _fmt(cell, pct=False, nd=3):
    if cell is None:
        return "n/a"
    m, s = cell["mean"], cell["std"]
    return f"{100 * m:.1f}±{100 * s:.1f}%" if pct else f"{m:.{nd}f}±{s:.{nd}f}"


def to_markdown(r: dict) -> str:
    L = [f"Seeds: {r['seeds']}. Values are mean±std over seeds. Synthetic data only.\n"]
    for sc, data in r["scenarios"].items():
        L.append(f"### {sc}" + (f"  \n_{data['_notes']}_" if data.get("_notes") else ""))
        L.append("\n| strategy | accuracy | post-drift acc | detect delay | models trained | recalls | precision | retrieval recall | false-match | missed-match | recall success | µs/sample |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for s in [k for k in data if k != "_notes"]:
            d = data[s]
            L.append(f"| {LABELS.get(s, s)} | {_fmt(d['accuracy'], True)} | {_fmt(d['post_drift_acc'], True)} | {_fmt(d['detection_delay'], nd=1)} | "
                     f"{_fmt(d['models_trained'], nd=1)} | {_fmt(d['recalls'], nd=1)} | {_fmt(d['retrieval_precision'], True)} | "
                     f"{_fmt(d['retrieval_recall'], True)} | {_fmt(d['false_match_rate'], True)} | {_fmt(d['missed_match_rate'], True)} | "
                     f"{_fmt(d['recall_success_rate'], True)} | {_fmt(d['us_per_sample'], nd=1)} |")
        L.append("")
    a = r["scenarios"].get("abrupt_recurring")
    if a:
        L.append("### Memory per stored concept (fingerprint only, models excluded)\n")
        L.append("| strategy | bytes per concept |\n|---|---|")
        for s in ("dejastream", "dejastream_mv"):
            if a[s].get("bytes_per_concept"):
                L.append(f"| {LABELS[s]} | {a[s]['bytes_per_concept']['mean']:.0f} |")
        L.append("")
    L.append("### Fingerprint cost\n\n| window | legacy 17-d (ms) | multi-view (ms) |\n|---|---|---|")
    for k, v in r["fingerprint_latency"].items():
        L.append(f"| {k} | {v['legacy_ms']:.3f} | {v['multiview_ms']:.3f} |")
    return "\n".join(L) + "\n"
