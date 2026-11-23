"""M6: multiple-candidate selection protocols (simulation) and paired validation inside the engine (streams)."""
from __future__ import annotations

import json
import time

import numpy as np

from ..bench_mv import _agg, run_one
from ..mv_engine import RecallPolicy
from ..scenarios import SCENARIOS
from ..validation import Pred, ValidationConfig, select_candidate


def _preds(y, acc, rng):
    return np.where(rng.random(len(y)) < acc, y, 1 - y)


def exp_protocols(trials=250, seed=5, ns=(30, 60, 100), m=3):
    rng = np.random.default_rng(seed)
    scen = {"null (no candidate better)": (0.70, [0.70, 0.70, 0.70]), "small gain (+0.05 for one candidate)": (0.70, [0.75, 0.70, 0.70]),
            "large gain (+0.20 for one candidate)": (0.70, [0.90, 0.70, 0.70])}
    protos = {"legacy point rule (acc>=0.70, >= baseline, pick best)": None,
              "pick_best + paired CI (biased)": "pick_best", "bonferroni + paired CI": "bonferroni", "split_confirm + paired CI": "split_confirm"}
    out = {}
    for sname, (pb, pcs) in scen.items():
        for n in ns:
            row = {p: {"accept": 0, "accept_correct_candidate": 0, "optimism": []} for p in protos}
            for _ in range(trials):
                y = rng.integers(0, 2, n)
                base = Pred(_preds(y, pb, rng))
                cands = {f"c{i}": Pred(_preds(y, p, rng)) for i, p in enumerate(pcs)}
                accs = {k: float(np.mean(v.pred == y)) for k, v in cands.items()}
                babs = float(np.mean(base.pred == y))
                for pname, prot in protos.items():
                    if prot is None:
                        best = max(accs, key=accs.get)
                        chosen = best if (accs[best] >= 0.70 and accs[best] >= babs) else None
                        selm = accs[best]
                    else:
                        cfg = ValidationConfig(objective="superiority", min_delta=0.0, selection=prot, n_boot=120, min_samples=min(30, n // 2), floor="none", seed=int(rng.integers(1 << 30)))
                        sel = select_candidate(y, cands, base, cfg)
                        chosen, selm = sel.chosen, sel.selected_metric
                    r = row[pname]
                    r["accept"] += chosen is not None
                    r["accept_correct_candidate"] += chosen == "c0" and pcs[0] > pb
                    if selm is not None and chosen is not None:
                        r["optimism"].append(selm - pcs[int(chosen[1:])])
            out[f"{sname} | n={n}"] = {p: {"accept_rate": r["accept"] / trials, "accepts_the_truly_better_one": r["accept_correct_candidate"] / trials,
                                          "optimism_of_reported_metric": float(np.mean(r["optimism"])) if r["optimism"] else None} for p, r in row.items()}
    return out


VARIANTS = {
    "legacy point (acc>=0.70, no uncertainty)": RecallPolicy(),
    "paired non-inferiority + bonferroni + significant-majority floor": RecallPolicy(mode="paired", validation=ValidationConfig(floor="majority")),
    "paired non-inferiority + bonferroni + point floor (default)": RecallPolicy(mode="paired"),
    "paired non-inferiority + bonferroni, no floor": RecallPolicy(mode="paired", validation=ValidationConfig(floor="none")),
    "paired superiority(0.02) + bonferroni + point floor": RecallPolicy(mode="paired", validation=ValidationConfig(objective="superiority", min_delta=0.02)),
    "paired non-inferiority + pick_best (biased) + point floor": RecallPolicy(mode="paired", validation=ValidationConfig(selection="pick_best")),
    "paired non-inferiority + split_confirm + point floor": RecallPolicy(mode="paired", validation=ValidationConfig(selection="split_confirm", confirm_min=30)),
}


def exp_streams(seeds=(1, 2, 3, 4, 5), scenarios=("abrupt_recurring", "noisy_recurring", "similar_x_different_y", "sea", "gradual")):
    res = {}
    for sc in scenarios:
        per = {v: [] for v in VARIANTS}
        for seed in seeds:
            data = SCENARIOS[sc](seed=seed)
            for v, pol in VARIANTS.items():
                per[v].append(run_one(data, "dejastream_mv", recall=pol))
        res[sc] = {v: _agg(rows) for v, rows in per.items()}
    return res


def to_md(prot, streams) -> str:
    L = ["### Selection protocols (simulation, 250 trials per cell, 3 candidates)\n", "| scenario | protocol | accept rate | accepts the truly better candidate | optimism of the reported metric |\n|---|---|---|---|---|"]
    for k, rows in prot.items():
        for p, v in rows.items():
            opt = "n/a" if v["optimism_of_reported_metric"] is None else f"{v['optimism_of_reported_metric']:+.3f}"
            L.append(f"| {k} | {p} | {v['accept_rate']:.3f} | {v['accepts_the_truly_better_one']:.3f} | {opt} |")
    L.append("\n### Paired validation inside the engine (5 seeds, mean±std)\n")
    f = lambda c, pct=False: "n/a" if c is None else (f"{100 * c['mean']:.1f}±{100 * c['std']:.1f}%" if pct else f"{c['mean']:.2f}±{c['std']:.2f}")
    for sc, vv in streams.items():
        L.append(f"\n**{sc}**\n\n| variant | accuracy | post-drift acc | models trained | recalls | precision | retrieval recall | false-match | recall success |\n|---|---|---|---|---|---|---|---|---|")
        for v, d in vv.items():
            L.append(f"| {v} | {f(d['accuracy'], True)} | {f(d['post_drift_acc'], True)} | {f(d['models_trained'])} | {f(d['recalls'])} | {f(d['retrieval_precision'], True)} | "
                     f"{f(d['retrieval_recall'], True)} | {f(d['false_match_rate'], True)} | {f(d['recall_success_rate'], True)} |")
    return "\n".join(L)


if __name__ == "__main__":
    t0 = time.time()
    prot, streams = exp_protocols(), exp_streams()
    json.dump({"protocols": prot, "streams": streams}, open("results/research/e9_validation.json", "w"), indent=1)
    md = to_md(prot, streams)
    open("results/research/e9_validation.md", "w").write(md)
    print(md)
    print("seconds", round(time.time() - t0))
