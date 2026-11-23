"""Turn the raw experiment JSON into markdown tables (so no number in the report is typed by hand)."""
from __future__ import annotations

import json

import numpy as np

from .exp_ablation import ABLATION_SCENARIOS, CORE_VARIANTS
from .presets import ablation_variants

KEY = [("accuracy", "accuracy %", True), ("post_drift_acc", "post-drift acc %", True), ("models_trained", "models trained", False),
       ("recalls", "recalls", False), ("retrieval_precision", "match precision %", True), ("retrieval_recall", "match recall %", True),
       ("false_match_rate", "false-match %", True), ("missed_match_rate", "missed-match %", True), ("candidate_recall@1", "cand. R@1 %", True),
       ("candidate_recall@3", "cand. R@3 %", True), ("candidate_recall@5", "cand. R@5 %", True), ("high_conf_precision", "high-conf. precision %", True)]
COST = [("us_per_sample", "µs / sample", False), ("fp_ms", "fingerprint ms", False), ("match_ms", "match ms", False), ("bytes_per_concept", "bytes / concept", False)]


def cell(c, pct, nd=1):
    if c is None:
        return "n/a"
    m, s = c["mean"], c["std"]
    return f"{100 * m:.1f}±{100 * s:.1f}" if pct else f"{m:.{nd}f}±{s:.{nd}f}"


def core_table(res) -> str:
    scs = sorted({k.split("||")[0] for k in res if not k.startswith("delayed")})
    L = []
    for sc in scs:
        L.append(f"\n**{sc}**\n\n| variant | " + " | ".join(n for _, n, _ in KEY) + " |\n|---|" + "---|" * len(KEY))
        for v in CORE_VARIANTS:
            d = res.get(f"{sc}||{v}")
            if d:
                L.append(f"| {v} | " + " | ".join(cell(d.get(k), p) for k, _, p in KEY) + " |")
    return "\n".join(L)


def ablation_tables(res) -> str:
    variants = list(ablation_variants())
    L = ["\n#### Macro-average over the eight ablation scenarios (mean of per-scenario means)\n", "| variant | " + " | ".join(n for _, n, _ in KEY[:9]) + " |\n|---|" + "---|" * 9]
    for v in variants:
        vals = []
        for k, _, p in KEY[:9]:
            xs = [res[f"{sc}||{v}"][k]["mean"] for sc in ABLATION_SCENARIOS if res.get(f"{sc}||{v}", {}).get(k)]
            vals.append(f"{(100 if p else 1) * np.mean(xs):.1f}" if xs else "n/a")
        L.append(f"| {v} | " + " | ".join(vals) + " |")
    for sc in ABLATION_SCENARIOS:
        L.append(f"\n**{sc}**\n\n| variant | " + " | ".join(n for _, n, _ in KEY[:9]) + " |\n|---|" + "---|" * 9)
        for v in variants:
            d = res.get(f"{sc}||{v}")
            if d:
                L.append(f"| {v} | " + " | ".join(cell(d.get(k), p) for k, _, p in KEY[:9]) + " |")
    return "\n".join(L)


def cost_table(res) -> str:
    L = ["\n| variant (abrupt_recurring) | " + " | ".join(n for _, n, _ in COST) + " |\n|---|" + "---|" * len(COST)]
    for v in ablation_variants():
        d = res.get(f"abrupt_recurring||{v}")
        if d:
            L.append(f"| {v} | " + " | ".join(cell(d.get(k), False, 1 if k != "match_ms" and k != "fp_ms" else 2) for k, _, _ in COST) + " |")
    return "\n".join(L)


def delayed_table(res) -> str:
    L = ["\n| labels delayed by | variant | accuracy % | post-drift acc % | models trained | recalls | match precision % | match recall % |\n|---|---|---|---|---|---|---|---|"]
    for dl in ("50", "200"):
        for v in ("MV schema 2.0 (baseline)", "D full multi-view (improved)"):
            d = res.get(f"delayed_labels_{dl}||{v}")
            if d:
                L.append(f"| {dl} | {v} | {cell(d['accuracy'], True)} | {cell(d['post_drift_acc'], True)} | {cell(d['models_trained'], False)} | {cell(d['recalls'], False)} | "
                         f"{cell(d['retrieval_precision'], True)} | {cell(d['retrieval_recall'], True)} |")
    return "\n".join(L)


def paired_table(paired, a="D full multi-view (improved)", b="MV schema 2.0 (baseline)", metrics=("accuracy", "post_drift_acc", "models_trained", "retrieval_precision", "retrieval_recall", "false_match_rate", "candidate_recall@1")) -> str:
    L = [f"\n**{a}  versus  {b}** (per-scenario mean difference, exact paired sign-flip test over 10 seeds, Holm-adjusted p; * = adjusted p < 0.05)\n"]
    for m in metrics:
        rows = paired.get(f"{a} vs {b} | {m}", [])
        L.append(f"\n*{m}*\n\n| scenario | {b.split(' ')[0]} | {a.split(' ')[0]} | difference | adjusted p |\n|---|---|---|---|---|")
        for r in rows:
            f = (lambda v: f"{v:.3f}")
            L.append(f"| {r['scenario']} | {f(r['mean_b'])} | {f(r['mean_a'])} | {r['diff']:+.3f}{'*' if r['p_holm'] < 0.05 else ''} | {r['p_holm']:.3f} |")
    return "\n".join(L)


def scorecard(paired, a="D full multi-view (improved)", b="MV schema 2.0 (baseline)") -> str:
    """Count scenarios where the improved system is significantly better / worse (adjusted p < 0.05) per metric; 'better' depends on the metric."""
    lower_better = {"models_trained", "false_match_rate", "missed_match_rate", "us_per_sample"}
    L = ["| metric | significantly better | significantly worse | no significant difference |\n|---|---|---|---|"]
    for key, rows in paired.items():
        if not key.startswith(f"{a} vs {b} |"):
            continue
        m = key.split("| ")[1]
        better = sum((r["diff"] < 0 if m in lower_better else r["diff"] > 0) and r["p_holm"] < 0.05 for r in rows)
        worse = sum((r["diff"] > 0 if m in lower_better else r["diff"] < 0) and r["p_holm"] < 0.05 for r in rows)
        L.append(f"| {m} | {better} | {worse} | {len(rows) - better - worse} |")
    return "\n".join(L)


if __name__ == "__main__":
    res = json.load(open("results/research/e11_ablation.json"))
    open("results/research/e11_ablation.md", "w").write("### Core variants on every scenario\n" + core_table(res) + "\n\n### Ablation\n" + ablation_tables(res) +
                                                      "\n\n### Cost (abrupt_recurring)\n" + cost_table(res) + "\n\n### Delayed labels (abrupt_recurring)\n" + delayed_table(res))


def real_tables(r) -> str:
    L = []
    for name, d in r["natural"].items():
        base = list(d["runs"].values())[0]["majority_baseline"]
        L.append(f"\n**{name}**: {d['n']} rows, positive rate {d['positive_rate']:.3f} (majority-class accuracy {base:.3f}). {d['note']}\n")
        L.append("| variant | accuracy | 1st half | 2nd half | worst 1000-window | models trained | drifts | recalls | false alarms | µs/sample |\n|---|---|---|---|---|---|---|---|---|---|")
        for v, x in d["runs"].items():
            L.append(f"| {v} | {x['accuracy']:.3f} | {x['acc_first_half']:.3f} | {x['acc_second_half']:.3f} | {x['worst_1000_window']:.3f} | {x['models_trained']} | {x['drifts']} | {x['recalls']} | {x['false_alarms']} | {x['us_per_sample']:.0f} |")
    L.append("\n**Constructed recurrence** (chronological blocks of a real stream replayed as A B A B A; ground truth = block; the repetition is an exact replay, not a natural recurrence)\n")
    for sc, d in r["constructed"].items():
        L.append(f"\n*{sc}*\n\n| variant | accuracy | first 300 after drift | models trained | recalls | match precision | match recall | false-match | candidate R@1 |\n|---|---|---|---|---|---|---|---|---|")
        for v, x in d.items():
            f = lambda k: "n/a" if x.get(k) is None else f"{x[k]:.3f}"
            L.append(f"| {v} | {x['accuracy']:.3f} | {x['post_drift_acc']:.3f} | {x['models_trained']} | {x['recalls']} | {f('retrieval_precision')} | {f('retrieval_recall')} | {f('false_match_rate')} | {f('candidate_recall@1')} |")
    return "\n".join(L)
