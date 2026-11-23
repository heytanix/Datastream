"""Turn results/bench.json into LaTeX fragments so the PDF always shows the measured numbers."""
import json
from pathlib import Path
from statistics import mean

root = Path(__file__).resolve().parent
r = json.loads((root.parent / "results" / "bench.json").read_text())
names = {"static": "Static model", "retrain": "Retrain on drift", "dejastream": "DejaStream v1", "dejastream_mv": "DejaStream MV"}
rows = []
for k, v in r["strategies"].items():
    rows.append(f"{names[k]} & {v['accuracy']*100:.1f}\\% & {v['post_drift_300_acc']*100:.1f}\\% & "
                f"{v['detected_changes']}/{v['true_changes']} & {v['mean_delay'] if v['mean_delay'] is not None else '--'} & "
                f"{v['models_trained']} & {v['recalls']} & {v['false_alarms']} \\\\")
head = ("\\renewcommand{\\arraystretch}{1.35}\n\\begin{tabular}{@{}lccccccc@{}}\n\\toprule\n"
        "\\textbf{Strategy} & \\textbf{Overall acc.} & \\textbf{Acc. first 300 after drift} & \\textbf{Changes detected} & "
        "\\textbf{Mean delay} & \\textbf{Models trained} & \\textbf{Recalls} & \\textbf{False alarms}\\\\\n\\midrule\n")
(root / "results_table.tex").write_text(head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
(root / "results_meta.tex").write_text(
    f"\\newcommand{{\\nsamples}}{{{r['samples']:,}}}\n\\newcommand{{\\nchanges}}{{{len(r['change_points'])-1}}}\n".replace(",", "{,}"))


# ---- multi-view benchmark (results/bench_mv.json) -> docs/results_mv_table.tex, docs/results_mv_cost.tex ----
mv_path = root.parent / "results" / "bench_mv.json"
if mv_path.exists():
    m = json.loads(mv_path.read_text())

    def cell(c, pct=False, nd=1):
        if c is None:
            return "--"
        mean, std = c["mean"], c["std"]
        return f"{100*mean:.1f}$\\pm${100*std:.1f}" if pct else f"{mean:.{nd}f}$\\pm${std:.{nd}f}"

    shown = ["abrupt_recurring", "noisy_recurring", "gradual", "incremental", "similar_x_different_y", "sea"]
    pretty = {"abrupt_recurring": "abrupt, recurring", "noisy_recurring": "noisy recurring", "gradual": "gradual drift",
              "incremental": "incremental drift", "similar_x_different_y": "similar $X$, different $Y$", "sea": "SEA"}
    rows = []
    for sc in shown:
        for key, label in (("dejastream", "v1"), ("dejastream_mv", "MV")):
            d = m["scenarios"][sc][key]
            name = pretty[sc] if label == "v1" else ""
            rows.append(f"{name} & {label} & {cell(d['accuracy'], True)} & {cell(d['post_drift_acc'], True)} & "
                        f"{cell(d['models_trained'])} & {cell(d['retrieval_precision'], True)} & "
                        f"{cell(d['retrieval_recall'], True)} & {cell(d['false_match_rate'], True)} \\\\")
        rows.append("\\addlinespace[2pt]")
    head = ("\\renewcommand{\\arraystretch}{1.25}\\begin{tabular}{@{}llccccrr@{}}\\toprule\n"
            "\\textbf{Scenario} & & \\textbf{Accuracy \\%} & \\textbf{First 300 after drift \\%} & \\textbf{Models trained} & "
            "\\textbf{Retrieval precision \\%} & \\textbf{Retrieval recall \\%} & \\textbf{False-match \\%}\\\\\\midrule\n")
    (root / "results_mv_table.tex").write_text(head + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
    a = m["scenarios"]["abrupt_recurring"]
    lat = m["fingerprint_latency"]
    cost = ("\\renewcommand{\\arraystretch}{1.25}\\begin{tabular}{@{}lcc@{}}\\toprule\n& \\textbf{v1 (17-d)} & \\textbf{MV (schema 2.0)}\\\\\\midrule\n"
            f"Fingerprint time, 100 samples, 4 features & {lat['n100_d4']['legacy_ms']:.3f} ms & {lat['n100_d4']['multiview_ms']:.3f} ms\\\\\n"
            f"Fingerprint time, 300 samples, 4 features & {lat['n300_d4']['legacy_ms']:.3f} ms & {lat['n300_d4']['multiview_ms']:.3f} ms\\\\\n"
            f"Fingerprint time, 100 samples, 10 features & {lat['n100_d10']['legacy_ms']:.3f} ms & {lat['n100_d10']['multiview_ms']:.3f} ms\\\\\n"
            f"Whole engine, per sample (abrupt scenario) & {a['dejastream']['us_per_sample']['mean']:.1f} $\\mu$s & {a['dejastream_mv']['us_per_sample']['mean']:.1f} $\\mu$s\\\\\n"
            f"Memory per stored concept (fingerprint only) & {a['dejastream']['bytes_per_concept']['mean']:.0f} B & {a['dejastream_mv']['bytes_per_concept']['mean']:.0f} B\\\\\n"
            "\\bottomrule\\end{tabular}\n")
    (root / "results_mv_cost.tex").write_text(cost)
    (root / "results_mv_meta.tex").write_text(f"\\newcommand{{\\mvseeds}}{{{len(m['seeds'])}}}\n")


# ---- research results: ablation (seeds 1-5) and fresh-seed paired confirmation -> page 11 ---------------------
abl_path, pair_path = root.parent / "results" / "research" / "e11_ablation.json", root.parent / "results" / "research" / "e13_paired.json"
if abl_path.exists():
    abl = json.loads(abl_path.read_text())
    variants = ["v1 (legacy engine)", "MV schema 2.0 (baseline)", "A statistical only", "B stat + distribution", "C stat + dist + temporal",
                "D full multi-view (improved)", "E = D without noise correction", "F = D without adaptive thresholds",
                "G = D without confidence gating", "H = D without predictive validation", "J = D + paired validation", "K = D with legacy fingerprint merge"]
    scen = ["abrupt_recurring", "noisy_recurring", "gradual", "similar_x_different_y", "overlapping", "imbalanced", "same_marginal_diff_corr", "feature_availability"]
    cols = [("accuracy", True), ("post_drift_acc", True), ("models_trained", False), ("retrieval_recall", True), ("retrieval_precision", True), ("false_match_rate", True)]
    rows = []
    for v in variants:
        vals = []
        for k, pct in cols:
            xs = [abl[f"{sc}||{v}"][k]["mean"] for sc in scen if abl.get(f"{sc}||{v}", {}).get(k)]
            vals.append(f"{(100 if pct else 1) * mean(xs):.1f}" if xs else "--")
        rows.append(f"{v} & " + " & ".join(vals) + " \\\\")
    head = ("\\renewcommand{\\arraystretch}{1.2}\\begin{tabular}{@{}lcccccc@{}}\\toprule\n\\textbf{Variant} & \\textbf{Accuracy \\%} & \\textbf{First 300 after drift \\%} & "
            "\\textbf{Models trained} & \\textbf{Match recall \\%} & \\textbf{Match precision \\%} & \\textbf{False-match \\%}\\\\\\midrule\n")
    (root / "results_research_macro.tex").write_text(head + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
if pair_path.exists():
    pr = json.loads(pair_path.read_text())
    lower = {"models_trained", "false_match_rate", "missed_match_rate", "us_per_sample"}
    metrics = [("accuracy", "accuracy"), ("post_drift_acc", "first 300 after drift"), ("models_trained", "models trained"), ("retrieval_recall", "match recall"),
               ("retrieval_precision", "match precision"), ("false_match_rate", "false-match rate"), ("candidate_recall@1", "candidate recall@1"), ("us_per_sample", "time per sample")]
    comps = [("F = D without adaptive thresholds", "MV schema 2.0 (baseline)", "improved (F) vs MV 2.0"),
             ("D full multi-view (improved)", "MV schema 2.0 (baseline)", "improved (D) vs MV 2.0"), ("MV schema 2.0 (baseline)", "v1 (legacy engine)", "MV 2.0 vs v1")]
    rows = []
    for m, label in metrics:
        cells = []
        for a, b, _ in comps:
            r = pr.get(f"{a} vs {b} | {m}", [])
            better = sum((x["diff"] < 0 if m in lower else x["diff"] > 0) and x["p_holm"] < 0.05 for x in r)
            worse = sum((x["diff"] > 0 if m in lower else x["diff"] < 0) and x["p_holm"] < 0.05 for x in r)
            cells.append(f"{better} / {worse} / {len(r) - better - worse}")
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    head = ("\\renewcommand{\\arraystretch}{1.2}\\begin{tabular}{@{}lccc@{}}\\toprule\n\\textbf{Metric (scenarios: better / worse / no significant difference)} & "
            + " & ".join(f"\\textbf{{{c[2]}}}" for c in comps) + "\\\\\\midrule\n")
    (root / "results_research_score.tex").write_text(head + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
