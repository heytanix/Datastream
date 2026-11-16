"""Turn results/bench.json into LaTeX fragments so the PDF always shows the measured numbers."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
r = json.loads((root.parent / "results" / "bench.json").read_text())
names = {"static": "Static model", "retrain": "Retrain on drift", "dejastream": "DejaStream"}
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
