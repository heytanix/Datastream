"""Real-data runs. NATURAL streams have no ground-truth concepts: only accuracy, model counts, drift/recall counts and cost are reported.
CONSTRUCTED recurrence replays chronological blocks of a real stream in a chosen order; the block identity is then the ground truth, and the
result describes recognition of replayed data, not of naturally recurring concepts."""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from ..bench_mv import run_one
from ..datasets import load_arff, load_intel_berkeley, load_uci_electricity, replay_stream
from ..scenarios import ScenarioData
from .presets import ablation_variants

DATA = "data"
VARIANTS = ["v1 (legacy engine)", "MV schema 2.0 (baseline)", "D full multi-view (improved)", "F = D without adaptive thresholds"]


def natural_streams():
    out = {}
    X, y, names, nominal = load_arff(f"{DATA}/elec2/electricity.arff", "class", drop=("date",))
    out["ELEC2 (45,312 rows, natural order)"] = (X, y, f"features {names}; date dropped (a monotone counter)")
    X, y, names, nominal = load_arff(f"{DATA}/airlines/airlines.arff", "Delay", max_rows=100000)
    keep = [names.index(n) for n in ("Airline", "Flight", "DayOfWeek", "Time", "Length")]
    out["Airlines (first 100,000 rows)"] = (X[:, keep], y, "features Airline(code), Flight, DayOfWeek, Time, Length; airports omitted; file order is not strictly chronological")
    X, y, t, _, names = load_intel_berkeley(f"{DATA}/intel/data.txt", mote_id=1, max_rows=60000)
    ok = y >= 0
    out["Intel Berkeley mote 1 (first 60,000 readings)"] = (X[ok], y[ok], f"features {names}; DERIVED label: temperature rises over the next 30 readings; raw sensor glitches kept")
    X, y, t, _, names = load_uci_electricity(f"{DATA}/uci/LD2011_2014.txt", clients=(0, 1, 2, 3, 4, 5), max_rows=70000)
    ok = y >= 0
    out["UCI load diagrams, 6 clients (first 70,000 intervals)"] = (X[ok], y[ok], f"features {names}; DERIVED label: total demand rises next interval")
    return out


def run_natural(X, y, strategy, kw):
    from ..strategies import make_engine
    eng = make_engine(strategy, n_features=X.shape[1], **kw)
    n = len(y)
    correct = np.zeros(n, int)
    ev = []
    t0 = time.perf_counter()
    for t in range(n):
        p, e = eng.process(t, X[t], int(y[t]))
        correct[t] = int(p == y[t])
        ev += e
    secs = time.perf_counter() - t0
    k = lambda name: sum(e["kind"] == name for e in ev)
    half = n // 2
    return {"accuracy": float(correct.mean()), "acc_first_half": float(correct[:half].mean()), "acc_second_half": float(correct[half:].mean()),
            "worst_1000_window": float(np.convolve(correct, np.ones(1000) / 1000, mode="valid").min()), "models_trained": int(eng.models_trained),
            "drifts": k("drift"), "recalls": k("recall"), "false_alarms": k("false_alarm"), "new_concepts": k("new_concept"), "us_per_sample": 1e6 * secs / n,
            "majority_baseline": float(max(y.mean(), 1 - y.mean()))}


def constructed(X, y, blocks, order, name):
    Xr, yr, truth = replay_stream(X, y, blocks, order)
    cps, last = [], None
    for i, b in enumerate(truth):
        if b != last:
            cps.append((i, b))
            last = b
    return ScenarioData(name, Xr, yr, truth, cps, True, "CONSTRUCTED recurrence by replaying chronological blocks")


def main(out="results/research/e14_real.json"):
    variants = ablation_variants()
    res = {"natural": {}, "constructed": {}}
    streams = natural_streams()
    for name, (X, y, note) in streams.items():
        res["natural"][name] = {"note": note, "n": int(len(y)), "positive_rate": float(y.mean()), "runs": {}}
        for v in ["static", "retrain"] + VARIANTS:
            strat, kw = ({"static": ("static", {}), "retrain": ("retrain", {})}.get(v) or variants[v])
            res["natural"][name]["runs"][v] = run_natural(X, y, strat, kw)
            print(f"[natural] {name[:30]:30} {v}", file=sys.stderr, flush=True)
    # constructed recurrence: A B A B A with blocks of real, chronologically adjacent data
    X, y, _ = streams["ELEC2 (45,312 rows, natural order)"]
    for label, (X, y, _), blocks in (("ELEC2", streams["ELEC2 (45,312 rows, natural order)"], [(0, 6000), (25000, 31000)]),
                                     ("UCI", streams["UCI load diagrams, 6 clients (first 70,000 intervals)"], [(0, 6000), (40000, 46000)]),
                                     ("Intel", streams["Intel Berkeley mote 1 (first 60,000 readings)"], [(0, 6000), (30000, 36000)])):
        sc = constructed(X, y, blocks, [0, 1, 0, 1, 0], f"{label} replay A B A B A")
        res["constructed"][sc.name] = {}
        for v in VARIANTS:
            strat, kw = variants[v]
            res["constructed"][sc.name][v] = run_one(sc, strat, **kw)
            print(f"[constructed] {sc.name} {v}", file=sys.stderr, flush=True)
    json.dump(res, open(out, "w"), indent=1)
    return res


if __name__ == "__main__":
    main()
