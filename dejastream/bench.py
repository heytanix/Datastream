"""Offline benchmark: run all strategies over the same stream, no bus, print and save results."""
from __future__ import annotations

import csv
import json
import time

import numpy as np

from .engine import STRATEGIES, Engine
from .stream import ConceptStream


def run(seed: int = 7, seg_len: int = 1500, out: str | None = "results/bench.json", verbose: bool = True):
    stream = ConceptStream(seed=seed, seg_len=seg_len)
    cps = stream.change_points()
    engines = {s: Engine(s) for s in STRATEGIES}
    correct = {s: [] for s in STRATEGIES}
    events = {s: [] for s in STRATEGIES}
    t0 = time.time()
    n = 0
    for t, x, y, c in stream:
        n += 1
        for s, e in engines.items():
            p, ev = e.process(t, x, y)
            correct[s].append(int(p == y))
            events[s].extend(ev)
    secs = time.time() - t0

    res = {"samples": n, "seconds": round(secs, 2), "change_points": cps, "strategies": {}}
    for s in STRATEGIES:
        c = np.array(correct[s])
        seg_acc = [round(float(c[a:b].mean()), 4) for (a, _), (b, _) in zip(cps, cps[1:] + [(n, "")])]
        # recovery cost: mean accuracy over the first 300 samples after each change (skip the first segment)
        post = [float(c[a:a + 300].mean()) for a, _ in cps[1:]]
        # detection delay: first drift event after each change point
        drifts = [e["t"] for e in events[s] if e["kind"] == "drift"]
        delays = []
        for a, _ in cps[1:]:
            nxt = [d for d in drifts if d >= a]
            delays.append(nxt[0] - a if nxt and (nxt[0] - a) < 1000 else None)
        seen, recalls = set(), 0
        for (a, name), _ in zip(cps, cps):
            seen.add(name)
        res["strategies"][s] = {
            "accuracy": round(float(c.mean()), 4),
            "post_drift_300_acc": round(float(np.mean(post)), 4) if post else None,
            "segment_acc": seg_acc,
            "drifts_detected": len(drifts),
            "detected_changes": sum(d is not None for d in delays),
            "true_changes": len(cps) - 1,
            "mean_delay": round(float(np.mean([d for d in delays if d is not None])), 1) if any(d is not None for d in delays) else None,
            "models_trained": engines[s].models_trained,
            "recalls": sum(e["kind"] == "recall" for e in events[s]),
            "false_alarms": sum(e["kind"] == "false_alarm" for e in events[s]),
            "new_concepts": sum(e["kind"] == "new_concept" for e in events[s]),
            "repository_size": len(engines[s].repo),
        }
    if out:
        with open(out, "w") as f:
            json.dump(res, f, indent=1)
        with open(out.replace(".json", "_rolling.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t"] + list(STRATEGIES))
            k = 100
            cs = {s: np.convolve(correct[s], np.ones(k) / k, mode="valid") for s in STRATEGIES}
            for i in range(0, len(cs["static"]), 50):
                w.writerow([i + k] + [round(float(cs[s][i]), 4) for s in STRATEGIES])
    if verbose:
        print(f"{n} samples in {secs:.1f}s  ({len(cps)-1} true concept changes)")
        hdr = f"{'strategy':<12}{'accuracy':>9}{'post-drift':>11}{'detected':>10}{'delay':>7}{'trained':>8}{'recalls':>8}{'false':>6}"
        print(hdr)
        for s, r in res["strategies"].items():
            print(f"{s:<12}{r['accuracy']:>9.3f}{r['post_drift_300_acc']:>11.3f}"
                  f"{r['detected_changes']:>5}/{r['true_changes']:<4}{str(r['mean_delay']):>7}{r['models_trained']:>8}{r['recalls']:>8}{r['false_alarms']:>6}")
    return res, events, engines
