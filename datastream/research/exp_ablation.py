"""M8: end-to-end ablation on controlled streams (5 seeds), written incrementally so a long run can be resumed."""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from ..bench_mv import _agg, run_one
from ..scenarios import SCENARIOS
from .presets import ablation_variants

ALL = list(SCENARIOS)
CORE_VARIANTS = ["v1 (legacy engine)", "MV schema 2.0 (baseline)", "D full multi-view (improved)"]
ABLATION_SCENARIOS = ["abrupt_recurring", "noisy_recurring", "gradual", "similar_x_different_y", "overlapping", "imbalanced",
                      "same_marginal_diff_corr", "feature_availability"]


def run(out="results/research/e11_ablation.json", seeds=(1, 2, 3, 4, 5), log=sys.stdout):
    variants = ablation_variants()
    res = json.load(open(out)) if os.path.exists(out) else {}
    t0 = time.time()
    for sc in ALL:
        names = list(variants) if sc in ABLATION_SCENARIOS else CORE_VARIANTS
        for vname in names:
            key = f"{sc}||{vname}"
            if key in res:
                continue
            strat, kw = variants[vname]
            rows = [run_one(SCENARIOS[sc](seed=s), strat, **kw) for s in seeds]
            res[key] = _agg(rows)
            json.dump(res, open(out, "w"))
            print(f"[{time.time() - t0:6.0f}s] {sc:26} {vname}", file=log, flush=True)
    # delayed labels on a recurring stream
    for delay in (50, 200):
        for vname in ("MV schema 2.0 (baseline)", "D full multi-view (improved)"):
            key = f"delayed_labels_{delay}||{vname}"
            if key in res:
                continue
            strat, kw = variants[vname]
            res[key] = _agg([run_one(SCENARIOS["abrupt_recurring"](seed=s), strat, label_delay=delay, **kw) for s in seeds])
            json.dump(res, open(out, "w"))
            print(f"[{time.time() - t0:6.0f}s] delayed_labels_{delay:3} {vname}", file=log, flush=True)
    return res


if __name__ == "__main__":
    run()
