"""Assemble docs/RESEARCH.md from the experiment outputs. Tables come from results/research/*.json and *.md; prose is in
``PROSE`` below and was written after reading those results."""
from __future__ import annotations

import json
import os

from .summarise import cost_table, delayed_table, paired_table, real_tables, scorecard

R = "results/research"


def read(name: str) -> str:
    p = os.path.join(R, name)
    return open(p).read() if os.path.exists(p) else f"_missing: {name}_"


def main(out: str = "docs/RESEARCH.md", prose_path: str = "docs/RESEARCH_PROSE.md") -> None:
    abl = json.load(open(f"{R}/e11_ablation.json"))
    paired = json.load(open(f"{R}/e13_paired.json")) if os.path.exists(f"{R}/e13_paired.json") else None
    prose = open(prose_path).read() if os.path.exists(prose_path) else ""
    parts = [prose.strip(), "\n\n---\n\n# Appendix: all measured tables (generated from `results/research/`)\n"]
    parts.append("## A1. Milestone 1: audit measurements\nSee [`AUDIT.md`](AUDIT.md) and `results/research/audit.json`.\n")
    parts.append("## A2. Noise model and effective sample size (E1)\n\n" + read("e1_noise.md"))
    parts.append("\n\n## A3. Distribution metrics, tail term, temporal statistics, correlations (E2, E3)\n\n" + read("e2_e3_dist_time.md"))
    parts.append("\n\n## A4. Level-1 retrieval, view ablation, weights, aggregation (E4)\n\n" + read("e4_retrieval.md"))
    parts.append("\n\n## A5. Thresholds and confidence calibration (E7, E8)\n\n" + read("e7_e8_thresholds_confidence.md"))
    parts.append("\n\n## A6. Validation protocols (E9)\n\n" + read("e9_validation.md"))
    parts.append("\n\n## A7. Mergeable memory and eviction (E10)\n\n" + read("e10_memory.md"))
    parts.append("\n\n## A8. End-to-end ablation, seeds 1-5 (E11)\n\n" + read("e11_ablation.md"))
    if paired:
        parts.append("\n\n## A9. Fresh-seed confirmation, seeds 11-20, exact paired permutation tests (E13)\n")
        parts.append("\n### Scorecard: scenarios (of 18) where the difference is significant after Holm correction\n")
        for a, b in (("F = D without adaptive thresholds", "MV schema 2.0 (baseline)"), ("D full multi-view (improved)", "MV schema 2.0 (baseline)"),
                     ("MV schema 2.0 (baseline)", "v1 (legacy engine)")):
            parts.append(f"\n**{a}  vs  {b}**\n\n" + scorecard(paired, a, b))
        for a, b in (("F = D without adaptive thresholds", "MV schema 2.0 (baseline)"), ("MV schema 2.0 (baseline)", "v1 (legacy engine)")):
            parts.append("\n" + paired_table(paired, a, b))
    parts.append("\n\n## A10. Scaling with repository size and fingerprint cost (E12)\n\n" + read("e12_scaling.md"))
    if os.path.exists(f"{R}/e14_real.json"):
        parts.append("\n\n## A11. Real datasets (E14): natural streams and constructed recurrence\n" + real_tables(json.load(open(f"{R}/e14_real.json"))))
    open(out, "w").write("\n".join(parts) + "\n")


if __name__ == "__main__":
    main()
