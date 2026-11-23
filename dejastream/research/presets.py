"""Named configurations used by the experiments ("legacy" = schema 2.0 defaults)."""
from __future__ import annotations

from ..matcher import MatchConfig
from ..similarity import SimilarityConfig


def legacy_similarity(**kw) -> SimilarityConfig:
    return SimilarityConfig(**kw)


def improved_similarity(**kw) -> SimilarityConfig:
    """Options that M3 supported by experiment: symmetric PSI, tail term, standardised temporal, Fisher-z correlations."""
    base = dict(psi_binning="pooled", tail_weight=1.0, time_norm="standardised", corr_transform="fisher")
    base.update(kw)
    return SimilarityConfig(**base)


# ---- engine variants used by the ablation study ---------------------------------------------------
from ..matcher import MatchConfig as _MC
from ..mv_engine import RecallPolicy as _RP


def improved_match(**over) -> MatchConfig:
    """Level 1 adaptive + full-scan fallback, empirical thresholds, mergeable memory."""
    base = dict(top_k_mode="adaptive", expand_on_new=True, threshold_mode="empirical", merge_mode="mergeable")
    base.update(over)
    return MatchConfig(**base)


def ablation_variants() -> dict[str, tuple[str, dict]]:
    """name -> (strategy, engine kwargs). A-D ablate fingerprint views; E-H remove one mechanism from the full system."""
    W = lambda s, d, t, p: {"stat": s, "dist": d, "time": t, "pred": p}
    sim = lambda w, **kw: improved_similarity(weights=w, group_agg="mean", **kw)
    full = W(0.4, 0.2, 0.4, 0.2)
    return {
        "v1 (legacy engine)": ("dejastream", {}),
        "MV schema 2.0 (baseline)": ("dejastream_mv", {}),
        "A statistical only": ("dejastream_mv", dict(sim=sim(W(1, 0, 0, 0)), match=improved_match())),
        "B stat + distribution": ("dejastream_mv", dict(sim=sim(W(0.5, 0.5, 0, 0)), match=improved_match())),
        "C stat + dist + temporal": ("dejastream_mv", dict(sim=sim(W(0.4, 0.2, 0.4, 0)), match=improved_match())),
        "D full multi-view (improved)": ("dejastream_mv", dict(sim=sim(full), match=improved_match())),
        "E = D without noise correction": ("dejastream_mv", dict(sim=sim(full, noise_mode="none"), match=improved_match())),
        "F = D without adaptive thresholds": ("dejastream_mv", dict(sim=sim(full), match=improved_match(threshold_mode="legacy"))),
        "G = D without confidence gating": ("dejastream_mv", dict(sim=sim(full), match=improved_match(conf_high=0.0, min_margin=0.0))),
        "H = D without predictive validation": ("dejastream_mv", dict(sim=sim(full), match=improved_match(), recall=_RP(mode="none"))),
        "J = D + paired validation": ("dejastream_mv", dict(sim=sim(full), match=improved_match(), recall=_RP(mode="paired"))),
        "K = D with legacy fingerprint merge": ("dejastream_mv", dict(sim=sim(full), match=improved_match(merge_mode="legacy"))),
    }
