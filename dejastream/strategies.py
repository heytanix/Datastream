"""Strategy registry: legacy engines plus the multi-view engine."""
from __future__ import annotations

from .engine import Engine
from .mv_engine import MVEngine

LEGACY = ("static", "retrain", "dejastream")
STRATEGIES = LEGACY + ("dejastream_mv",)
LABELS = {"static": "Static model", "retrain": "Retrain on drift", "dejastream": "DejaStream (v1)",
          "dejastream_mv": "DejaStream MV"}


def make_engine(strategy: str, **kw):
    if strategy == "dejastream_mv":
        return MVEngine(**kw)
    return Engine(strategy, **kw)
