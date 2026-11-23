"""Strategy registry: legacy engines plus the multi-view engine."""
from __future__ import annotations

from .engine import Engine
from .mv_engine import MVEngine

LEGACY = ("static", "retrain", "datastream")
STRATEGIES = LEGACY + ("datastream_mv",)
LABELS = {"static": "Static model", "retrain": "Retrain on drift", "datastream": "Datastream (v1)",
          "datastream_mv": "Datastream MV"}


def make_engine(strategy: str, **kw):
    if strategy == "datastream_mv":
        return MVEngine(**kw)
    return Engine(strategy, **kw)
