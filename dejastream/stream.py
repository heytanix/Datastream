"""Layer 1: data source. Synthetic stream with recurring concepts.

Each concept has its own feature distribution p(x) and its own labelling rule p(y|x),
so a drift can be seen both in the inputs (fingerprint) and in the errors (detector).
The schedule revisits earlier concepts, which is what makes recall worthwhile.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

D = 4


@dataclass(frozen=True)
class Concept:
    name: str
    mu: tuple
    sigma: tuple
    w: tuple


CONCEPTS = {
    "A": Concept("A", (0, 0, 0, 0), (1.0, 1.0, 1.0, 1.0), (1, 1, 0, 0)),
    "B": Concept("B", (2.5, -1.5, 1, 0), (0.8, 1.2, 1.0, 1.0), (1, -1, 0.5, 0)),
    "C": Concept("C", (-2, 2, -1, 1.5), (1.5, 0.7, 1.0, 0.8), (-1, -0.5, 1, 1)),
    "D": Concept("D", (1, 3, -2, -1), (0.7, 1.0, 1.4, 1.0), (0.2, 1, -1, 1)),
}

# A, B, C, D all come back at least once.
DEFAULT_SCHEDULE = "A B C A B D C A D B A".split()


class ConceptStream:
    """Yields (t, x, y, concept_name). Finite for one pass, loops if ``loops`` > 1 or 0 (forever)."""

    def __init__(self, schedule=None, seg_len: int = 1500, noise: float = 0.04,
                 seed: int = 7, loops: int = 1):
        self.schedule = list(schedule or DEFAULT_SCHEDULE)
        self.seg_len, self.noise, self.loops = seg_len, noise, loops
        self.rng = np.random.default_rng(seed)

    def change_points(self) -> list[tuple[int, str]]:
        """(t, concept) at which each segment begins, for one pass."""
        return [(i * self.seg_len, c) for i, c in enumerate(self.schedule)]

    def __iter__(self):
        t, loop = 0, 0
        while self.loops == 0 or loop < self.loops:
            for name in self.schedule:
                c = CONCEPTS[name]
                mu, sg, w = (np.array(v, float) for v in (c.mu, c.sigma, c.w))
                X = self.rng.normal(mu, sg, size=(self.seg_len, D))
                y = ((X - mu) @ w > 0).astype(int)
                flip = self.rng.random(self.seg_len) < self.noise
                y = np.where(flip, 1 - y, y)
                for i in range(self.seg_len):
                    yield t, X[i], int(y[i]), name
                    t += 1
            loop += 1
