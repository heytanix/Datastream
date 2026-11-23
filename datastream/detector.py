"""Layer 3: drift detection on the stream of prediction errors (Page-Hinkley)."""
from __future__ import annotations


class PageHinkley:
    def __init__(self, delta: float = 0.02, lam: float = 9.0, min_n: int = 30):
        self.delta, self.lam, self.min_n = delta, lam, min_n
        self.reset()

    def reset(self):
        self.n, self.mean, self.m, self.m_min = 0, 0.0, 0.0, 0.0

    def update(self, err: int) -> bool:
        self.n += 1
        self.mean += (err - self.mean) / self.n
        self.m += err - self.mean - self.delta
        self.m_min = min(self.m_min, self.m)
        return self.n >= self.min_n and (self.m - self.m_min) > self.lam
