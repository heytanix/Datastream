"""Layer 4: concept fingerprint.

A fingerprint is a fixed-length vector that describes how a window of labelled data
behaves: per-feature mean and spread, class-conditional mean gaps, label rate and
feature/label correlation. Two windows from the same concept land close together;
windows from different concepts land far apart.
"""
from __future__ import annotations

import numpy as np

from .stream import D


def fingerprint_names() -> list[str]:
    n = [f"mean_x{i}" for i in range(D)] + [f"std_x{i}" for i in range(D)]
    n += [f"gap_x{i}" for i in range(D)] + [f"corr_x{i}y" for i in range(D)] + ["label_rate"]
    return n


def fingerprint(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    mean, std = X.mean(0), X.std(0) + 1e-9
    pos, neg = y == 1, y == 0
    gap = (X[pos].mean(0) - X[neg].mean(0)) if pos.any() and neg.any() else np.zeros(X.shape[1])
    yc = y - y.mean()
    corr = ((X - mean) * yc[:, None]).mean(0) / (std * (y.std() + 1e-9))
    return np.concatenate([mean, std, gap, corr, [y.mean()]])


class Scaler:
    """Running per-dimension std over every fingerprint seen, so no one dimension dominates."""

    PRIOR_N = 5          # pseudo-observations of std 1.0, so the first few fingerprints are not over-trusted

    def __init__(self, dim: int):
        self.n, self.mean, self.m2 = 0, np.zeros(dim), np.zeros(dim)

    def update(self, f: np.ndarray):
        self.n += 1
        d = f - self.mean
        self.mean += d / self.n
        self.m2 += d * (f - self.mean)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt((self.PRIOR_N + self.m2) / (self.PRIOR_N + max(self.n - 1, 0))) + 0.05

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Root-mean-square z-distance between two fingerprints."""
        return float(np.sqrt(np.mean(((a - b) / self.std) ** 2)))
