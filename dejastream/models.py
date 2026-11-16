"""Layer 4: base learner. Small numpy logistic regression with online updates."""
from __future__ import annotations

import numpy as np


class LogReg:
    def __init__(self, d: int, lr: float = 0.05):
        self.w, self.b, self.lr = np.zeros(d), 0.0, lr
        self.mu, self.sd = np.zeros(d), np.ones(d)
        self.seen = 0

    def _p(self, Z):
        return 1.0 / (1.0 + np.exp(-np.clip(Z @ self.w + self.b, -30, 30)))

    def fit(self, X, y, epochs: int = 300, lr: float = 0.3):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        self.w[:], self.b = 0.0, 0.0
        for _ in range(epochs):
            g = self._p(Z) - y
            self.w -= lr * (Z.T @ g / len(y) + 1e-3 * self.w)
            self.b -= lr * g.mean()
        self.seen += len(y)
        return self

    def predict(self, x) -> int:
        return int(self._p(((x - self.mu) / self.sd)[None, :])[0] > 0.5)

    def partial_fit(self, x, y):
        z = (x - self.mu) / self.sd
        g = float(self._p(z[None, :])[0]) - y
        self.w -= self.lr * g * z
        self.b -= self.lr * g
        self.seen += 1

    def accuracy(self, X, y) -> float:
        Z = (np.asarray(X) - self.mu) / self.sd
        return float(((self._p(Z) > 0.5).astype(int) == np.asarray(y)).mean())

    def to_dict(self) -> dict:
        return {"w": self.w.tolist(), "b": self.b, "mu": self.mu.tolist(), "sd": self.sd.tolist()}
