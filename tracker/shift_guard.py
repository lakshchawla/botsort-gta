"""ShiftGuard: notices that the embeddings of ONE BoT-SORT track moved to another body.

BoT-SORT's own smooth_feat follows the new embeddings within ~10 frames (EMA 0.9), so the evidence of an identity
switch is gone almost as soon as it happens. The guard keeps a SLOW reference instead (count-based moving average,
learning rate max(1/n, eta_min)) and freezes it while the incoming embeddings look different:

    distance to the reference >= suspect                 -> 'suspect'  (reference frozen, samples kept)
    ... comes back below suspect * settle_ratio          -> 'settled'  (a pose / silhouette change: carry on)
    ... still high after `confirm` consecutive samples   -> 'split'    (another body: the caller mints a new track id)

Pure numpy, no DeepStream.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


def _unit(v):
    return v / np.linalg.norm(v)


class ShiftGuard:
    def __init__(self, suspect: float = 0.10, confirm: int = 3, settle_ratio: float = 0.8, eta_min: float = 0.02):
        self.suspect, self.confirm, self.settle_ratio, self.eta_min = suspect, confirm, settle_ratio, eta_min
        self.ref: Optional[np.ndarray] = None
        self.n = 0
        self.run = 0
        self.tmp: List[np.ndarray] = []
        self.peak = 0.0
        self.shifted: List[np.ndarray] = []          # the samples that triggered the last 'split'

    def reseed(self, samples) -> None:
        """Start over from these samples (a new track after a split)."""
        self.ref, self.n, self.run, self.tmp, self.peak = None, 0, 0, [], 0.0
        for x in samples:
            self._learn(x)

    def _learn(self, x) -> None:
        x = np.asarray(x, dtype=np.float32)
        if self.ref is None:
            self.ref, self.n = x.copy(), 1
            return
        self.n += 1
        eta = max(1.0 / self.n, self.eta_min)
        self.ref = _unit((1 - eta) * self.ref + eta * x)

    def step(self, x) -> str:
        x = np.asarray(x, dtype=np.float32)
        if self.ref is None:
            self._learn(x)
            return "ok"
        d = 1.0 - float(self.ref @ x)
        if self.run == 0:
            if d < self.suspect:
                self._learn(x)
                return "ok"
            self.run, self.tmp, self.peak = 1, [x], d
            return "suspect"
        if d < self.suspect * self.settle_ratio:
            self.run, self.tmp, self.peak = 0, [], 0.0
            self._learn(x)
            return "settled"
        self.run += 1
        self.tmp.append(x)
        self.peak = max(self.peak, d)
        if self.run >= self.confirm:
            self.shifted = list(self.tmp)
            self.run, self.tmp = 0, []
            return "split"
        return "suspect"
