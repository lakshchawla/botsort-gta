"""The ONLY place embedding math lives.

EmbeddingTrack is a single-stream, unit-norm exponential moving average of the ReID vectors seen so
far by one entity (a tracklet or a global_entry). Anything that wants to change how embeddings are
accumulated or compared (attention memory, windows, multi-model ...) only has to provide the same
members: update / value / last / count / distance / copy.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

EPS = 1e-9
MIN_WEIGHT = 0.0


def unit(vec) -> Optional[np.ndarray]:
    """float32 copy scaled to unit length; None for zero / non-finite input."""
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < EPS:
        return None
    return v / n


class EmbeddingTrack:
    def __init__(self, alpha: float = 0.8, window: int = 0):
        self.alpha = alpha                         # weight of the history in the EMA
        self.window = window                       # >0: also keep the last `window` samples
        self._win: Optional[deque] = deque(maxlen=window) if window > 0 else None
        self.value: Optional[np.ndarray] = None    # moving average, unit norm
        self.last: Optional[np.ndarray] = None     # newest accepted sample, unit norm
        self.count = 0                             # accepted samples

    @property
    def empty(self) -> bool:
        return self.value is None

    def update(self, vec, damping: float = 0.0) -> bool:
        """Fold one sample in. damping in [0,1] (occlusion / low confidence) pushes the effective
        alpha towards 1, i.e. the sample moves the average less. Returns False for unusable input."""
        v = unit(vec)
        if v is None:
            return False
        if self.value is None:
            self.value = v
        else:
            a = self.alpha + (1.0 - self.alpha) * min(max(float(damping), 0.0), 1.0)
            self.value = unit(a * self.value + (1.0 - a) * v)
        self.last = v
        self.count += 1
        if self._win is not None:
            self._win.append(v)
        return True

    @property
    def window_count(self) -> int:
        return 0 if self._win is None else len(self._win)

    def window_mean(self) -> Optional[np.ndarray]:
        """Unit-norm plain mean of the last `window` accepted samples (the 'smooth feature');
        None if the window is off or empty."""
        if not self._win:
            return None
        return unit(np.mean(np.stack(self._win), axis=0))

    def reset(self, samples, damping: float = 0.0) -> None:
        """Forget the history and rebuild from `samples` (oldest first)."""
        self.value = self.last = None
        self.count = 0
        if self._win is not None:
            self._win.clear()
        for v in samples:
            self.update(v, damping)

    def distance(self, other) -> float:
        """Cosine distance to a vector or another EmbeddingTrack; 2.0 if either is empty."""
        b = other.value if isinstance(other, EmbeddingTrack) else unit(other)
        if self.value is None or b is None:
            return 2.0
        return 1.0 - float(np.dot(self.value, b))

    def copy(self) -> "EmbeddingTrack":
        c = EmbeddingTrack(self.alpha, self.window)
        if self._win is not None:
            c._win.extend(v.copy() for v in self._win)
        c.value = None if self.value is None else self.value.copy()
        c.last = None if self.last is None else self.last.copy()
        c.count = self.count
        return c
