"""
Shared spatio-temporal prior for cross-camera identity matching.

Extracted from global_registry.py so both it and global_tracklet_association.py
(the tracklet-level engine) use the exact same implementation instead of two
copies drifting apart.
"""

from __future__ import annotations

import math
from typing import Optional


class SpatioTemporalPrior:
    """P_st(dt, camera transition) for a cross-camera matching cost matrix.

    windows: {(cam_a, cam_b): (min_transit_sec, max_transit_sec)} - the
    physically plausible elapsed-time window for a person moving between
    that camera pair (non-overlapping FOVs). Pairs are canonicalized, so
    one entry covers both directions. Same-camera transitions and pairs
    with no configured window are uninformative (P=1, log_p=0) - fill in
    topology data incrementally, most-trafficked pairs first.

    Outside the window P drops to p_outside (small but NONZERO): -log P
    becomes a large finite cost that competes with the solver's dummy
    new-identity columns rather than an infinite mask - an implausible
    transition makes "new identity" win, it doesn't crash the assignment.
    """

    def __init__(self, windows: Optional[dict] = None, p_outside: float = 1e-6):
        self._windows = {self._canon(a, b): v for (a, b), v in (windows or {}).items()}
        self._log_p_outside = math.log(p_outside)

    @staticmethod
    def _canon(cam_a, cam_b):
        return tuple(sorted((cam_a, cam_b), key=str))

    def log_p(self, cam_a, t_a_sec: float, cam_b, t_b_sec: float) -> float:
        if cam_a == cam_b or cam_a is None or cam_b is None:
            return 0.0
        window = self._windows.get(self._canon(cam_a, cam_b))
        if window is None:
            return 0.0
        min_t, max_t = window
        elapsed = t_b_sec - t_a_sec
        if elapsed < min_t:
            return self._log_p_outside
        if max_t is not None and elapsed > max_t:
            return self._log_p_outside
        return 0.0
