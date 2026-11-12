"""GlobalEntry: the node of the global gallery - one identity.

Holds the identity's moving-average embedding (EmbeddingTrack), who currently wears it, and which
tracklets have been assigned to it. All embedding arithmetic is delegated to EmbeddingTrack.

Two small deques implement the "varied views" idea:
  varied_views      (size 8)  confirmed extra looks of this body (turned around, other pose / silhouette)
  tmp_varied_views  (size 3)  quarantine: the latest embeddings of the wearer that do NOT look like the identity

view_step() is the state machine on top of them (the entry itself is not updated while a sample is quarantined):
  distance to the anchor (moving average + varied views) low                          -> 'ok'
  distance rises ("medium / high")                                                    -> 'suspect'   (sample goes to tmp_varied_views)
  it settles back below the settle level                                              -> 'settled'   (tmp -> varied_views: a good new view)
  it stays high for `confirm` consecutive samples                                     -> 'shift'     (the id moved to another body; tmp cleared)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .embedding import EmbeddingTrack, unit


@dataclass
class Membership:
    first_frame: int
    last_frame: int
    cam: object


class GlobalEntry:
    def __init__(self, global_id: int, embedding: EmbeddingTrack, frame: int, cam, key=None,
                 views_size: int = 8, tmp_size: int = 3):
        self.global_id = global_id
        self.embedding = embedding
        self.first_frame = frame          # birth frame: dwell time = now - first_frame
        self.last_frame = frame           # last frame any member updated it
        self.last_cam = cam
        self.active_key: Optional[tuple] = key   # tracklet (source_id, track_id) wearing it right now
        self.members: Dict[tuple, Membership] = {} if key is None else {key: Membership(frame, frame, cam)}
        self.mismatch_streak = 0          # consecutive rejected samples from the current wearer
        self.varied_views = deque(maxlen=views_size)
        self.tmp_varied_views = deque(maxlen=tmp_size)
        self.suspect_run = 0              # consecutive quarantined samples of the current wearer
        self.last_shift_samples: List = []

    # --- membership -----------------------------------------------------------
    def attach(self, key: tuple, frame: int, cam) -> None:
        """A tracklet starts wearing this identity."""
        if key != self.active_key:
            self._clear_quarantine()
        self.active_key = key
        self.touch(key, frame, cam)
        self.mismatch_streak = 0

    def touch(self, key: tuple, frame: int, cam) -> None:
        self.last_frame, self.last_cam = frame, cam
        m = self.members.get(key)
        if m is None:
            self.members[key] = Membership(frame, frame, cam)
        else:
            m.last_frame = frame

    def release(self, key: tuple) -> None:
        if self.active_key == key:
            self.active_key = None
            self._clear_quarantine()

    def worn_by_other(self, key: tuple, frame: int, tol: int, cam=None, live_keys=None) -> bool:
        """True if another tracklet is wearing this identity right now ('one body, one place').
        Same camera as the wearer: it is worn iff the wearer is visible this frame (in `live_keys`);
        a tracklet that was merely lost - the tracker re-acquired the person under a new id - does
        not block a takeover. Other camera: the wearer counts if it updated the entry within `tol`
        frames (absorbs the 2-4 frames by which batched streams are processed out of step)."""
        if self.active_key is None or self.active_key == key:
            return False
        if live_keys is not None and cam is not None and cam == self.last_cam:
            return self.active_key in live_keys
        return frame - self.last_frame <= tol

    # --- varied views ---------------------------------------------------------
    def _clear_quarantine(self) -> None:
        self.tmp_varied_views.clear()
        self.suspect_run = 0

    def anchor_distance(self, vec) -> float:
        """Distance of a sample to what this identity is known to look like: the moving average or any
        confirmed varied view, whichever is closest."""
        v = unit(vec)
        d = self.embedding.distance(v)
        for view in self.varied_views:
            d = min(d, 1.0 - float(view @ v))
        return d

    def add_views(self, samples, dedup_dist: float = 0.02) -> int:
        """Push samples into varied_views, skipping near-duplicates of what is already known."""
        added = 0
        for x in samples:
            if self.anchor_distance(x) >= dedup_dist:
                self.varied_views.append(unit(x)); added += 1
        return added

    def view_step(self, vec, suspect: float, settle: float, confirm: int, dedup_dist: float = 0.02) -> str:
        """One new clean sample of the current wearer. Returns 'ok' | 'suspect' | 'settled' | 'shift'."""
        v = unit(vec)
        d = self.anchor_distance(v)
        if self.suspect_run == 0:
            if d < suspect:
                return "ok"
            self.tmp_varied_views.append(v); self.suspect_run = 1
            return "suspect"
        if d < settle:                                             # the distance came back: same body, a new look
            self.add_views(list(self.tmp_varied_views), dedup_dist)
            self._clear_quarantine()
            return "settled"
        self.tmp_varied_views.append(v); self.suspect_run += 1
        if self.suspect_run >= confirm:                            # it stays high: the id moved to another body
            self.last_shift_samples = list(self.tmp_varied_views)
            self._clear_quarantine()
            return "shift"
        return "suspect"

    # --- embedding ------------------------------------------------------------
    def absorb(self, tracklet_embedding: EmbeddingTrack, damping: float = 0.0) -> None:
        """Fold a tracklet's current moving average into the identity (done once at commit)."""
        if not tracklet_embedding.empty:
            self.embedding.update(tracklet_embedding.value, damping)

    def observe(self, vec, outlier_thresh: float, damping: float = 0.0) -> bool:
        """Gate one new sample against the identity. Far samples are refused and counted as a
        mismatch; near samples update the moving average. Returns accepted."""
        if self.embedding.distance(vec) >= outlier_thresh:
            self.mismatch_streak += 1
            return False
        self.mismatch_streak = 0
        self.embedding.update(vec, damping)
        return True

    def __repr__(self):
        return (f"GlobalEntry(gid={self.global_id}, wearer={self.active_key}, n={self.embedding.count}, "
                f"members={len(self.members)}, last_frame={self.last_frame})")
