"""Associator: tracklet intake -> status -> fast-commit | mint | pending, over a FAISS gallery.

Per tracklet (Tracklet.assoc_status):
  NEW        first time we see it; triaged immediately
  PENDING    not (yet) eligible, or eligible but nothing to commit to and not mature enough to mint
  COMMITTED  carries a global id (t_global_id) and keeps its global_entry fresh

No clustering / solver in this stage: an eligible tracklet either fast-commits to an existing
global_entry, or (once mature, and only when it is clearly novel) mints a new one. A near-tie
between entries never mints: it waits, then commits to the nearest after `ambiguous_patience_frames`.
"""
from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from .base import AssociationBase
from .embedding import EmbeddingTrack
from .fast_commit import Decision, FastCommit
from .gallery import FaissGallery
from .global_entry import GlobalEntry

log = logging.getLogger("association")


class AssocStatus(enum.Enum):
    NEW = "new"
    PENDING = "pending"
    COMMITTED = "committed"


@dataclass
class AssociatorConfig:
    min_tracklet_len: int = 15           # frames a tracklet must have lived
    min_embeddings: int = 2              # accepted ReID samples before it may fast-commit
    mint_min_embeddings: int = 4         # ...before it may mint a new identity
    fast_commit_thresh: float = 0.3
    min_margin: float = 0.07
    ema_alpha: float = 0.9             # global_entry moving average
    outlier_reject_thresh: float = 0.30
    revoke_streak: int = 5
    overlap_freeze_thresh: float = 0.2
    overlap_cooldown_frames: int = 10
    co_occurrence_tol_frames: int = 5
    ambiguous_patience_frames: int = 45  # near-tie: wait this long, then commit to the nearest entry
    # varied views (see GlobalEntry): quarantine samples that stop looking like the identity; if the distance
    # settles back they become a new view of the same body, if it stays high the id moved to another body
    views_enabled: bool = False
    view_suspect_dist: float = 0.06      # distance to the identity's anchor from which a sample is quarantined
    view_settle_ratio: float = 0.8       # "settled" = back below suspect * ratio
    view_confirm_frames: int = 4         # still high after this many consecutive samples = the body changed
    view_dedup_dist: float = 0.02        # a view this close to a known one is not stored again
    views_size: int = 8                  # varied_views capacity per identity
    tmp_views_size: int = 3              # tmp_varied_views capacity (the quarantine)
    views_in_search: bool = True         # the stored views are also searched by fast-commit
    use_gpu: bool = False


class Associator(AssociationBase):
    """Fast-commit association (v1 of the minimal package). Implements the AssociationBase contract."""

    @classmethod
    def from_config(cls, cfg: dict) -> "Associator":
        return cls(AssociatorConfig(**(cfg or {})))

    def __init__(self, cfg: Optional[AssociatorConfig] = None):
        super().__init__()
        self.cfg = cfg or AssociatorConfig()
        self.gallery = FaissGallery(use_gpu=self.cfg.use_gpu,
                                    views_in_search=self.cfg.views_enabled and self.cfg.views_in_search)
        self.fast_commit = FastCommit(self.gallery, self.cfg.fast_commit_thresh,
                                      self.cfg.min_margin, self.cfg.co_occurrence_tol_frames)
        self._next_gid = 1
        self._cooldown: Dict[tuple, int] = {}
        self._amb_since: Dict[tuple, int] = {}         # key -> first frame it was ambiguous
        self._tracked: Dict[tuple, object] = {}       # key -> tracklet, for n_pending()

    # --- API -------------------------------------------------------------------
    def step(self, cam_id, live: list, frame: int) -> list:
        """Once per camera frame. `live` = that camera's ACTIVE tracklets. Returns the tracklets
        that obtained a global id during this call."""
        newly = []
        live_keys = {t.key for t in live}
        for t in live:
            if getattr(t, "assoc_status", None) is None:
                t.assoc_status = AssocStatus.NEW
            self._tracked[t.key] = t
            if t.assoc_status is AssocStatus.COMMITTED:
                self._refresh(t, cam_id, frame, live_keys)
            elif self._advance(t, cam_id, frame, live_keys):
                newly.append(t)
        return newly

    def release(self, key: tuple) -> None:
        """The tracklet left: free its identity (the entry stays in the gallery)."""
        self._cooldown.pop(key, None)
        self._amb_since.pop(key, None)
        self._tracked.pop(key, None)
        for e in self.gallery.entries.values():
            e.release(key)

    def size(self) -> int:
        return len(self.gallery)

    def n_pending(self) -> int:
        return sum(1 for t in self._tracked.values() if t.assoc_status is AssocStatus.PENDING)

    def entry(self, gid: int) -> Optional[GlobalEntry]:
        return self.gallery.entries.get(gid)

    # --- internals --------------------------------------------------------------
    def _eligible(self, t) -> bool:
        return (t.clean and t.tracklet_len >= self.cfg.min_tracklet_len
                and t.embedding.count >= self.cfg.min_embeddings)

    def _advance(self, t, cam_id, frame: int, live_keys: set) -> bool:
        """NEW / PENDING tracklet: triage, then fast-commit or mint. True if it got an id."""
        key = t.key
        if t.peer_iou > self.cfg.overlap_freeze_thresh:        # occluded: wait until it separates
            self._cooldown[key] = self.cfg.overlap_cooldown_frames
        if self._cooldown.get(key, 0) > 0 and t.peer_iou <= self.cfg.overlap_freeze_thresh:
            self._cooldown[key] -= 1
            t.assoc_status = AssocStatus.PENDING
            return False
        if not self._eligible(t):
            t.assoc_status = AssocStatus.PENDING
            return False
        t.assoc_status = AssocStatus.PENDING
        if not t.new_embedding_this_frame:                      # decide only on fresh evidence
            return False

        d = self.fast_commit.decide(t.embedding.value, key, frame, cam_id, live_keys)
        forced = False
        if d.reason == "ambiguous":
            since = self._amb_since.setdefault(key, frame)
            if frame - since < self.cfg.ambiguous_patience_frames:
                return False                                    # wait: the average may firm up
            d = Decision(d.candidate, "match", d.dist, d.margin)   # patience over: nearest entry wins
            forced = True
        if d.gid is not None:
            entry = self.gallery.entries[d.gid]
            entry.attach(key, frame, cam_id)
            entry.absorb(t.embedding)
            self.gallery.mark_dirty()
            self._amb_since.pop(key, None)
            self._commit(t, d.gid, frame)
            self._emit("fast_commit", frame, t, global_id=d.gid, dist=round(d.dist, 4),
                       margin=round(d.margin, 4), forced=forced)
            return True
        if t.embedding.count >= self.cfg.mint_min_embeddings:   # clearly novel: empty / too_far / worn_elsewhere
            gid = self._mint(t, cam_id, frame)
            self._commit(t, gid, frame)
            self._emit("new_identity", frame, t, global_id=gid, reason=d.reason)
            return True
        return False

    def _mint(self, t, cam_id, frame: int) -> int:
        gid = self._next_gid
        self._next_gid += 1
        emb = EmbeddingTrack(self.cfg.ema_alpha)
        emb.update(t.embedding.value)
        emb.count = t.embedding.count
        self.gallery.add(GlobalEntry(gid, emb, frame, cam_id, key=t.key,
                                     views_size=self.cfg.views_size, tmp_size=self.cfg.tmp_views_size))
        return gid

    @staticmethod
    def _commit(t, gid: int, frame: int) -> None:
        t.t_global_id = gid
        t.t_identity_since_frame = frame
        t.assoc_status = AssocStatus.COMMITTED

    def _refresh(self, t, cam_id, frame: int, live_keys: set) -> None:
        """COMMITTED tracklet: keep the entry alive, learn from clean new samples, watch for a
        silent identity swap (varied-views quarantine, then sustained mismatch -> revoke)."""
        entry = self.gallery.entries.get(t.t_global_id)
        if entry is None:
            return
        if entry.active_key != t.key:
            entry.attach(t.key, frame, cam_id)
        else:
            entry.touch(t.key, frame, cam_id)
        if not (t.new_embedding_this_frame and t.clean):
            return
        c = self.cfg
        if c.views_enabled:
            r = entry.view_step(t.embedding.last, c.view_suspect_dist, c.view_suspect_dist * c.view_settle_ratio,
                                c.view_confirm_frames, c.view_dedup_dist)
            if r == "suspect":                 # quarantined in tmp_varied_views: the identity is NOT updated with it
                return
            if r == "settled":                 # came back: a new look of the same body
                self.gallery.mark_dirty()
                self._emit("view_added", frame, t, global_id=entry.global_id, views=len(entry.varied_views))
            elif r == "shift":                 # stayed high: the id moved to another body
                self._shift(t, entry, cam_id, frame, live_keys)
                return
        if entry.observe(t.embedding.last, self.cfg.outlier_reject_thresh, damping=t.peer_iou):
            self.gallery.mark_dirty()
        elif entry.mismatch_streak >= self.cfg.revoke_streak:
            self._emit("revoked", frame, t, global_id=entry.global_id, streak=entry.mismatch_streak)
            entry.release(t.key)
            entry.mismatch_streak = 0
            t.t_global_id = 0
            t.assoc_status = AssocStatus.PENDING

    def _shift(self, t, entry, cam_id, frame: int, live_keys: set) -> None:
        """The wearer's embeddings left the identity and stayed away: give the id up and decide again."""
        samples, old = list(entry.last_shift_samples), entry.global_id
        entry.release(t.key)
        entry.mismatch_streak = 0
        t.t_global_id = 0
        t.assoc_status = AssocStatus.PENDING
        t.embedding.reset(samples)                       # the tracklet now means "what it looks like after the shift"
        self._amb_since.pop(t.key, None)
        self._cooldown.pop(t.key, None)
        self._advance(t, cam_id, frame, live_keys)       # call step again: fast-commit somewhere else, or mint
        new = t.t_global_id
        self._emit("shift", frame, t, global_id=old, to_global_id=new, back_to_same=(new == old))
        if new == old:                                   # it matched the same identity again: a new view after all
            entry.add_views(samples, self.cfg.view_dedup_dist)
            self.gallery.mark_dirty()

    def _emit(self, kind: str, frame: int, t, **fields) -> None:
        ev = {"event": kind, "frame": frame, "source": t.source_id, "track_id": t.track_id, **fields}
        self.events.append(ev)
        log.info("%s %s", kind, fields)
