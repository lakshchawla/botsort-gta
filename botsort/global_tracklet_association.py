from __future__ import annotations

import json
import math
import os
import queue
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import faiss
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN

from . import matching
from .priors import SpatioTemporalPrior

SQRT_2PI = math.sqrt(2.0 * math.pi)


def _gaussian_log_likelihood(x: float, mu: float, sigma: float) -> float:
    return -0.5 * ((x - mu) / sigma) ** 2 - math.log(sigma * SQRT_2PI)


class CameraTopologyPrior:  
    def __init__(self, bonuses: Optional[dict] = None, default: float = 0.0):
        self._bonuses = {self._canon(a, b): v for (a, b), v in (bonuses or {}).items()}
        self._default = default

    @staticmethod
    def _canon(cam_a, cam_b):
        return tuple(sorted((cam_a, cam_b), key=str))

    def log_bonus(self, cam_a, cam_b) -> float:
        if cam_a == cam_b:
            return 0.0
        return self._bonuses.get(self._canon(cam_a, cam_b), self._default)


@dataclass
class FeatureStreamConfig:
    """Calibration for one named ReID embedding stream (one model).

    name                   : must match the key used in STrack.smooth_feats
                              (e.g. "reidnet", "clipreid").
    weight                 : this stream's contribution to the combined
                              appearance log-likelihood - independent models
                              can be trusted more or less (a stronger
                              discriminator gets a higher weight).
    sigma_feat             : cosine-distance std for this stream's Gaussian
                              appearance term.
    outlier_reject_thresh  : cosine-distance floor above which a new
                              observation is "confidently inconsistent" with
                              this stream's established centroid and is
                              dropped rather than blended in (see
                              GalleryEntry.add_evidence). Also used to derive
                              this stream's share of the default
                              new-identity cost (see GTA.__init__).
    is_primary             : exactly one stream should be marked primary -
                              it backs the FAISS fast-commit index and is the
                              stream BoT-SORT's own short-term association
                              keeps using via STrack.smooth_feat/curr_feat
                              (see bot_sort.py). If none is marked, the first
                              stream in the list is used.
    """
    name: str
    weight: float = 1.0
    sigma_feat: float = 0.15
    outlier_reject_thresh: float = 0.30
    is_primary: bool = False


def multi_model_log_likelihood(feats_a: dict, feats_b: dict, streams: dict) -> Optional[float]:
    """Weighted sum of per-stream Gaussian appearance log-likelihoods, over
    whichever stream names both feature bundles actually share. None if they
    share no configured stream in common - nothing to compare (e.g. one
    tracklet only ever got a "reidnet" embedding, the gallery entry only has
    "clipreid" - shouldn't normally happen since both run every frame, but a
    momentarily-missing model shouldn't crash the solver)."""
    common = [name for name in streams if name in feats_a and name in feats_b]
    if not common:
        return None
    total = 0.0
    for name in common:
        cfg = streams[name]
        cos_dist = 1.0 - float(np.dot(feats_a[name], feats_b[name]))
        total += cfg.weight * _gaussian_log_likelihood(cos_dist, 0.0, cfg.sigma_feat)
    return total


class GalleryEntry:
    """One real-world identity, living in the gallery for the lifetime of the
    process (never pruned) - this unbounded lifetime is what fixes the old
    GTA's "reappear after prune_after_frames and you're unlinkable forever"
    failure mode.

    Appearance is summarized as one EMA centroid PER STREAM (one per ReID
    model), each gated independently against corruption by
    `add_evidence`'s per-stream outlier check.
    """

    def __init__(self, global_id: int, feats: dict, frame_id: int,
                 bbox: Optional[np.ndarray], ema_alpha: float = 0.9, cam_source=None):
        self.global_id = global_id
        self.active_tid: Optional[int] = None
        # id_assignment_frame: the frame this identity was BORN - i.e. when
        # _register() first minted this global_id. Immutable (never touched
        # again after construction), unlike last_frame below which keeps
        # advancing on every accepted observation. This is what dwelling-time
        # detection is built on (see botsort/dwelling.py): dwell time for a
        # global_id is current_frame - first_frame, read straight off the
        # SAME GalleryEntry regardless of which camera currently holds it -
        # a cluster's identity stays one object across cameras by
        # construction, so this needs no separate per-camera bookkeeping.
        self.first_frame = frame_id
        self.last_frame = frame_id
        self.last_bbox = bbox.copy() if bbox is not None else None
        self.ema_alpha = ema_alpha
        self.centroids: dict = {name: vec.copy() for name, vec in feats.items()}
        self.update_count = 1
        # Logging-only: which camera last claimed this identity.
        self.last_cam_source = cam_source
        # Consecutive REJECTED tracklet-level observations (weighted-majority
        # verdict across streams, see add_evidence) for whoever currently
        # holds active_tid. Mirrors GlobalRegistry's identity_revoke_streak -
        # if BoT-SORT's own association silently handed this track_id to a
        # different person, this is what eventually notices and revokes.
        self.mismatch_streak = 0

    def add_evidence(self, feats: dict, frame_id: int, bbox: Optional[np.ndarray],
                      stream_cfg: dict, occlusion_score: float = 0.0) -> bool:
        """Blend a new tracklet observation's per-stream features into this
        identity's centroids. Each stream is gated independently against ITS
        OWN centroid (a bad frame on one model can't corrupt another
        model's), but the overall accept/reject verdict - what
        `mismatch_streak` counts, and therefore what identity_revoke_streak
        eventually acts on - is a WEIGHTED-MAJORITY vote across streams, so
        one model's transient noise can't revoke a good identity on its own.

        occlusion_score (0.0-1.0): scales down the new observation's
        contribution the same way GlobalRegistry's did - effective_alpha ->
        1.0 (freeze) as occlusion_score -> 1.0.
        """
        self.last_frame = frame_id
        if bbox is not None:
            self.last_bbox = bbox.copy()

        reject_weight = 0.0
        total_weight = 0.0
        effective_alpha = self.ema_alpha + (1.0 - self.ema_alpha) * float(occlusion_score)
        for name, vec in feats.items():
            cfg = stream_cfg.get(name)
            if cfg is None:
                continue
            if name not in self.centroids:
                # A stream this identity hasn't seen before (e.g. a model
                # that was momentarily missing on earlier observations) -
                # seed it rather than gating against a centroid that doesn't
                # exist yet.
                self.centroids[name] = vec.copy()
                continue
            total_weight += cfg.weight
            cos_dist = 1.0 - float(np.dot(self.centroids[name], vec))
            if cos_dist >= cfg.outlier_reject_thresh:
                reject_weight += cfg.weight
                continue
            c = effective_alpha * self.centroids[name] + (1.0 - effective_alpha) * vec
            cn = np.linalg.norm(c)
            self.centroids[name] = (c / cn).astype(np.float32) if cn > 1e-9 else c.astype(np.float32)

        self.update_count += 1
        accepted = total_weight <= 0.0 or reject_weight < 0.5 * total_weight
        if accepted:
            self.mismatch_streak = 0
        else:
            self.mismatch_streak += 1
        return accepted

    def __repr__(self):
        return (f"GalleryEntry(gid={self.global_id}, tid={self.active_tid}, "
                f"streams={sorted(self.centroids.keys())}, n_updates={self.update_count}, "
                f"last_frame={self.last_frame})")


@dataclass
class TrackletEvidence:
    tid: int
    cam_source: object
    track_ref: object
    first_visible: int
    last_visible: int
    feats: dict
    bbox: Optional[np.ndarray] = None
    cannot_link_gids: set = field(default_factory=set)
    hits: int = 1
    dead: bool = False
    # Set by _try_fast_commit(): this tracklet was handed an identity
    # immediately (skipping the queue), but NOT yet confirmed - it stays in
    # the pending pool so the next _solve_pending tick re-evaluates it
    # against the whole gallery with full exclusivity, same as any other
    # pending tracklet. If the solver agrees, this is a no-op (confirm); if
    # it disagrees, the tracklet is reassigned there (correct) - see
    # _try_fast_commit's docstring for why this is the fix for a fast-commit
    # that can otherwise never be undone.
    provisional_gid: Optional[int] = None


@dataclass
class EmbeddingWindow:
    """Bounded per-tracklet history of periodically-sampled multi-stream
    embeddings - richer temporal evidence than a single frame's smooth_feat.
    Maintained for every pending (not-yet-identified or still-provisional)
    tracklet regardless of whether a confusion zone currently applies
    (cheap: a maxlen deque append every window_sample_interval frames), and
    only ever CONSULTED when the fast-commit path's nearest neighbor turns
    out to be in a DBSCAN-flagged confusion group (see
    GTA._resolve_confusion) - a single frame's embedding isn't reliable
    enough to tell two suspiciously-similar identities apart, but an
    average over several samples spread across ~window_size *
    window_sample_interval frames is."""
    samples: deque = field(default_factory=lambda: deque(maxlen=10))
    last_sampled_frame: int = -10 ** 9


class GTA:
    """See module docstring. All thresholds need empirical calibration
    against the CHIRLA harness (evaluate_tracking.py) - the defaults here
    are starting points carried over from GlobalRegistry's own previously-
    used values where a direct analogue exists, not tuned values (same
    posture every threshold in this codebase already takes).

    streams                : list[FeatureStreamConfig], one per ReID model.
                              Defaults to a single stream named "primary"
                              (matches STrack._DEFAULT_STREAM) for back-
                              compat with a single-SGIE pipeline.
    min_tracklet_len       : frames a tracklet must survive before it's
                              trusted enough to enter the pending pool
                              (mirrors GlobalRegistry's min_frames / the old
                              GTA's min_tracklet_len).
    solver_interval_frames : how often the Hungarian batch solve over the
                              pending pool runs (mirrors the old GTA's
                              window_frames / GlobalRegistry's
                              solver_interval_frames).
    mu_gap_sec/sigma_gap_sec : center/spread of the time-gap Gaussian term -
                              a loose prior in the absence of real per-
                              camera-pair topology (see camera_prior).
    new_identity_cost      : flat override for tau_new (the solver's dummy
                              "mint a fresh identity" column cost). If None
                              (default), computed PER TRACKLET from exactly
                              the streams it has evidence for, as the cost
                              of "a borderline-rejected observation" on each
                              (see __init__) - a real gallery candidate must
                              beat that to win the tracklet.
    fast_commit_thresh/min_margin : gate the optional fast-commit path
                              (FAISS top-2 + margin on the primary stream) -
                              deliberately tighter than the solver's implicit
                              acceptance bar, since a false positive here
                              bypasses Hungarian exclusivity entirely.
    identity_revoke_streak : see GalleryEntry.add_evidence / mismatch_streak.
    overlap_freeze_thresh/overlap_cooldown_frames : while two tracks overlap
                              (occlusion proxy via peer IOU), don't refresh
                              or query a not-yet-identified tracklet's
                              evidence; after separation, wait
                              overlap_cooldown_frames for the embedding to
                              recover. Mirrors GlobalRegistry exactly.
    confusion_eps/confusion_refresh_interval : periodic DBSCAN over gallery
                              primary-stream centroids to detect groups of
                              suspiciously close identities (hard samples).
                              Purely a DETECTOR - see window_size etc. below
                              for how a confusion-flagged candidate actually
                              gets resolved. The batch solver doesn't need
                              this: Hungarian exclusivity already
                              structurally prevents two candidates both
                              winning one identity.
    window_size/window_sample_interval : every pending (not-yet-identified
                              or still-provisional) tracklet keeps a bounded
                              history of up to window_size multi-stream
                              embedding snapshots, sampled once every
                              window_sample_interval frames (see
                              EmbeddingWindow). Cheap to maintain
                              unconditionally; only read when the
                              fast-commit path's nearest neighbor lands in a
                              DBSCAN-flagged confusion group.
    window_min_samples/confusion_margin : when a fast-commit candidate is in
                              a confusion group, a SINGLE frame's embedding
                              isn't trusted to disambiguate it - the whole
                              window is scored against every identity in
                              that group (average multi-model
                              log-likelihood) instead. window_min_samples is
                              the fewest window samples required before this
                              is even attempted (otherwise defer to the
                              pending pool); confusion_margin is the
                              required log-likelihood gap between the best-
                              and second-best-scoring identity in the group
                              before committing to the winner - too close
                              and _resolve_confusion declines to guess. See
                              _resolve_confusion's docstring.
    """

    def __init__(
        self,
        streams: Optional[list] = None,
        min_tracklet_len: int = 30,
        solver_interval_frames: int = 600,
        mu_gap_sec: float = 0.5,
        sigma_gap_sec: float = 20.0,
        new_identity_cost: Optional[float] = None,
        fast_commit_thresh: float = 0.2,
        min_margin: float = 0.09,
        ema_alpha: float = 0.9,
        identity_revoke_streak: int = 5,
        overlap_freeze_thresh: float = 0.3,
        overlap_cooldown_frames: int = 10,
        confusion_eps: float = 0.15,
        confusion_refresh_interval: int = 100,
        window_size: int = 10,
        window_sample_interval: int = 5,
        window_min_samples: int = 5,
        confusion_margin: float = 1.0,
        camera_prior: Optional[CameraTopologyPrior] = None,
        st_prior: Optional[SpatioTemporalPrior] = None,
        frame_rate: float = 30.0,
        use_gpu: bool = True,
        reid_log_path: Optional[str] = None,
        log_flush_interval: int = 500,
    ):
        if not streams:
            streams = [FeatureStreamConfig(name="primary", is_primary=True)]
        self._stream_cfg: dict = {s.name: s for s in streams}
        primary = next((s.name for s in streams if s.is_primary), streams[0].name)
        self._primary_stream = primary

        self.min_tracklet_len = min_tracklet_len
        self.solver_interval_frames = solver_interval_frames
        self.mu_gap_sec = mu_gap_sec
        self.sigma_gap_sec = sigma_gap_sec
        self.fast_commit_thresh = fast_commit_thresh
        self.min_margin = min_margin
        self.ema_alpha = ema_alpha
        self.identity_revoke_streak = identity_revoke_streak
        self.overlap_freeze_thresh = overlap_freeze_thresh
        self.overlap_cooldown_frames = overlap_cooldown_frames
        self.confusion_eps = confusion_eps
        self.confusion_refresh_interval = confusion_refresh_interval
        self.window_size = window_size
        self.window_sample_interval = window_sample_interval
        self.window_min_samples = min(window_min_samples, window_size)
        self.confusion_margin = confusion_margin
        self.camera_prior = camera_prior
        self.st_prior = st_prior
        self.frame_rate = frame_rate

        # Per-stream appearance log-likelihood AT the "borderline rejected
        # observation" boundary (cos_dist == outlier_reject_thresh) - always
        # negative. Combined with _time_best_ll (the time term's own ceiling
        # - gap_sec exactly at mu_gap_sec) below, this gives a
        # same-units-as-the-real-cost-matrix "boundary" combined
        # log-likelihood: a real candidate must do BETTER than "borderline
        # appearance + best-possible timing" to win, whether that candidate
        # is another pending tracklet (_cluster_pending) or an existing
        # gallery identity (_solve_pending)'s dummy new-identity column.
        #
        # This isn't optional bookkeeping: sigma_gap_sec's own Gaussian
        # normalization constant (-log(sigma_gap_sec*sqrt(2*pi))) is
        # strongly negative for any reasonably loose time prior (e.g. -3.9
        # at sigma_gap_sec=20), so a purely-appearance-based threshold
        # would incorrectly veto even a perfect appearance+timing match -
        # the same normalization-constant trap the old GTA's manually-tuned
        # link_threshold=-6.0 existed to work around.
        self._time_best_ll = _gaussian_log_likelihood(self.mu_gap_sec, self.mu_gap_sec, self.sigma_gap_sec)
        self._per_stream_reject_ll = {
            name: cfg.weight * _gaussian_log_likelihood(cfg.outlier_reject_thresh, 0.0, cfg.sigma_feat)
            for name, cfg in self._stream_cfg.items()
        }
        self._flat_new_identity_cost = new_identity_cost

        self._tid_cooldown: dict[int, int] = {}
        self._confusion_gids: set[int] = set()
        # gid -> the full set of gids (including itself) DBSCAN grouped it
        # with this refresh - lets _resolve_confusion look up "who else is
        # this candidate confusable with" in O(1), not just "is it confused
        # at all". Rebuilt alongside _confusion_gids in
        # _refresh_confusion_zones.
        self._confusion_group_of: dict[int, frozenset] = {}
        self._last_confusion_refresh: int = 0
        # tid -> EmbeddingWindow, one per pending (not-yet-identified or
        # still-provisional) tracklet - see EmbeddingWindow's docstring.
        # Populated in _refresh_pending, consulted only by
        # _resolve_confusion, cleaned up once a tracklet is finally
        # resolved (_solve_pending).
        self._embedding_windows: dict[int, EmbeddingWindow] = {}

        self._pending: dict[int, TrackletEvidence] = {}
        self._last_solve_frame: int = 0
        # gid -> frame_id at which an OWNED live track last refreshed that
        # identity - read when refreshing pending evidence / building the
        # solver's cost matrix so an identity actively worn by someone else
        # can't be double-claimed (co-occurrence cannot-link).
        self._gid_seen_at: dict[int, int] = {}

        self._next_gid = 1
        self._gallery: dict[int, GalleryEntry] = {}
        self._tid_to_gid: dict[int, int] = {}

        # FAISS index over primary-stream centroids only - backs the
        # fast-commit path. The batch solver doesn't use FAISS at all (its
        # cost matrix is dense over the pending pool x gallery, both
        # expected to stay small - "one camera network's worth of people").
        self._use_gpu = use_gpu
        self._fast_index: Optional[faiss.Index] = None
        self._fast_index_dim: Optional[int] = None
        self._fast_index_gids: list[int] = []
        self._gallery_dirty = False

        self.reid_log_path = reid_log_path
        self.log_flush_interval = log_flush_interval
        self._log_queue: "queue.Queue" = queue.Queue()
        self._log_stop = threading.Event()
        self._log_thread: Optional[threading.Thread] = None
        if reid_log_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(reid_log_path)), exist_ok=True)
            self._log_thread = threading.Thread(
                target=self._log_worker, name="gta-log-writer", daemon=True
            )
            self._log_thread.start()

    def _boundary_ll(self, stream_names) -> float:
        """Combined log-likelihood of 'a borderline-rejected observation on
        each of these streams, arriving at exactly the best possible time' -
        the merge/match acceptance floor shared by _cluster_pending and
        _new_identity_cost_feats."""
        return sum(self._per_stream_reject_ll.get(name, 0.0) for name in stream_names) + self._time_best_ll

    # --- per-frame driver --------------------------------------------------

    def step(self, tracker, frame_id: int) -> list:
        """Run once per camera per batch tick (same call site/contract as
        the old GTA/GlobalRegistry - see multi_camera_tracker.py). Cheap
        per-frame bookkeeping (refresh already-identified centroids, refresh
        pending evidence, attempt the fast-commit path) every call; the
        Hungarian batch solve only runs once every `solver_interval_frames`.

        Returns the STracks whose t_global_id was just resolved during this
        call - a fresh fast-commit, or a batch-solve outcome, this tick.
        """
        cam_source = getattr(tracker, "cam_source", None)
        assigned_this_step: list = []

        self._refresh_confusion_zones(frame_id)

        live = tracker.tracked_stracks
        live_tids = {t.track_id for t in live}

        # --- Pairwise inter-track overlap (occlusion proxy), same as
        # GlobalRegistry: scales down centroid updates and freezes pending
        # evidence while two people overlap.
        if len(live) >= 2:
            iou_cost = matching.iou_distance(live, live)
            iou_mat = 1.0 - iou_cost
            np.fill_diagonal(iou_mat, 0.0)
            peer_iou_vals = iou_mat.max(axis=1)
        else:
            peer_iou_vals = np.zeros(len(live))
        track_overlap = {t.track_id: float(peer_iou_vals[i]) for i, t in enumerate(live)}

        for track in live:
            tid = track.track_id
            occlusion_score = track_overlap.get(tid, 0.0)
            ev = self._pending.get(tid)
            provisional = ev is not None and ev.provisional_gid is not None

            if track.t_global_id != 0 and not provisional:
                # Solver-confirmed (or never went through the fast-commit
                # path at all): fully trusted, gets the always-refresh +
                # revoke-streak treatment.
                self._refresh_identified(track, cam_source, frame_id, occlusion_score)
                continue

            if track.is_touching_edge or track.tracklet_len < self.min_tracklet_len or not track.smooth_feats:
                continue

            if occlusion_score > self.overlap_freeze_thresh:
                self._tid_cooldown[tid] = self.overlap_cooldown_frames
                continue
            if self._tid_cooldown.get(tid, 0) > 0:
                self._tid_cooldown[tid] -= 1
                continue

            # Not-yet-identified AND provisional (fast-committed but still
            # awaiting solver confirmation) both keep accumulating PENDING
            # evidence - the only difference is a provisional tracklet
            # doesn't get a second fast-commit attempt (it already has a
            # claim; let the next solve confirm or correct it).
            self._refresh_pending(track, cam_source, frame_id)
            if not provisional and self._try_fast_commit(track, cam_source, frame_id, live_tids):
                assigned_this_step.append(track)

        if self._gallery_dirty:
            self._rebuild_fast_index()
            self._gallery_dirty = False

        if frame_id - self._last_solve_frame >= self.solver_interval_frames:
            self._last_solve_frame = frame_id
            assigned_this_step.extend(self._solve_pending(frame_id))
            if self._gallery_dirty:
                self._rebuild_fast_index()
                self._gallery_dirty = False

        if self.reid_log_path is not None and assigned_this_step:
            snapshot = {
                t.t_global_id: self._gallery[t.t_global_id].centroids.get(self._primary_stream)
                for t in assigned_this_step if t.t_global_id in self._gallery
            }
            snapshot = {k: v for k, v in snapshot.items() if v is not None}
            if snapshot:
                self._log_queue.put((frame_id, snapshot))

        return assigned_this_step

    def _refresh_identified(self, track, cam_source, frame_id: int, occlusion_score: float) -> None:
        """A track that already carries a global_id: keep its GalleryEntry's
        centroids fresh, and watch for a silent identity swap underneath it
        (BoT-SORT's own IOU association handing this track_id to a different
        person mid-track) via the mismatch-streak revoke, exactly mirroring
        GlobalRegistry.step()'s always-refresh behavior for identified
        tracks."""
        tid = track.track_id
        entry = self._gallery.get(track.t_global_id)
        if entry is None:
            return
        entry.active_tid = tid
        entry.last_cam_source = cam_source
        self._tid_to_gid[tid] = track.t_global_id
        self._gid_seen_at[track.t_global_id] = frame_id
        if not track.smooth_feats or track.is_touching_edge:
            return
        self._gallery_dirty = True
        accepted = entry.add_evidence(track.smooth_feats, frame_id, getattr(track, "tlwh", None),
                                       self._stream_cfg, occlusion_score=occlusion_score)
        if not accepted and entry.mismatch_streak >= self.identity_revoke_streak:
            print(f"[GTA] identity revoked: track={tid} global_id={entry.global_id} "
                  f"(weighted-majority appearance mismatch {entry.mismatch_streak} observations straight)")
            entry.active_tid = None
            entry.mismatch_streak = 0
            self._tid_to_gid.pop(tid, None)
            track.t_global_id = 0

    # --- pending pool + fast-commit path ------------------------------------

    def _refresh_pending(self, track, cam_source, frame_id: int) -> None:
        tid = track.track_id
        feats_copy = {name: vec.astype(np.float32).copy() for name, vec in track.smooth_feats.items()}
        bbox_copy = np.asarray(track.tlwh, dtype=np.float64).copy() if hasattr(track, "tlwh") else None
        ev = self._pending.get(tid)
        if ev is None:
            ev = TrackletEvidence(
                tid=tid, cam_source=cam_source, track_ref=track,
                first_visible=frame_id, last_visible=frame_id,
                feats=feats_copy, bbox=bbox_copy,
            )
            self._pending[tid] = ev
        else:
            ev.last_visible = frame_id
            ev.feats = feats_copy
            ev.bbox = bbox_copy
            ev.hits += 1
            ev.track_ref = track

        # Cross-camera co-occurrence: any identity actively worn by a
        # DIFFERENT track this same frame can never be this tracklet - one
        # body, one place. All cameras sharing this GTA instance stamp
        # _gid_seen_at with the same frame_id per batch tick (see
        # MultiCameraTracker.update_batch), so equality means "on screen at
        # the same instant".
        for gid, seen_at in self._gid_seen_at.items():
            if seen_at == frame_id:
                entry = self._gallery.get(gid)
                if entry is not None and entry.active_tid not in (None, tid):
                    ev.cannot_link_gids.add(gid)

        # Moving embedding window: sample once every window_sample_interval
        # frames (not every frame - window_size samples should span several
        # seconds, not a fraction of one) - see EmbeddingWindow's docstring.
        hist = self._embedding_windows.get(tid)
        if hist is None:
            hist = self._embedding_windows[tid] = EmbeddingWindow(samples=deque(maxlen=self.window_size))
        if frame_id - hist.last_sampled_frame >= self.window_sample_interval:
            hist.samples.append(feats_copy)
            hist.last_sampled_frame = frame_id

    def _try_fast_commit(self, track, cam_source, frame_id: int, live_tids: set) -> bool:
        """Deliberately conservative: FAISS top-2 + margin on the primary
        stream only. Lets an obviously-confident re-appearance skip the
        queue instead of waiting up to solver_interval_frames - kept for
        latency/UX, not correctness (see module docstring).

        The commit is PROVISIONAL, not final: track.t_global_id is set
        immediately (so display/downstream consumers see it right away), but
        the tracklet's evidence stays in the pending pool
        (TrackletEvidence.provisional_gid marks it) instead of being popped,
        and its evidence is NOT blended into the gallery centroid yet -
        deferred until _solve_pending actually re-checks this pairing
        against the full gallery with exclusivity at the next tick. If the
        solver agrees, the deferred evidence is folded in then; if it
        disagrees, the tracklet is reassigned there instead. Without this,
        a fast-commit is a one-way door - the periodic solver never revisits
        an already-identified track, so a wrong fast-commit could otherwise
        never be corrected except via the (much slower, sustained-mismatch-
        only) identity_revoke_streak safety net.
        """
        feat = track.smooth_feats.get(self._primary_stream)
        if feat is None or self._fast_index is None or self._fast_index.ntotal == 0:
            return False

        vec = np.ascontiguousarray(feat.astype(np.float32).reshape(1, -1))
        k = min(2, self._fast_index.ntotal)
        sims, idxs = self._fast_index.search(vec, k=k)
        pos = int(idxs[0, 0])
        if pos < 0:
            return False

        gid = self._fast_index_gids[pos]
        cos_dist = 1.0 - float(sims[0, 0])
        cos_dist2 = 1.0
        if idxs.shape[1] > 1 and int(idxs[0, 1]) >= 0:
            cos_dist2 = 1.0 - float(sims[0, 1])

        tid = track.track_id
        resolution_log = ""
        if gid in self._confusion_gids:
            # A single frame's embedding isn't trusted to tell suspiciously-
            # similar identities apart - use this tracklet's whole moving
            # window instead (see _resolve_confusion). Overrides the
            # FAISS-nearest-neighbor gid entirely if the window disagrees.
            resolved_gid = self._resolve_confusion(tid, gid)
            if resolved_gid is None:
                return False
            if resolved_gid != gid:
                resolution_log = f" (confusion-resolved: faiss_nn={gid} -> window_winner={resolved_gid})"
            gid = resolved_gid
        elif cos_dist >= self.fast_commit_thresh or (cos_dist2 - cos_dist) < self.min_margin:
            return False

        ev = self._pending.get(tid)
        if ev is not None and gid in ev.cannot_link_gids:
            return False

        entry = self._gallery[gid]
        worn_elsewhere = self._gid_seen_at.get(gid, -1) >= frame_id - 1
        claimed = (entry.active_tid not in (None, tid)
                   and (entry.active_tid in live_tids or worn_elsewhere))
        if claimed:
            return False

        entry.active_tid = tid
        entry.last_cam_source = cam_source
        # NOT entry.add_evidence(...) here - deferred until the solver
        # confirms this pairing (see docstring). Blending now would let this
        # tracklet's own evidence contaminate the very centroid the next
        # solve is supposed to re-check it against.
        self._tid_to_gid[tid] = gid
        track.t_global_id = gid
        track.t_identity_since_frame = frame_id
        ev = self._pending.get(tid)
        if ev is not None:
            ev.provisional_gid = gid
        # NOT popped from self._pending - stays eligible for the next solve.
        # NOT self._gallery_dirty - no centroid changed yet, so the FAISS
        # fast-commit index doesn't need rebuilding for this alone.
        # Stamp _gid_seen_at here too (not just _refresh_identified) - a
        # fast-commit that happens mid-tick must be visible to the
        # co-occurrence veto for any OTHER still-pending tracklet processed
        # later in this same step() call (including this tick's own solve,
        # if solver_interval_frames also elapsed this frame) - otherwise a
        # second lookalike could win the same identity in the same tick
        # because the claim looked "unseen" to the veto.
        self._gid_seen_at[gid] = frame_id
        print(f"[GTA] fast commit (provisional): track={tid} (cam={cam_source}) -> global_id={gid} "
              f"(dist={cos_dist:.3f}, margin={cos_dist2 - cos_dist:.3f}){resolution_log}")
        return True

    def _resolve_confusion(self, tid: int, seed_gid: int) -> Optional[int]:
        """The fast-commit path's FAISS nearest-neighbor query only ever
        sees ONE frame's embedding - fine most of the time, but not when
        that neighbor (seed_gid) is DBSCAN-flagged as confusable with other
        gallery identities (see _refresh_confusion_zones): two suspiciously
        similar identities can trade places from one noisy frame to the
        next. Replaces the old fix (just tightening the accept threshold on
        that same single frame, which made a wrong commit less likely but
        did nothing to actually decide RIGHT) with this tracklet's whole
        moving embedding window (EmbeddingWindow, window_size samples, one
        every window_sample_interval frames - real temporal evidence, not
        one snapshot): every identity in seed_gid's confusion group is
        scored by its AVERAGE multi-model log-likelihood against the whole
        window, and the winner only counts if it beats the runner-up by at
        least confusion_margin.

        Returns the winning gid, or None if the window is too short
        (window_min_samples) or no candidate wins clearly - the caller
        (_try_fast_commit) treats None as "can't safely fast-commit this
        tick", falling through to the pending pool so the batch solver
        (which sees the WHOLE pool with exclusivity, not just this one
        tracklet vs. this one confusion group) decides instead.
        """
        hist = self._embedding_windows.get(tid)
        if hist is None or len(hist.samples) < self.window_min_samples:
            return None

        group = self._confusion_group_of.get(seed_gid, frozenset({seed_gid}))
        scores: dict[int, float] = {}
        for gid in group:
            entry = self._gallery.get(gid)
            if entry is None:
                continue
            sample_lls = [
                multi_model_log_likelihood(sample, entry.centroids, self._stream_cfg)
                for sample in hist.samples
            ]
            sample_lls = [ll for ll in sample_lls if ll is not None]
            if not sample_lls:
                continue
            scores[gid] = sum(sample_lls) / len(sample_lls)

        if not scores:
            return None

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_gid, best_score = ranked[0]
        if len(ranked) > 1 and (best_score - ranked[1][1]) < self.confusion_margin:
            return None
        return best_gid

    # --- interval-triggered solver: the default decision path --------------
    #
    # Two stages per tick, deliberately split because they solve different
    # problems:
    #
    #   Stage 1 (_cluster_pending, GAEC): merges pending tracklets AMONG
    #   THEMSELVES. This is the case the gallery-assignment stage below
    #   structurally cannot handle - two brand-new, temporally-disjoint
    #   tracklets of a person who has no gallery entry yet (nothing to
    #   anchor a Hungarian match against) would otherwise each mint their
    #   own identity independently, one tick apart or even the same tick.
    #   GAEC (borrowed from the previous version of this file) is the right
    #   tool here: general correlation clustering over a small, per-tick-only
    #   pool - not a persistent graph re-clustered forever (that was the old
    #   design's actual bug, not GAEC itself).
    #
    #   Stage 2 (_solve_pending, Hungarian): assigns each resulting CLUSTER
    #   (one merged evidence bundle, one or more member tracklets) to either
    #   an EXISTING gallery identity or a dummy "new identity" column.
    #   Column exclusivity is what prevents two lookalike clusters both
    #   winning the same identity. Unlike stage 1, no temporal-overlap veto
    #   is needed here: two different clusters are, by construction,
    #   different people, so at most one can legitimately win any given
    #   gallery column anyway - the assignment problem already guarantees
    #   that. The only hard veto still required is a cluster's span
    #   conflicting with an identity's CURRENTLY-CLAIMED live occupancy
    #   (the gallery column has no "busy until" state of its own) - reusing
    #   the exact _gid_seen_at/active_tid co-occurrence check GlobalRegistry
    #   already gets right.

    def _cluster_pending(self, pendings: list) -> list:
        """GAEC over the pending pool: repeatedly merge the cluster pair with
        the largest positive SUM of member-pairwise signed edge weights,
        stopping once no remaining pair's aggregate is positive. Temporal
        overlap between any member pair is a hard veto (one body can't be in
        two places), OR'd forward into the merged cluster so it still blocks
        a future merge once the original conflicting pair is several hops
        apart. Returns a list of member groups (list[TrackletEvidence])."""
        n = len(pendings)
        cluster_members: dict[int, list[int]] = {c: [c] for c in range(n)}
        if n <= 1:
            return [[pendings[idx] for idx in idxs] for idxs in cluster_members.values()]

        def pair_key(a, b):
            return (a, b) if a < b else (b, a)

        agg: dict[tuple, float] = {}
        forbidden: dict[tuple, bool] = {}
        for i in range(n):
            for j in range(i + 1, n):
                a, b = pendings[i], pendings[j]
                key = (i, j)
                overlap = a.last_visible >= b.first_visible and b.last_visible >= a.first_visible
                # Two tracklets each provisionally claiming a DIFFERENT
                # identity is a real contradiction (merging them would mean
                # deciding they're the same person while also, separately,
                # already being two different confirmed-pending people) -
                # hard veto, same shape as the temporal-overlap one. Sharing
                # the SAME provisional_gid is fine (e.g. one closed and a
                # later tracklet re-claimed the now-free identity) - that's
                # a legitimate same-person merge.
                provisional_conflict = (
                    a.provisional_gid is not None and b.provisional_gid is not None
                    and a.provisional_gid != b.provisional_gid
                )
                if overlap or provisional_conflict:
                    agg[key] = 0.0
                    forbidden[key] = True
                    continue
                ll = multi_model_log_likelihood(a.feats, b.feats, self._stream_cfg)
                if ll is None:
                    agg[key] = 0.0
                    forbidden[key] = True
                    continue
                early, late = (a, b) if a.last_visible < b.first_visible else (b, a)
                gap_sec = max((late.first_visible - early.last_visible) / self.frame_rate, 0.0)
                time_ll = _gaussian_log_likelihood(gap_sec, self.mu_gap_sec, self.sigma_gap_sec)
                # Zero-crossing point: the same "borderline appearance +
                # best-possible timing" boundary used for the new-identity
                # dummy cost (see _new_identity_cost_feats / _boundary_ll) -
                # two pending tracklets only merge if their combined
                # evidence does BETTER than that boundary.
                shared = [name for name in self._stream_cfg if name in a.feats and name in b.feats]
                agg[key] = (ll + time_ll) - self._boundary_ll(shared)
                forbidden[key] = False

        while True:
            best_key, best_val = None, 0.0
            for key, val in agg.items():
                if val > best_val and not forbidden.get(key, False):
                    best_key, best_val = key, val
            if best_key is None:
                break

            ca, cb = best_key
            cluster_members[ca].extend(cluster_members[cb])
            del cluster_members[cb]

            for ck in list(cluster_members.keys()):
                if ck == ca:
                    continue
                key_a, key_b = pair_key(ca, ck), pair_key(cb, ck)
                val_b = agg.pop(key_b, None)
                forb_b = forbidden.pop(key_b, False)
                if val_b is not None:
                    agg[key_a] = agg.get(key_a, 0.0) + val_b
                    forbidden[key_a] = forbidden.get(key_a, False) or forb_b
            for key in [k for k in agg if cb in k]:
                agg.pop(key, None)
                forbidden.pop(key, None)

        return [[pendings[idx] for idx in idxs] for idxs in cluster_members.values()]

    @staticmethod
    def _merge_cluster(members: list) -> dict:
        """Combine a GAEC cluster's member tracklets into one evidence
        bundle: a hits-weighted average per stream (renormalized to unit
        length), the widest visibility span, and the union of cannot-link
        constraints. cam_source/bbox are taken from the earliest/latest
        member respectively - representative points for the gap/camera-prior
        terms, not evidence themselves."""
        feat_sums: dict = {}
        feat_counts: dict = {}
        cannot_link: set = set()
        for mm in members:
            cannot_link |= mm.cannot_link_gids
            for name, vec in mm.feats.items():
                feat_sums[name] = feat_sums.get(name, 0.0) + vec.astype(np.float64) * mm.hits
                feat_counts[name] = feat_counts.get(name, 0) + mm.hits
        merged_feats = {}
        for name, s in feat_sums.items():
            v = s / max(feat_counts[name], 1)
            n = np.linalg.norm(v)
            merged_feats[name] = (v / n).astype(np.float32) if n > 1e-9 else v.astype(np.float32)

        earliest = min(members, key=lambda mm: mm.first_visible)
        latest = max(members, key=lambda mm: mm.last_visible)
        return {
            "members": members,
            "feats": merged_feats,
            "first_visible": earliest.first_visible,
            "last_visible": latest.last_visible,
            "cam_source": earliest.cam_source,
            "bbox": latest.bbox,
            "cannot_link_gids": cannot_link,
        }

    def _new_identity_cost_feats(self, feats: dict) -> float:
        """tau_new: cost of the dummy 'mint a fresh identity' column, in the
        same units as a real candidate's cost (-combined_log_likelihood) -
        see _boundary_ll. A real gallery candidate must do better than
        'borderline appearance + best-possible timing' to beat this."""
        if self._flat_new_identity_cost is not None:
            return self._flat_new_identity_cost
        return -self._boundary_ll(feats.keys())

    def _solve_pending(self, frame_id: int) -> list:
        if not self._pending:
            return []

        pendings = list(self._pending.values())
        groups = self._cluster_pending(pendings)
        clusters = [self._merge_cluster(group) for group in groups]

        gids = list(self._gallery.keys())
        n, m = len(clusters), len(gids)
        MASKED = 1e6  # >> any real cost

        cost = np.full((n, m + n), MASKED, dtype=np.float64)
        for i, cl in enumerate(clusters):
            cost[i, m + i] = self._new_identity_cost_feats(cl["feats"])

        for i, cl in enumerate(clusters):
            own_tids = {mm.tid for mm in cl["members"]}
            for j, gid in enumerate(gids):
                if gid in cl["cannot_link_gids"]:
                    continue
                entry = self._gallery[gid]
                seen_at = self._gid_seen_at.get(gid)
                if (seen_at is not None and seen_at >= cl["first_visible"]
                        and entry.active_tid is not None and entry.active_tid not in own_tids):
                    continue

                ll = multi_model_log_likelihood(cl["feats"], entry.centroids, self._stream_cfg)
                if ll is None:
                    continue

                gap_sec = max((cl["first_visible"] - entry.last_frame) / self.frame_rate, 0.0)
                time_ll = _gaussian_log_likelihood(gap_sec, self.mu_gap_sec, self.sigma_gap_sec)
                cam_ll = 0.0
                if self.camera_prior is not None and entry.last_cam_source is not None and cl["cam_source"] is not None:
                    cam_ll = self.camera_prior.log_bonus(entry.last_cam_source, cl["cam_source"])
                st_ll = 0.0
                if self.st_prior is not None:
                    st_ll = self.st_prior.log_p(
                        entry.last_cam_source, entry.last_frame / self.frame_rate,
                        cl["cam_source"], cl["first_visible"] / self.frame_rate,
                    )
                cost[i, j] = -(ll + time_ll + cam_ll + st_ll)

        rows, cols = linear_sum_assignment(cost)
        resolved: list = []
        for i, j in zip(rows, cols):
            cl = clusters[i]
            tids = [mm.tid for mm in cl["members"]]
            if j < m and cost[i, j] < MASKED:
                gid = gids[j]
                entry = self._gallery[gid]
                entry.last_cam_source = cl["cam_source"]
                entry.add_evidence(cl["feats"], cl["last_visible"], cl["bbox"], self._stream_cfg)
                verb, cost_str = "solver match", f"cost={cost[i, j]:.3f}"
            else:
                gid = self._register(cl["feats"], cl["last_visible"], cl["bbox"], cam_source=cl["cam_source"])
                entry = self._gallery[gid]
                verb, cost_str = "solver new identity", "cost=n/a"

            active_member = max(
                (mm for mm in cl["members"] if mm.track_ref is not None and not mm.dead),
                key=lambda mm: mm.last_visible, default=None,
            )
            if active_member is not None:
                entry.active_tid = active_member.tid
                self._tid_to_gid[active_member.tid] = gid

            for mm in cl["members"]:
                # Release a stale provisional claim on the OLD gid this
                # member fast-committed to, if the solver landed somewhere
                # else - otherwise that identity stays falsely "claimed" by
                # a track_id that no longer uses it, blocking a legitimate
                # future match to it. Its centroid was never touched by this
                # member (add_evidence was deferred, see _try_fast_commit),
                # so no rollback is needed there.
                was_provisional = mm.provisional_gid is not None
                confirmed = was_provisional and mm.provisional_gid == gid
                corrected = was_provisional and mm.provisional_gid != gid
                if corrected:
                    old_entry = self._gallery.get(mm.provisional_gid)
                    if old_entry is not None and old_entry.active_tid == mm.tid:
                        old_entry.active_tid = None

                if mm.track_ref is not None:
                    mm.track_ref.t_global_id = gid
                    if not confirmed:
                        # A brand-new resolution, or a correction of a wrong
                        # provisional guess - identity age starts now. A
                        # confirm leaves it alone: the identity was actually
                        # right since the original fast-commit frame.
                        mm.track_ref.t_identity_since_frame = frame_id
                    resolved.append(mm.track_ref)

                if corrected:
                    print(f"[GTA] fast-commit corrected: track={mm.tid} "
                          f"provisional_gid={mm.provisional_gid} -> global_id={gid}")

            print(f"[GTA] {verb}: tracks={tids} -> global_id={gid} ({cost_str})")

        # Every member of every cluster is finally resolved (confirmed,
        # corrected, or freshly minted) by this point - its embedding
        # window has done its job (if it was ever needed) and can go.
        for cl in clusters:
            for mm in cl["members"]:
                self._embedding_windows.pop(mm.tid, None)

        self._pending.clear()
        self._gallery_dirty = True
        return resolved

    def _register(self, feats: dict, frame_id: int, bbox: Optional[np.ndarray], cam_source=None) -> int:
        gid = self._next_gid
        self._next_gid += 1
        self._gallery[gid] = GalleryEntry(gid, feats, frame_id, bbox, ema_alpha=self.ema_alpha, cam_source=cam_source)
        return gid

    def _rebuild_fast_index(self) -> None:
        gids = [g for g, e in self._gallery.items() if self._primary_stream in e.centroids]
        if not gids:
            self._fast_index = None
            self._fast_index_gids = []
            return
        dim = self._gallery[gids[0]].centroids[self._primary_stream].shape[-1]
        if self._fast_index is None or self._fast_index_dim != dim:
            cpu_index = faiss.IndexFlatIP(dim)
            idx = cpu_index
            if self._use_gpu:
                try:
                    res = faiss.StandardGpuResources()
                    idx = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                except Exception as exc:
                    print(f"[GTA] GPU FAISS unavailable ({exc}); using CPU index")
            self._fast_index = idx
            self._fast_index_dim = dim
        else:
            self._fast_index.reset()
        centroids = np.ascontiguousarray(
            np.stack([self._gallery[g].centroids[self._primary_stream] for g in gids], axis=0).astype(np.float32)
        )
        self._fast_index.add(centroids)
        self._fast_index_gids = gids

    def _refresh_confusion_zones(self, frame_id: int) -> None:
        """Periodic DBSCAN over gallery primary-stream centroids: purely a
        DETECTOR of which identities are suspiciously close to each other
        (hard-sample groups) - resolving a fast-commit candidate that lands
        in one of these groups is _resolve_confusion's job (moving
        embedding window), not this method's."""
        if frame_id - self._last_confusion_refresh < self.confusion_refresh_interval:
            return
        self._last_confusion_refresh = frame_id
        entries_with_primary = {g: e for g, e in self._gallery.items() if self._primary_stream in e.centroids}
        if len(entries_with_primary) < 2:
            self._confusion_gids = set()
            self._confusion_group_of = {}
            return
        gids = list(entries_with_primary.keys())
        vecs = np.stack([entries_with_primary[g].centroids[self._primary_stream] for g in gids])
        cos_dist = (1.0 - (vecs @ vecs.T)).clip(0.0, 2.0).astype(np.float64)
        labels = DBSCAN(eps=self.confusion_eps, min_samples=2, metric='precomputed').fit_predict(cos_dist)

        groups: dict[int, set] = {}
        for i, lbl in enumerate(labels):
            if lbl >= 0:
                groups.setdefault(int(lbl), set()).add(gids[i])

        prev = self._confusion_gids
        self._confusion_gids = {gid for grp in groups.values() for gid in grp}
        self._confusion_group_of = {gid: frozenset(grp) for grp in groups.values() for gid in grp}
        if self._confusion_gids != prev:
            print(f"[GTA] confusion zones updated: {len(self._confusion_gids)} entries "
                  f"groups={[sorted(g) for g in groups.values()]}")


    def deactivate_track(self, track_id: int) -> None:
        self._tid_cooldown.pop(track_id, None)
        ev = self._pending.get(track_id)
        if ev is not None:
            ev.dead = True
            ev.track_ref = None
        gid = self._tid_to_gid.pop(track_id, None)
        if gid is None:
            return
        entry = self._gallery.get(gid)
        if entry is not None and entry.active_tid == track_id:
            entry.active_tid = None

    def _log_worker(self):
        buffer: list = []
        while not self._log_stop.is_set():
            try:
                item = self._log_queue.get(timeout=0.5)
                buffer.append(item)
                if len(buffer) >= self.log_flush_interval:
                    self._write_buffer(buffer)
                    buffer.clear()
            except queue.Empty:
                if buffer:
                    self._write_buffer(buffer)
                    buffer.clear()
        while True:
            try:
                buffer.append(self._log_queue.get_nowait())
            except queue.Empty:
                break
        if buffer:
            self._write_buffer(buffer)

    def _write_buffer(self, buffer: list):
        if not buffer:
            return
        with open(self.reid_log_path, "a") as f:
            for frame_id, gid_feat_map in buffer:
                line = {"frame": frame_id,
                        "gid": [[int(g), v.tolist()] for g, v in gid_feat_map.items()]}
                f.write(json.dumps(line) + "\n")

    def stop_logging(self):
        if self._log_thread is None:
            return
        self._log_stop.set()
        self._log_thread.join(timeout=15)
        self._log_thread = None

    def get_gallery_entries(self) -> list:
        return list(self._gallery.values())

    def gallery_entry(self, global_id: int) -> Optional[GalleryEntry]:
        """O(1) lookup of one identity's GalleryEntry, or None if global_id
        isn't (or isn't yet) a real gallery identity - e.g. t_global_id==0,
        or a still-provisional fast-commit's gid this GTA instance hasn't
        actually resolved. Used by botsort/dwelling.py to read
        entry.first_frame for cross-camera dwell-time detection."""
        return self._gallery.get(global_id)

    def size(self) -> int:
        return len(self._gallery)

    def __repr__(self):
        active = sum(1 for e in self._gallery.values() if e.active_tid is not None)
        return (f"GTA(identities={len(self._gallery)}, active={active}, "
                f"pending={len(self._pending)}, streams={list(self._stream_cfg.keys())})")
