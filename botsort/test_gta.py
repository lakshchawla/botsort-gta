"""
Standalone regression checks for botsort/global_tracklet_association.py's
unified, tracklet-level, multi-model cross-camera ReID engine.

Not pytest-based (matches this repo's standalone test style, see
botsort/test_interval_solver.py) - run: `python3 -m botsort.test_gta`.

Drives GTA.step()/deactivate_track() directly with duck-typed fake
tracks/trackers (only the attributes step() actually reads) so tracklet
timing and feature similarity are exact and deterministic - what's under
test is purely GTA's decision policy, not Kalman/IOU association noise.
"""

from __future__ import annotations

import numpy as np

from .global_tracklet_association import GTA, FeatureStreamConfig, multi_model_log_likelihood


def _unit(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


class FakeTrack:
    def __init__(self, tid, feats, tracklet_len=100, bbox=(0, 0, 50, 150)):
        self.track_id = tid
        if not isinstance(feats, dict):
            feats = {"primary": feats}
        self.smooth_feats = {name: _unit(np.asarray(v, dtype=np.float32)) for name, v in feats.items()}
        self.t_global_id = 0
        self.t_identity_since_frame = 0
        self.is_touching_edge = False
        self.tracklet_len = tracklet_len
        self._tlwh = np.asarray(bbox, dtype=np.float64)

    @property
    def tlwh(self):
        return self._tlwh.copy()

    @property
    def tlbr(self):
        ret = self._tlwh.copy()
        ret[2:] += ret[:2]
        return ret


class FakeTracker:
    def __init__(self, cam_source, tracks=()):
        self.cam_source = cam_source
        self.tracked_stracks = list(tracks)


def _make_gta(**kwargs):
    defaults = dict(
        min_tracklet_len=1, solver_interval_frames=50,
        mu_gap_sec=2.0, sigma_gap_sec=20.0, frame_rate=30.0,
    )
    defaults.update(kwargs)
    return GTA(**defaults)


def test_reappearance_links_via_persistent_gallery() -> bool:
    """Tracklet A (cam1, frames 1-30) closes; tracklet B (cam2, first seen at
    frame 90 - a 2.0s gap at 30fps, exactly at mu_gap_sec) has a near-identical
    feature. Neither has a gallery entry yet, so this exercises the
    pending-vs-pending GAEC clustering stage (_cluster_pending) feeding the
    gallery-assignment stage (_solve_pending) - the whole point of
    tracklet-level GTA."""
    rng = np.random.RandomState(0)
    gta = _make_gta()
    alice = _unit(rng.randn(128))

    track_a = FakeTrack(101, alice)
    cam1 = FakeTracker(1, [track_a])
    gta.step(cam1, 1)
    gta.step(cam1, 30)
    gta.deactivate_track(101)

    track_b = FakeTrack(201, _unit(alice + rng.randn(128).astype(np.float32) * 0.01))
    cam2 = FakeTracker(2, [track_b])
    gta.step(cam2, 90)          # 90 - 0 >= solver_interval_frames(50) -> tick fires here

    ok = track_b.t_global_id != 0 and gta.size() == 1
    print(f"[reappearance_links] gid={track_b.t_global_id} identities={gta.size()} "
          f"(expect 1) -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_overlapping_tracklets_never_merge() -> bool:
    """Tracklet C (cam1) and tracklet D (cam2) overlap in time - identical
    features must NOT be enough to merge them, since one body can't be in
    two places (two cameras) at once. The hard temporal-overlap veto in
    _cluster_pending must win over a perfect appearance score."""
    rng = np.random.RandomState(1)
    gta = _make_gta(solver_interval_frames=20)
    alice = _unit(rng.randn(128))

    track_c = FakeTrack(102, alice, bbox=(0, 0, 50, 150))
    track_d = FakeTrack(202, alice.copy(), bbox=(500, 500, 50, 150))
    cam = FakeTracker(1, [track_c, track_d])
    gta.step(cam, 1)
    gta.step(cam, 25)          # 25 >= solver_interval_frames(20) -> tick fires

    ok = (track_c.t_global_id != 0 and track_d.t_global_id != 0
          and track_c.t_global_id != track_d.t_global_id and gta.size() == 2)
    print(f"[overlapping_tracklets_never_merge] c={track_c.t_global_id} "
          f"d={track_d.t_global_id} identities={gta.size()} (expect 2, distinct) -> "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def test_reappearance_after_long_gap_still_links() -> bool:
    """The headline regression this rewrite exists to fix: the previous
    version of this file pruned a tracklet node after `prune_after_frames`
    (default 5x window_frames) and could never re-link it - a real CHIRLA
    investigation measured this as the dominant cause of 2-3x
    over-fragmentation (see GTA_ID_SHIFT_ANALYSIS.txt). The new engine's
    gallery never expires: a reappearance thousands of frames after the
    identity was last seen must still link to it."""
    rng = np.random.RandomState(2)
    gta = _make_gta(sigma_gap_sec=200.0)   # loose enough to not itself penalize a huge gap
    alice = _unit(rng.randn(128))

    track_e = FakeTrack(103, alice)
    cam1 = FakeTracker(1, [track_e])
    gta.step(cam1, 1)
    gta.step(cam1, 10)
    gta.deactivate_track(103)
    gta.step(FakeTracker(1, []), 60)   # tick fires, mints the identity

    minted_ok = gta.size() == 1
    old_gid = next(iter(gta._gallery.keys())) if gta._gallery else None

    # Old prune_after_frames default would have been 5 * window_frames - here
    # window_frames(=solver_interval_frames)=50, so 5x = 250. Reappear WAY
    # past that (frame 5000, a ~4900-frame / 163s gap) - the old design
    # could never link this; the persistent gallery must.
    track_f = FakeTrack(301, alice.copy())
    cam2 = FakeTracker(2, [track_f])
    gta.step(cam2, 5000)

    ok = (minted_ok and old_gid is not None
          and track_f.t_global_id == old_gid and gta.size() == 1)
    print(f"[reappearance_after_long_gap] minted={minted_ok} old_gid={old_gid} "
          f"reappeared_gid={track_f.t_global_id} identities={gta.size()} (expect 1, same id) -> "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def test_exclusivity_prevents_lookalike_id_theft() -> bool:
    """Two lookalikes (both close to an existing identity) appear at the same
    time on a different camera. Hungarian column exclusivity in
    _solve_pending must let AT MOST ONE of them win the existing identity -
    this is the mechanism that replaces the old per-frame margin gate
    entirely (see module docstring)."""
    rng = np.random.RandomState(4)
    gta = _make_gta(fast_commit_thresh=-1.0)   # force everything through the solver
    dave = _unit(rng.randn(128))

    track_a = FakeTrack(121, dave)
    cam1 = FakeTracker(1, [track_a])
    gta.step(cam1, 1)
    gta.step(cam1, 20)
    gta.deactivate_track(121)
    gta.step(FakeTracker(1, []), 60)
    gid_dave = next(iter(gta._gallery.keys()))

    look1 = _unit(dave + rng.randn(128).astype(np.float32) * 0.01)
    look2 = _unit(dave + rng.randn(128).astype(np.float32) * 0.01)
    track_b = FakeTrack(221, look1, bbox=(0, 0, 50, 150))
    track_c = FakeTrack(222, look2, bbox=(500, 500, 50, 150))
    cam2 = FakeTracker(2, [track_b, track_c])
    gta.step(cam2, 120)
    gta.step(cam2, 200)         # 200 - 60 >= solver_interval_frames(50) -> tick fires

    winners = [track_b.t_global_id == gid_dave, track_c.t_global_id == gid_dave]
    ok = (sum(winners) == 1 and track_b.t_global_id != track_c.t_global_id
          and track_b.t_global_id != 0 and track_c.t_global_id != 0)
    print(f"[exclusivity_prevents_id_theft] gid_dave={gid_dave} "
          f"b={track_b.t_global_id} c={track_c.t_global_id} winners={winners} "
          f"(expect exactly one True) -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_gaec_rejects_chain_merge_in_pending_pool() -> bool:
    """Three temporally disjoint, never-before-registered tracklets A, B, C:
    A-B and B-C each score decently on their own, but A-C is a near-
    orthogonal (clearly different person) appearance match. A single-edge
    greedy walk would chain all three together via B without ever
    re-consulting A-C. GAEC's cluster-aggregate criterion (_cluster_pending)
    catches this: once A and B are one cluster, merging in C is scored on
    sum(weight(A,C), weight(B,C)) - the strongly negative A-C term drags
    that sum negative, so the merge is correctly refused and C stays its own
    identity."""
    rng = np.random.RandomState(5)
    gta = _make_gta()

    v = _unit(rng.randn(128))
    w = _unit(rng.randn(128))
    w = _unit(w - float(np.dot(w, v)) * v)   # orthogonalize: cos_dist(v, w) = 1.0

    track_a = FakeTrack(131, v)
    cam1 = FakeTracker(1, [track_a])
    gta.step(cam1, 1)
    gta.step(cam1, 10)

    track_b = FakeTrack(231, _unit(v + w))   # "between" v and w - moderately close to both
    cam2 = FakeTracker(2, [track_b])
    gta.step(cam2, 70)                       # visible from 70; gap from A = 2.0s

    track_c = FakeTrack(132, w)               # far from A (orthogonal), moderately close to B
    cam1c = FakeTracker(1, [track_c])
    gta.step(cam1c, 150)                      # visible from 150; gap from B = 2.33s
    gta.step(cam1c, 200)                      # 200 - 70 >= solver_interval_frames(50) -> tick fires

    ids = {track_a.t_global_id, track_b.t_global_id, track_c.t_global_id}
    ok = 0 not in ids and len(ids) == 2 and gta.size() == 2
    print(f"[gaec_rejects_chain_merge] a={track_a.t_global_id} b={track_b.t_global_id} "
          f"c={track_c.t_global_id} identities={gta.size()} (expect 2 distinct ids, not "
          f"all 3 chained into 1) -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_dual_model_fusion_vetoes_single_stream_collision() -> bool:
    """A synthetic case proving the weighted multi-model combination
    actually uses BOTH signals rather than one dominating: person B's
    "reidnet" embedding is a near-duplicate of person A's (a collision on
    that model alone), but their "clipreid" embedding is unrelated - a
    genuinely different person. Forcing the decision through the solver
    (fast_commit_thresh disabled, since the fast path deliberately only
    consults the primary stream) must NOT match B to A."""
    streams = [
        FeatureStreamConfig(name="reidnet", weight=1.0, sigma_feat=0.15, outlier_reject_thresh=0.30, is_primary=True),
        FeatureStreamConfig(name="clipreid", weight=1.0, sigma_feat=0.15, outlier_reject_thresh=0.30),
    ]
    rng = np.random.RandomState(7)
    gta = _make_gta(streams=streams, fast_commit_thresh=-1.0)

    a_reidnet = _unit(rng.randn(256))
    a_clipreid = _unit(rng.randn(1280))
    track_a = FakeTrack(1, {"reidnet": a_reidnet, "clipreid": a_clipreid})
    cam1 = FakeTracker(1, [track_a])
    gta.step(cam1, 1)
    gta.step(cam1, 20)
    gta.deactivate_track(1)
    gta.step(FakeTracker(1, []), 60)
    gid_a = next(iter(gta._gallery.keys()))

    b_reidnet = _unit(a_reidnet + rng.randn(256).astype(np.float32) * 0.01)   # near-duplicate
    b_clipreid = _unit(rng.randn(1280))                                       # unrelated
    track_b = FakeTrack(2, {"reidnet": b_reidnet, "clipreid": b_clipreid})
    cam2 = FakeTracker(2, [track_b])
    gta.step(cam2, 120)
    gta.step(cam2, 200)

    ok = track_b.t_global_id != 0 and track_b.t_global_id != gid_a
    print(f"[dual_model_fusion_vetoes_collision] gid_a={gid_a} "
          f"b_resolved={track_b.t_global_id} (expect different id - clipreid vetoes "
          f"the reidnet-only collision) -> {'PASS' if ok else 'FAIL'}")

    # Sanity check on the raw formula too - a genuine same-person case
    # (both streams agree) must score strictly better than the collision case.
    stream_cfg = {s.name: s for s in streams}
    b_agree_clipreid = _unit(a_clipreid + rng.randn(1280).astype(np.float32) * 0.01)
    ll_agree = multi_model_log_likelihood(
        {"reidnet": a_reidnet, "clipreid": a_clipreid},
        {"reidnet": b_reidnet, "clipreid": b_agree_clipreid}, stream_cfg,
    )
    ll_collide = multi_model_log_likelihood(
        {"reidnet": a_reidnet, "clipreid": a_clipreid},
        {"reidnet": b_reidnet, "clipreid": b_clipreid}, stream_cfg,
    )
    formula_ok = ll_agree > ll_collide
    print(f"[dual_model_fusion_formula] ll_agree={ll_agree:.2f} ll_collide={ll_collide:.2f} "
          f"-> {'PASS' if formula_ok else 'FAIL'}")
    return ok and formula_ok


def test_identity_revoke_on_sustained_mismatch() -> bool:
    """Once a track carries a global_id, GTA trusts it and just refreshes the
    gallery centroid - it never re-queries. If BoT-SORT's own association
    silently hands that track_id to a different person (occlusion/crossing
    paths), sustained appearance mismatch must eventually revoke the
    identity (GalleryEntry.mismatch_streak / identity_revoke_streak), not
    reinforce a wrong label forever.

    Seeds the gallery directly (white-box) rather than via a full
    mint-then-deactivate dance - deactivate_track() deliberately clears a
    closed tracklet's track_ref (nothing should still write into a discarded
    STrack), so this test instead sets up the "track_id already carries a
    resolved identity" precondition (_refresh_identified's entry point)
    directly, exactly as bot_sort.py's real STrack objects reach it every
    frame while still live in tracker.tracked_stracks."""
    rng = np.random.RandomState(8)
    gta = _make_gta(identity_revoke_streak=3)
    alice = _unit(rng.randn(128))
    bob = _unit(rng.randn(128))

    gid = gta._register({"primary": alice}, frame_id=0, bbox=None, cam_source=1)
    minted_ok = gid in gta._gallery

    # Same track_id (as BoT-SORT would keep it across a silent mis-continuation),
    # but the appearance is now confidently a different person.
    track = FakeTrack(500, bob)
    track.t_global_id = gid   # still carries the old identity, as BoT-SORT would leave it
    for frame in (1, 2, 3):
        gta.step(FakeTracker(1, [track]), frame)

    ok = minted_ok and track.t_global_id == 0
    print(f"[identity_revoke_on_sustained_mismatch] minted={minted_ok} gid_before={gid} "
          f"gid_after={track.t_global_id} (expect 0, revoked) -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_fast_commit_false_positive_is_corrected_by_next_solve() -> bool:
    """The fast-commit path only ever consults FAISS top-2 + margin on the
    primary stream - with a single gallery entry, margin is trivially huge
    (cos_dist2 defaults to 1.0), so a mere lookalike can clear it. Without a
    probation window this would be a one-way door: fast-commit pops the
    tracklet from the pending pool and blends its evidence into the
    centroid immediately, so the periodic Hungarian solver never gets a
    chance to reconsider it, and only a sustained (identity_revoke_streak)
    mismatch could ever undo it.

    TrackletEvidence.provisional_gid fixes this: a fast-committed tracklet
    stays in the pending pool (and its evidence blend into the gallery
    centroid is deferred) until the next solve re-checks it against the
    WHOLE pool with full exclusivity. Here, a lookalike (track X) false-
    positives onto an existing identity's gid; the real owner (track Y)
    then reappears and is genuinely closer to that identity's (still-
    uncontaminated) centroid. The next solve must reassign the gid to Y and
    correct X away from it - proving the fast path is no longer a one-way
    door."""
    rng = np.random.RandomState(42)
    gta = _make_gta()

    p = _unit(rng.randn(128))                                        # person P's true appearance
    x_lookalike = _unit(p + rng.randn(128).astype(np.float32) * 0.05)  # a DIFFERENT person, moderately similar
    y_true = _unit(p + rng.randn(128).astype(np.float32) * 0.01)       # P actually returning - genuinely closer

    track_a = FakeTrack(1, p)
    cam1 = FakeTracker(1, [track_a])
    gta.step(cam1, 1)
    gta.step(cam1, 20)
    gta.deactivate_track(1)
    gta.step(FakeTracker(1, []), 60)   # solve fires, mints gid from A's evidence
    gid_g = next(iter(gta._gallery.keys()))

    # Lookalike false-positives onto gid_g via the fast-commit path (only
    # one gallery entry -> margin trivially clears).
    track_x = FakeTrack(2, x_lookalike, bbox=(0, 0, 50, 150))
    gta.step(FakeTracker(2, [track_x]), 65)
    false_positive_ok = track_x.t_global_id == gid_g and gta._pending[2].provisional_gid == gid_g

    # The true owner reappears, genuinely closer to gid_g's (still
    # uncontaminated - blend was deferred) original centroid.
    track_y = FakeTrack(3, y_true, bbox=(500, 500, 50, 150))
    gta.step(FakeTracker(2, [track_x, track_y]), 70)
    gta.step(FakeTracker(2, [track_x, track_y]), 120)   # forces the next solve

    ok = (false_positive_ok and track_y.t_global_id == gid_g
          and track_x.t_global_id != gid_g and track_x.t_global_id != 0)
    print(f"[fast_commit_false_positive_corrected] false_positive={false_positive_ok} "
          f"gid_g={gid_g} x_after={track_x.t_global_id} y_after={track_y.t_global_id} "
          f"(expect y==gid_g, x!=gid_g) -> {'PASS' if ok else 'FAIL'}")
    return ok


def run_all() -> bool:
    results = [
        test_reappearance_links_via_persistent_gallery(),
        test_overlapping_tracklets_never_merge(),
        test_reappearance_after_long_gap_still_links(),
        test_exclusivity_prevents_lookalike_id_theft(),
        test_gaec_rejects_chain_merge_in_pending_pool(),
        test_dual_model_fusion_vetoes_single_stream_collision(),
        test_identity_revoke_on_sustained_mismatch(),
        test_fast_commit_false_positive_is_corrected_by_next_solve(),
    ]
    passed = sum(results)
    print(f"\n{passed}/{len(results)} passed")
    return all(results)


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_all() else 1)
