import numpy as np

from association import Associator, AssociatorConfig, AssocStatus
from association.embedding import EmbeddingTrack, unit
from tracker.tracklet import Tracklet

rng = np.random.default_rng(2)


def rv(d=128):
    return unit(rng.normal(size=d))


def near(v, eps=0.03):
    return unit(v + eps * rng.normal(size=v.shape[0]))


def mk(cam, tid, vec, n=5, clean=True, length=100, peer_iou=0.0):
    t = Tracklet(cam, tid, 0, 0, EmbeddingTrack(0.8))
    for _ in range(n):
        t.embedding.update(near(vec, 0.01))
    t.clean, t.tracklet_len, t.peer_iou = clean, length, peer_iou
    return t


def step(a, cam, tracks, frame, fresh=True):
    for t in tracks:
        t.new_embedding_this_frame = fresh
        t.last_frame = frame
    return a.step(cam, list(tracks), frame)


def make(**kw):
    return Associator(AssociatorConfig(**kw))


def kinds(a):
    return [e["event"] for e in a.events]


def test_new_tracklet_is_marked_pending_when_not_eligible():
    a = make()
    cases = {"unclean": mk(0, 1, rv(), clean=False), "short": mk(0, 2, rv(), length=3),
             "few_samples": mk(0, 3, rv(), n=1)}
    for t in cases.values():
        assert t.assoc_status is None
    step(a, 0, list(cases.values()), 1)
    for name, t in cases.items():
        assert t.assoc_status is AssocStatus.PENDING and t.t_global_id == 0, name
    assert a.size() == 0 and a.n_pending() == 3


def test_pending_becomes_committed_by_minting_once_mature():
    a = make(min_embeddings=2, mint_min_embeddings=4)
    t = mk(0, 1, rv(), n=2)                          # eligible for fast-commit, not mature enough to mint
    step(a, 0, [t], 1)
    assert t.assoc_status is AssocStatus.PENDING and a.size() == 0
    for _ in range(2):
        t.embedding.update(near(t.embedding.value, 0.01))
    out = step(a, 0, [t], 2)
    assert out == [t] and t.assoc_status is AssocStatus.COMMITTED and t.t_global_id == 1
    assert a.size() == 1 and kinds(a) == ["new_identity"]
    assert a.entry(1).active_key == t.key


def test_no_decision_without_fresh_evidence():
    a = make()
    t = mk(0, 1, rv())
    step(a, 0, [t], 1, fresh=False)
    assert t.assoc_status is AssocStatus.PENDING and a.size() == 0


def test_second_camera_tracklet_fast_commits_to_existing_entry():
    a = make()
    p = rv()
    t1 = mk(0, 1, p); step(a, 0, [t1], 1)
    assert t1.t_global_id == 1
    a.release(t1.key)                                # t1 left the scene
    t2 = mk(1, 9, p); step(a, 1, [t2], 50)
    assert t2.t_global_id == 1 and a.size() == 1
    assert kinds(a) == ["new_identity", "fast_commit"]
    assert a.entry(1).active_key == t2.key and {t1.key, t2.key} <= set(a.entry(1).members)


def test_lookalike_at_the_same_time_cannot_steal_a_worn_identity():
    a = make()
    p = rv()
    t1 = mk(0, 1, p); step(a, 0, [t1], 10)
    t2 = mk(1, 2, near(p, 0.01)); step(a, 1, [t2], 11)       # same instant, other camera
    assert t1.t_global_id == 1 and t2.t_global_id == 2 and a.size() == 2
    assert a.events[-1]["reason"] == "worn_elsewhere"


def test_far_embedding_mints_new_identity():
    a = make()
    t1 = mk(0, 1, rv()); step(a, 0, [t1], 1); a.release(t1.key)
    t2 = mk(1, 2, rv()); step(a, 1, [t2], 100)
    assert t2.t_global_id == 2 and a.events[-1]["reason"] == "too_far"


def _two_close_entries(a, p, q):
    from association.global_entry import GlobalEntry
    for gid, v in ((1, p), (2, q)):
        e = EmbeddingTrack(0.8); e.update(v)
        a.gallery.add(GlobalEntry(gid, e, 0, 0))
    a._next_gid = 3


def test_ambiguous_never_mints_waits_then_commits_to_nearest():
    a = make(fast_commit_thresh=0.5, min_margin=0.07, ambiguous_patience_frames=30)
    p = rv(); q = near(p, 0.04)                               # two very close identities in the gallery
    _two_close_entries(a, p, q)
    mid = mk(5, 5, unit(p + 0.6 * q))                         # between them, margin < min_margin
    step(a, 5, [mid], 100)
    assert mid.t_global_id == 0 and a.size() == 2             # waits; does NOT mint a third entry
    step(a, 5, [mid], 120)
    assert mid.t_global_id == 0 and a.size() == 2
    step(a, 5, [mid], 131)                                    # patience (30 frames) over
    assert mid.t_global_id in (1, 2) and a.size() == 2        # nearest entry wins, still no new entry
    assert a.events[-1]["event"] == "fast_commit" and a.events[-1]["forced"] is True


def test_ambiguous_resolves_when_margin_opens_up():
    a = make(fast_commit_thresh=0.5, min_margin=0.07, ambiguous_patience_frames=1000)
    p = rv(); q = near(p, 0.04)
    _two_close_entries(a, p, q)
    t = mk(5, 5, unit(p + 0.6 * q))
    step(a, 5, [t], 100)
    assert t.t_global_id == 0
    t.embedding = EmbeddingTrack(0.8)                          # evidence firms up towards p
    for _ in range(6):
        t.embedding.update(near(p, 0.002))
    step(a, 5, [t], 101)
    assert t.t_global_id == 1 and a.events[-1]["forced"] is False


def test_sustained_mismatch_revokes_and_returns_to_pending():
    a = make(revoke_streak=3)
    p, other = rv(), rv()
    t = mk(0, 1, p); step(a, 0, [t], 1)
    assert t.assoc_status is AssocStatus.COMMITTED
    for f in (2, 3):
        t.embedding.update(other)                             # tracker silently switched person
        step(a, 0, [t], f)
        assert t.t_global_id == 1                             # not yet
    t.embedding.update(other)
    step(a, 0, [t], 4)
    assert t.t_global_id == 0 and t.assoc_status is AssocStatus.PENDING
    assert "revoked" in kinds(a) and a.entry(1).active_key is None


def test_committed_tracklet_keeps_updating_entry_with_clean_samples_only():
    a = make()
    p = rv()
    t = mk(0, 1, p); step(a, 0, [t], 1)
    n = a.entry(1).embedding.count
    t.embedding.update(near(p, 0.02)); t.clean = False
    step(a, 0, [t], 2)
    assert a.entry(1).embedding.count == n                    # unclean sample ignored
    t.clean = True
    t.embedding.update(near(p, 0.02))
    step(a, 0, [t], 3)
    assert a.entry(1).embedding.count == n + 1


def test_overlap_defers_decision_until_cooldown_passes():
    a = make(overlap_cooldown_frames=3)
    t = mk(0, 1, rv(), peer_iou=0.5, clean=False)             # occluded
    step(a, 0, [t], 1)
    assert t.assoc_status is AssocStatus.PENDING
    t.peer_iou, t.clean = 0.0, True                           # separated
    for f in (2, 3, 4):                                       # cooldown frames
        step(a, 0, [t], f); assert t.t_global_id == 0
    step(a, 0, [t], 5)
    assert t.t_global_id == 1


def test_release_frees_wearer_but_keeps_entry():
    a = make()
    t = mk(0, 1, rv()); step(a, 0, [t], 1)
    a.release(t.key)
    assert a.size() == 1 and a.entry(1).active_key is None and a.n_pending() == 0


def test_retracked_person_in_same_camera_reuses_identity():
    """The tracker drops a track and re-acquires the person as a new id a few frames later: same person,
    must not mint a duplicate."""
    a = make()
    p = rv()
    old = mk(0, 1, p); step(a, 0, [old], 100)
    assert old.t_global_id == 1
    new = mk(0, 2, near(p, 0.01)); step(a, 0, [new], 104)       # old is lost: not in this frame's live list
    assert new.t_global_id == 1 and a.size() == 1


def test_two_visible_people_in_one_camera_stay_distinct():
    a = make()
    p = rv()
    t1 = mk(0, 1, p); step(a, 0, [t1], 100)
    t2 = mk(0, 2, near(p, 0.01))
    step(a, 0, [t1, t2], 101)                                   # both visible at once
    assert t1.t_global_id == 1 and t2.t_global_id == 2
