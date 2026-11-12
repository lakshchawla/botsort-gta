import numpy as np

from association import Associator, AssociatorConfig, AssocStatus
from association.embedding import EmbeddingTrack, unit
from association.gallery import FaissGallery
from association.global_entry import GlobalEntry
from tracker.tracklet import Tracklet

rng = np.random.default_rng(5)
D = 128


def rv():
    return unit(rng.normal(size=D))


def ortho(p):
    q = rng.normal(size=D)
    return unit(q - float(q @ p) * p)


def at(p, dist):
    """a unit vector at exactly cosine distance `dist` from p"""
    q = ortho(p)
    c = 1.0 - dist
    return unit(c * p + np.sqrt(1 - c * c) * q)


def entry_at(p, **kw):
    e = EmbeddingTrack(0.8); e.update(p)
    return GlobalEntry(1, e, 0, 0, key=(0, 1), **kw)


# --------------------------------------------------------------------------- the entry's state machine
def test_normal_samples_are_ok_and_do_not_touch_the_quarantine():
    p = rv(); e = entry_at(p)
    for _ in range(5):
        assert e.view_step(at(p, 0.02), 0.06, 0.048, 4) == "ok"
    assert len(e.tmp_varied_views) == 0 and len(e.varied_views) == 0 and e.suspect_run == 0


def test_settling_back_turns_the_quarantine_into_a_varied_view():
    p = rv(); e = entry_at(p)
    a, b = at(p, 0.09), at(p, 0.09)
    assert e.view_step(a, 0.06, 0.048, 4) == "suspect" and len(e.tmp_varied_views) == 1
    assert e.view_step(b, 0.06, 0.048, 4) == "suspect" and e.suspect_run == 2
    assert e.view_step(at(p, 0.02), 0.06, 0.048, 4) == "settled"        # back near the identity
    assert len(e.tmp_varied_views) == 0 and e.suspect_run == 0
    assert len(e.varied_views) >= 1                                      # the quarantined looks were kept
    assert min(1 - float(v @ a) for v in e.varied_views) < 1e-5


def test_staying_high_is_a_shift_and_clears_the_quarantine():
    p = rv(); e = entry_at(p)
    q = [at(p, 0.12) for _ in range(5)]
    assert [e.view_step(x, 0.06, 0.048, 4) for x in q[:3]] == ["suspect"] * 3
    assert len(e.tmp_varied_views) == 3                                  # the quarantine never holds more than 3
    assert e.view_step(q[3], 0.06, 0.048, 4) == "shift"
    assert len(e.last_shift_samples) == 3 and len(e.tmp_varied_views) == 0 and e.suspect_run == 0
    assert len(e.varied_views) == 0                                      # a shift never becomes a view


def test_a_known_view_is_not_suspicious_any_more_and_views_are_capped():
    p = rv(); e = entry_at(p, views_size=8)
    look = at(p, 0.09)
    e.add_views([look])
    assert e.anchor_distance(look) < 1e-5 and e.view_step(look, 0.06, 0.048, 4) == "ok"
    for _ in range(20):
        e.add_views([at(p, 0.10)], dedup_dist=0.0)
    assert len(e.varied_views) == 8
    dup = len(e.varied_views)
    e.add_views([e.varied_views[-1]], dedup_dist=0.02)                    # near-duplicate: skipped
    assert len(e.varied_views) == dup


def test_changing_the_wearer_clears_the_quarantine_but_keeps_the_views():
    p = rv(); e = entry_at(p)
    e.add_views([at(p, 0.09)])
    e.view_step(at(p, 0.12), 0.06, 0.048, 4)
    assert e.suspect_run == 1
    e.attach((1, 9), 10, 1)
    assert e.suspect_run == 0 and len(e.tmp_varied_views) == 0 and len(e.varied_views) == 1


def test_gallery_can_search_the_varied_views():
    p = rv(); e = entry_at(p); look = at(p, 0.20)
    e.add_views([look])
    g_plain, g_views = FaissGallery(views_in_search=False), FaissGallery(views_in_search=True)
    for g in (g_plain, g_views):
        g.add(e)
    assert g_plain.search(look)[0][1] > 0.15
    gid, d = g_views.search(look)[0]
    assert gid == 1 and d < 1e-5
    assert len(g_views.search(look, k=3)) == 1                           # one hit per entry, not per row


# --------------------------------------------------------------------------- end to end in the associator
def mk(cam, tid, vec, n=5):
    t = Tracklet(cam, tid, 0, 0, EmbeddingTrack(0.8))
    for _ in range(n):
        t.embedding.update(vec)
    t.clean, t.tracklet_len, t.peer_iou = True, 100, 0.0
    return t


def feed(a, t, vec, frame, cam=0):
    """one new clean sample for the tracklet, then one association step"""
    t.embedding.update(vec)
    t.new_embedding_this_frame, t.last_frame = True, frame
    return a.step(cam, [t], frame)


def make(**kw):
    kw.setdefault("views_enabled", True)
    return Associator(AssociatorConfig(**kw))


def kinds(a):
    return [e["event"] for e in a.events]


def test_pose_change_that_settles_back_adds_a_view_and_keeps_the_identity():
    a = make(); p = rv()
    t = mk(0, 1, p); feed(a, t, p, 1)
    assert t.t_global_id == 1
    entry = a.entry(1); ema_before = entry.embedding.value.copy()
    f = 2
    for _ in range(2):                                        # two frames in a different pose
        feed(a, t, at(p, 0.09), f); f += 1
    assert np.allclose(entry.embedding.value, ema_before) and len(entry.tmp_varied_views) == 2   # entry not updated by them
    feed(a, t, at(p, 0.02), f)                                # settles
    assert "view_added" in kinds(a) and len(entry.varied_views) >= 1
    assert t.t_global_id == 1 and t.assoc_status is AssocStatus.COMMITTED and "shift" not in kinds(a)


def test_persistent_shift_gives_the_id_up_and_matches_the_other_known_identity():
    a = make(); A, B = rv(), rv()
    ta = mk(0, 1, A); feed(a, ta, A, 1)                       # identity 1 = A
    tb = mk(1, 2, B); feed(a, tb, B, 1, cam=1); a.release(tb.key)    # identity 2 = B (its tracklet is gone)
    assert (ta.t_global_id, a.size()) == (1, 2)
    entry1 = a.entry(1); ema_before = entry1.embedding.value.copy()
    f = 2
    for _ in range(3):                                        # the box now shows B, three samples
        feed(a, ta, at(B, 0.01), f); f += 1
    assert ta.t_global_id == 1                                # not yet
    feed(a, ta, at(B, 0.01), f)                               # 4th high sample: the id moved
    assert ta.t_global_id == 2, kinds(a)
    ev = next(e for e in a.events if e["event"] == "shift")
    assert (ev["global_id"], ev["to_global_id"], ev["back_to_same"]) == (1, 2, False)
    assert np.allclose(entry1.embedding.value, ema_before) and entry1.active_key is None   # identity 1 stayed clean
    assert a.entry(2).active_key == ta.key and "view_added" not in kinds(a)


def test_persistent_shift_to_an_unknown_person_ends_up_with_a_new_identity():
    a = make(); A, B = rv(), rv()
    t = mk(0, 1, A); feed(a, t, A, 1)
    f = 2
    for _ in range(4):
        feed(a, t, at(B, 0.01), f); f += 1
    assert t.t_global_id == 0 and t.assoc_status is AssocStatus.PENDING     # only 3 samples exist: not mature enough to mint
    while t.t_global_id == 0 and f < 12:
        feed(a, t, at(B, 0.01), f); f += 1
    assert t.t_global_id == 2 and a.size() == 2 and "new_identity" in kinds(a)
    assert a.entry(1).active_key is None


def test_persistent_new_look_that_still_matches_the_same_identity_becomes_a_view():
    a = make(); p = rv()
    t = mk(0, 1, p); feed(a, t, p, 1)
    look = at(p, 0.09)                                        # far enough to be suspect, near enough to still be this identity
    f = 2
    for _ in range(4):
        feed(a, t, at(look, 0.005), f); f += 1
    ev = next(e for e in a.events if e["event"] == "shift")
    assert ev["back_to_same"] is True and t.t_global_id == 1
    assert len(a.entry(1).varied_views) >= 1 and a.entry(1).anchor_distance(look) < 0.02


def test_views_off_means_the_old_behaviour():
    a = make(views_enabled=False); p = rv()
    t = mk(0, 1, p); feed(a, t, p, 1)
    for f in range(2, 8):
        feed(a, t, at(p, 0.09), f)
    assert "shift" not in kinds(a) and "view_added" not in kinds(a) and len(a.entry(1).varied_views) == 0
