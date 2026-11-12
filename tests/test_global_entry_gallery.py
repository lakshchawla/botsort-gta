import numpy as np

from association.embedding import EmbeddingTrack, unit
from association.gallery import FaissGallery
from association.global_entry import GlobalEntry

rng = np.random.default_rng(1)


def rv(d=64):
    return unit(rng.normal(size=d))


def entry(gid, vec, key=None, frame=0, cam=0):
    e = EmbeddingTrack(0.8); e.update(vec)
    return GlobalEntry(gid, e, frame, cam, key=key)


def test_observe_gates_and_counts_mismatch():
    v = rv(); e = entry(1, v)
    assert e.observe(unit(v + 0.05 * rv()), 0.3) and e.mismatch_streak == 0
    assert not e.observe(-v, 0.3) and e.mismatch_streak == 1
    assert not e.observe(-v, 0.3) and e.mismatch_streak == 2
    assert e.observe(v, 0.3) and e.mismatch_streak == 0            # a good sample resets the streak
    n = e.embedding.count
    e.observe(-v, 0.3)
    assert e.embedding.count == n                                   # rejected samples never enter the average


def test_membership_and_one_body_one_place():
    e = entry(1, rv(), key=(0, 5), frame=10)
    assert e.worn_by_other((1, 7), frame=12, tol=5)                 # another tracklet, recently seen
    assert not e.worn_by_other((0, 5), frame=12, tol=5)             # the wearer itself
    assert not e.worn_by_other((1, 7), frame=100, tol=5)            # wearer long gone
    e.release((0, 5))
    assert e.active_key is None and not e.worn_by_other((1, 7), 11, 5)
    e.attach((1, 7), 20, cam=1)
    assert e.active_key == (1, 7) and (1, 7) in e.members and (0, 5) in e.members


def test_absorb_moves_entry_towards_tracklet():
    a, b = rv(), rv()
    e = entry(1, a)
    t = EmbeddingTrack(); t.update(b)
    before = e.embedding.distance(b)
    e.absorb(t)
    assert e.embedding.distance(b) < before


def test_gallery_search_order_and_lazy_rebuild():
    g = FaissGallery()
    assert g.search(rv()) == [] and len(g) == 0
    a, b = rv(), rv()
    g.add(entry(1, a)); g.add(entry(2, b))
    hits = g.search(unit(a + 0.02 * rv()), k=2)
    assert [h[0] for h in hits] == [1, 2] and hits[0][1] < hits[1][1]
    assert hits[0][1] < 0.05
    # entry 2 drifts towards a; the index must reflect it after mark_dirty
    for _ in range(30):
        g.entries[2].embedding.update(a)
    g.mark_dirty()
    hits = g.search(a, k=2)
    assert hits[1][1] < 0.1
    assert len(g.search(a, k=5)) == 2


def test_same_camera_lost_wearer_does_not_block_takeover():
    e = entry(1, rv(), key=(0, 5), frame=10, cam=0)
    # wearer (0,5) is still visible in camera 0 -> worn
    assert e.worn_by_other((0, 9), frame=12, tol=5, cam=0, live_keys={(0, 5), (0, 9)})
    # wearer was lost (tracker re-acquired the person as (0,9)) -> free, even though it updated 2 frames ago
    assert not e.worn_by_other((0, 9), frame=12, tol=5, cam=0, live_keys={(0, 9)})
    # another camera: only recency counts
    assert e.worn_by_other((1, 3), frame=12, tol=5, cam=1, live_keys={(1, 3)})
    assert not e.worn_by_other((1, 3), frame=40, tol=5, cam=1, live_keys={(1, 3)})
