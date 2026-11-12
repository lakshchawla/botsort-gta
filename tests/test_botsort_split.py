import os
import numpy as np
import pytest

from tracker.botsort_backend import BotSortCore
from tracker.shift_guard import ShiftGuard

REPO = os.environ.get("BOTSORT_REPO", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "third_party", "BoT-SORT"))
rng = np.random.default_rng(11)


def unit(v):
    return v / np.linalg.norm(v)


def rv(d=64):
    return unit(rng.normal(size=d).astype(np.float32))


def at(p, dist):
    q = rng.normal(size=p.shape[0]).astype(np.float32)
    q = unit(q - float(q @ p) * p)
    c = 1.0 - dist
    return unit(c * p + np.sqrt(1 - c * c) * q)


def box(x=800, y=300, w=90, h=260):
    return [x, y, x + w, y + h]


def make(**kw):
    p = dict(track_high_thresh=0.6, track_low_thresh=0.1, new_track_thresh=0.6, track_buffer=30, match_thresh=0.8,
             proximity_thresh=0.5, appearance_thresh=0.055, split_enabled=True, split_dist=0.10, split_confirm=3)
    p.update(kw)
    return BotSortCore(REPO, p)


def run(core, cam, frames):
    ids = []
    for b, f in frames:
        out = core.update(cam, np.array([b], np.float32), np.array([0.9], np.float32), [f.copy()])
        ids.append(out[0].track_id if out else None)
    return ids


# ------------------------------------------------------------------ the guard on its own
def test_guard_states():
    A = rv(); g = ShiftGuard(suspect=0.10, confirm=3)
    assert [g.step(at(A, 0.01)) for _ in range(10)] == ["ok"] * 10
    assert g.step(at(A, 0.14)) == "suspect" and g.step(at(A, 0.01)) == "settled"        # a blip that comes back
    far = [at(A, 0.14) for _ in range(3)]
    assert [g.step(x) for x in far] == ["suspect", "suspect", "split"]
    assert len(g.shifted) == 3 and g.peak > 0.10


def test_guard_reference_is_frozen_while_suspicious():
    A = rv(); g = ShiftGuard(suspect=0.10, confirm=50)
    for _ in range(10):
        g.step(at(A, 0.01))
    ref = g.ref.copy()
    for _ in range(10):
        g.step(at(A, 0.15))
    assert np.allclose(g.ref, ref)                                   # the new look did not leak into the reference


# ------------------------------------------------------------------ the split inside BoT-SORT
def test_persistent_jump_to_another_body_gets_a_new_track_id():
    A = rv(); B = at(A, 0.14); core = make()
    ids = run(core, 0, [(box(), at(A, 0.01)) for _ in range(15)] + [(box(), at(B, 0.01)) for _ in range(10)])
    old, new = ids[0], ids[-1]
    assert old != new and len(set(ids[:15])) == 1 and len(set(ids[15:])) <= 2
    first_new = ids.index(new)
    assert 15 <= first_new <= 17                                     # the third shifted frame at the latest
    assert len(set(ids[first_new:])) == 1                            # and it stays stable afterwards
    ev = core.events
    assert len(ev) == 1 and ev[0]["old_track_id"] == old and ev[0]["new_track_id"] == new and ev[0]["peak_distance"] > 0.10


def test_a_blip_that_comes_back_does_not_split():
    A = rv(); core = make()
    ids = run(core, 0, [(box(), at(A, 0.01)) for _ in range(12)] + [(box(), at(A, 0.15)) for _ in range(2)]
                       + [(box(), at(A, 0.01)) for _ in range(10)])
    assert len(set(ids)) == 1 and core.events == []


def test_tracks_at_the_frame_edge_are_not_split():
    A = rv(); B = at(A, 0.14); core = make()
    edge = box(x=2, y=300)                                           # touches the left edge
    ids = run(core, 0, [(edge, at(A, 0.01)) for _ in range(12)] + [(edge, at(B, 0.01)) for _ in range(8)])
    assert len(set(ids)) == 1 and core.events == []


def test_disabled_keeps_the_old_behaviour():
    A = rv(); B = at(A, 0.14); core = make(split_enabled=False)
    ids = run(core, 0, [(box(), at(A, 0.01)) for _ in range(12)] + [(box(), at(B, 0.01)) for _ in range(8)])
    assert len(set(ids)) == 1 and core.events == []                  # BoT-SORT matches the box by IoU: the slide goes unnoticed


def test_new_ids_stay_unique_across_cameras():
    A = rv(); B = at(A, 0.14); core = make()
    ids0 = run(core, 0, [(box(), at(A, 0.01)) for _ in range(12)] + [(box(), at(B, 0.01)) for _ in range(8)])
    ids1 = run(core, 1, [(box(), at(A, 0.01)) for _ in range(12)] + [(box(), at(B, 0.01)) for _ in range(8)])
    assert not (set(ids0) & set(ids1)) and len(set(ids0)) == 2 and len(set(ids1)) == 2


def test_output_after_a_split_carries_the_embedding_for_the_new_track():
    A = rv(); B = at(A, 0.14); core = make()
    f_last = at(B, 0.01)
    for _ in range(12):
        core.update(0, np.array([box()], np.float32), np.array([0.9], np.float32), [at(A, 0.01)])
    outs = [core.update(0, np.array([box()], np.float32), np.array([0.9], np.float32), [at(B, 0.01)])[0] for _ in range(6)]
    split_frame = next(o for o, e in zip(outs, [1, 2, 3, 4, 5, 6]) if o.track_id != outs[0].track_id)
    assert split_frame.feat is not None and split_frame.det_index == 0
