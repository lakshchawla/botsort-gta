import numpy as np

from tracker.tracklet import TrackletStore
from tracker.types import FrameInfo, FrameTracks, TrackedObject

rng = np.random.default_rng(21)
V = rng.normal(size=32).astype(np.float32)
Q = dict(max_speed=0.04, max_aspect=0.65, min_emb_norm=22.40, min_height=64)


def store(quality=Q):
    return TrackletStore(1920, 1080, 30.0, classes=[1], quality=quality)


def feed(st, boxes, norm=22.6, tid=1, start=0):
    """boxes: (x, y, w, h) per frame for one track; returns the tracklet after the last frame"""
    for i, b in enumerate(boxes):
        f = start + i
        st.update([FrameTracks(FrameInfo(0, f, f / 30), [TrackedObject(0, f, tid, 1, tuple(b), 0.9, 0.9, reid=V.copy(), emb_norm=norm)])])
    return st.tracklets[(0, tid)]


def walk(n, x0=600, dx=1.0, w=90, h=260):
    return [(x0 + dx * i, 300, w, h) for i in range(n)]


def test_a_calm_normal_person_is_clean():
    tl = feed(store(), walk(10))
    assert tl.clean and tl.unclean_reason is None and tl.speed < 0.01 and 0.3 < tl.aspect < 0.4
    assert tl.embedding.count == 9                                 # all but the first frame (no motion history yet)


def test_fast_movement_is_refused_as_motion_blur():
    tl = feed(store(), walk(10, dx=20.0))                         # 20 px per frame on a 260 px person = 0.077 heights / frame
    assert not tl.clean and tl.unclean_reason == "speed" and tl.speed > 0.04
    assert tl.embedding.count == 0 and tl.sample_ref is None       # nothing from these frames reaches the embedding or the picture


def test_the_first_frame_of_a_track_is_not_trusted_yet():
    st = store()
    tl = feed(st, walk(1))
    assert not tl.clean and tl.unclean_reason == "warmup" and tl.embedding.count == 0
    feed(st, walk(1, x0=601), start=1)
    assert tl.clean and tl.embedding.count == 1


def test_a_wide_box_is_refused():
    tl = feed(store(), walk(10, w=170, h=260))                    # aspect 0.65+
    assert not tl.clean and tl.unclean_reason == "aspect" and tl.embedding.count == 0


def test_a_low_embedding_norm_is_refused_and_a_tiny_box_too():
    tl = feed(store(), walk(5), norm=22.30)
    assert not tl.clean and tl.unclean_reason == "norm"
    tl2 = feed(store(), [(600, 300, 20, 50)] * 5, tid=2)
    assert not tl2.clean and tl2.unclean_reason == "size"


def test_gate_off_means_the_old_behaviour():
    tl = feed(store(quality=None), walk(10, dx=20.0, w=170))
    assert tl.clean and tl.embedding.count == 10                   # no gate, no warm-up either


def test_geometry_gates_come_before_quality_in_the_reason():
    tl = feed(store(), [(2, 300, 90, 260)] * 4)                   # touches the left edge AND would pass quality
    assert tl.unclean_reason == "edge"


def test_the_picture_is_the_best_sample_not_the_last_one():
    st = store()
    calm1 = walk(6, x0=600, dx=1.0)                                # frames 0-5
    fast = walk(5, x0=700, dx=16.0)                                # frames 6-10: ~0.06 box heights per frame -> refused
    calm2 = walk(6, x0=900, dx=1.0)                                # frames 11-16 (the first one still carries the jump)
    tl = feed(st, calm1 + fast + calm2)
    frame, box, badness, vec = tl.sample_ref
    assert frame not in range(6, 11)                               # never one of the refused fast frames
    assert badness < 0.6 and frame == 16                           # calm frames tie on badness: the latest wins
    rec = tl.features()
    assert rec["image_frame"] == 16 and rec["image_kind"] == "best_sample" and np.allclose(rec["image_feat"], vec)


def test_a_sharper_earlier_sample_beats_a_blurrier_later_one():
    st = store()
    tl = feed(st, walk(8, dx=0.5) + walk(6, x0=620, dx=8.0))        # slow, then quicker but still under the limit
    frame, box, badness, vec = tl.sample_ref
    # speed is a median of the last 3 steps, so the single-frame jump at frame 8 is not mistaken for blur (frame 8 may tie
    # with the calm ones); the quicker frames after it (badness ~0.77) must not win
    assert frame <= 8 and badness < 0.6 and tl.features()["image_frame"] == frame
