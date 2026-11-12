import os
import sys

import numpy as np
import pytest

from tracker.botsort_backend import BotSortCore, _iou

REPO = os.environ.get("BOTSORT_REPO", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "third_party", "BoT-SORT"))
rng = np.random.default_rng(0)


def unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def person(x, y=300, w=80, h=240):
    return [x, y, x + w, y + h]


@pytest.fixture
def core():
    return BotSortCore(REPO, dict(track_high_thresh=0.6, track_low_thresh=0.1, new_track_thresh=0.6,
                                  track_buffer=30, match_thresh=0.8, proximity_thresh=0.5, appearance_thresh=0.1))


def run(core, cam, frames):
    """frames: list of (boxes, scores, feats). Returns the tracks of every frame."""
    return [core.update(cam, np.array(b, np.float32).reshape(-1, 4), np.array(s, np.float32), f) for b, s, f in frames]


def test_loading_the_repo_leaves_our_package_and_numpy_usable():
    from tracker.botsort_loader import load_botsort
    load_botsort(REPO)
    import tracker
    import tracker.tracklet                                    # our package still resolves after the repo was loaded
    assert hasattr(tracker.tracklet, "Tracklet") and "botsort_repo.bot_sort" in sys.modules
    assert np.float is float                                   # the shim the upstream code needs on numpy 2


def test_one_person_keeps_one_id_and_the_feature_is_used(core):
    f = unit(rng.normal(size=64))
    out = run(core, 0, [([person(100 + 4 * i)], [0.9], [f.copy()]) for i in range(15)])
    ids = {t.track_id for o in out for t in o}
    assert len(ids) == 1 and len(out[-1]) == 1
    last = out[-1][0]
    assert last.feat is not None and np.allclose(last.feat, f, atol=1e-5)  # the injected embedding reached the tracker
    assert last.det_index == 0 and abs(last.score - 0.9) < 1e-6


def test_new_tracks_are_reported_from_their_first_frame(core):
    """This checkout of BoT-SORT has the is_activated filter commented out, so tentative tracks are output;
    one-frame false positives disappear on their own and are further filtered by the tracklet store."""
    f, g = unit(rng.normal(size=64)), unit(rng.normal(size=64))
    frames = [([person(100)], [0.9], [f.copy()]) for _ in range(4)]
    frames += [([person(100), person(600)], [0.9, 0.9], [f.copy(), g.copy()]) for _ in range(3)]     # a 2nd person appears
    frames += [([person(100)], [0.9], [f.copy()]) for _ in range(2)]                                  # ...and is gone
    out = run(core, 0, frames)
    assert [len(o) for o in out] == [1, 1, 1, 1, 2, 2, 2, 1, 1]


def test_two_people_get_two_ids_and_detections_are_mapped_back(core):
    fa, fb = unit(rng.normal(size=64)), unit(rng.normal(size=64))
    frames = [([person(100 + 3 * i), person(700 - 3 * i)], [0.9, 0.8], [fa.copy(), fb.copy()]) for i in range(12)]
    out = run(core, 0, frames)[-1]
    assert len(out) == 2 and len({t.track_id for t in out}) == 2
    by_det = {t.det_index: t for t in out}
    assert set(by_det) == {0, 1}
    assert np.allclose(by_det[0].feat, fa, atol=1e-5) and np.allclose(by_det[1].feat, fb, atol=1e-5)


def test_ids_are_unique_across_cameras_even_when_created_late(core):
    fa = unit(rng.normal(size=64))
    run(core, 0, [([person(100 + 3 * i)], [0.9], [fa.copy()]) for i in range(8)])        # camera 0 already has a track
    out1 = run(core, 1, [([person(400 + 3 * i)], [0.9], [fa.copy()]) for i in range(8)])  # camera 1 starts afterwards
    ids0 = {t.track_id for t in core._tracker(0).tracked_stracks}
    ids1 = {t.track_id for o in out1 for t in o}
    assert ids0 and ids1 and not (ids0 & ids1)                 # the shared counter was not reset by the second camera


def test_lost_person_comes_back_with_the_same_id(core):
    f = unit(rng.normal(size=64))
    out = run(core, 0, [([person(300)], [0.9], [f.copy()]) for _ in range(10)])
    tid = out[-1][0].track_id
    run(core, 0, [([], [], []) for _ in range(5)])                                         # 5 frames without a detection
    back = run(core, 0, [([person(300)], [0.9], [f.copy()]) for _ in range(3)])[-1]
    assert [t.track_id for t in back] == [tid]


def test_low_score_match_reports_no_fresh_feature(core):
    """A detection below track_high_thresh is matched by IoU only and carries no embedding into the tracker."""
    f = unit(rng.normal(size=64))
    run(core, 0, [([person(300)], [0.9], [f.copy()]) for _ in range(10)])
    out = run(core, 0, [([person(300)], [0.3], [unit(rng.normal(size=64))])])[-1]
    assert len(out) == 1 and out[0].feat is None and out[0].det_index == 0


def test_iou_helper():
    a = np.array([[0, 0, 10, 10]], np.float32)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], np.float32)
    assert np.allclose(_iou(a, b), [[1.0, 1 / 3, 0.0]], atol=1e-6)


def test_backend_reads_its_config_and_declares_how_long_a_lost_id_can_come_back(tmp_path):
    from tracker.botsort_backend import BotSortBackend
    (tmp_path / "bs.yml").write_text("track_buffer: 600\nframe_rate: 30\nappearance_thresh: 0.055\n")
    be = BotSortBackend({"repo_root": REPO, "params": "bs.yml",
                         "sgie": {"config-file": "sgie.yml", "unique-id": 2}}, str(tmp_path))
    assert be.left_grace_frames == 600 and be.sgie_id == 2 and be.sgie_config == str(tmp_path / "sgie.yml")
    assert be.core.args.appearance_thresh == 0.055 and be.core.args.with_reid and be.core.args.cmc_method == "none"
