import numpy as np

from tracker.types import FrameInfo, FrameTracks, TrackedObject
from tracker.tracklet import TrackletStore

rng = np.random.default_rng(3)


def obj(tid, fn, reid, box=(300, 300, 100, 300)):
    return TrackedObject(0, fn, tid, 1, box, 0.9, 0.9, reid=reid)


def test_export_features_dict(tmp_path):
    st = TrackletStore(1920, 1080, 30.0, classes=[1], smooth_window=4)
    vs = [rng.normal(size=32).astype(np.float32) for _ in range(6)]
    for i, v in enumerate(vs):                              # frames 10..15 for track 7 (a clean person)
        st.update([FrameTracks(FrameInfo(0, 10 + i, (10 + i) / 30), [obj(7, 10 + i, v)])])
    st.update([FrameTracks(FrameInfo(0, 16, 16 / 30), [obj(8, 16, None)])])    # track 8: never has an embedding
    feats = st.export_features()
    assert set(feats) == {7, 8}
    f = feats[7]
    assert f["cam_id"] == 0 and f["track_id"] == 7 and f["start_frame"] == 10 and f["end_frame"] == 15
    unit_ = lambda x: x / np.linalg.norm(x)
    last4 = [unit_(v) for v in vs[-4:]]
    assert np.allclose(f["smooth_feat"], unit_(np.mean(last4, axis=0)), atol=1e-5)    # mean of the last 4 (window)
    assert np.allclose(f["last_feat"], unit_(vs[-1]), atol=1e-5)
    assert f["n_samples"] == 6 and f["n_window"] == 4
    g = feats[8]
    assert g["smooth_feat"] is None and g["last_feat"] is None and g["n_samples"] == 0
    assert g["start_frame"] == 16 == g["end_frame"]

    path = tmp_path / "f.npy"
    np.save(path, feats, allow_pickle=True)
    back = np.load(path, allow_pickle=True).item()
    assert set(back) == set(feats) and np.allclose(back[7]["last_feat"], f["last_feat"])


def test_same_track_id_on_two_cameras_is_refused():
    import pytest
    st = TrackletStore(1920, 1080, 30.0, classes=[1])
    v = rng.normal(size=8).astype(np.float32)
    st.update([FrameTracks(FrameInfo(0, 1, 1 / 30), [obj(5, 1, v)])])
    o = TrackedObject(1, 1, 5, 1, (300, 300, 100, 300), 0.9, 0.9, reid=v)
    st.update([FrameTracks(FrameInfo(1, 1, 1 / 30), [o])])
    with pytest.raises(ValueError):
        st.export_features()
