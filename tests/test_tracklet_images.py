import cv2
import numpy as np

from tracker.types import FrameInfo, FrameTracks, TrackedObject
from tracker.tracklet import TrackletStore
from tracker.tracklet_images import save_tracklet_images

W, H, N = 160, 120, 30


def make_video(path):
    """Frame k is uniformly gray level 8*k, with a bright 20x20 square at (40+k, 30)."""
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (W, H))
    for k in range(N):
        f = np.full((H, W, 3), 8 * k, np.uint8)
        f[30:50, 40 + k:60 + k] = 255
        vw.write(f)
    vw.release()


def test_crop_comes_from_the_recorded_frame_and_box(tmp_path):
    vid = tmp_path / "cam.avi"
    make_video(vid)
    feats = {
        5: {"cam_id": 0, "image_frame": 12, "image_box": (40 + 12, 30, 60 + 12, 50)},      # exactly the square at frame 12
        9: {"cam_id": 0, "image_frame": 3, "image_box": (0, 0, 40, 40)},                    # background only
        7: {"cam_id": 0, "image_frame": None, "image_box": None},                           # nothing to save
    }
    out = save_tracklet_images(feats, [{"uri": "file://" + str(vid)}], str(tmp_path / "imgs"), margin=0.0)
    assert set(out) == {5, 9}
    assert out[5].endswith("5.jpg") and out[9].endswith("9.jpg")                            # named by track id
    sq = cv2.imread(out[5]); bg = cv2.imread(out[9])
    assert sq.shape[:2] == (20, 20) and sq.mean() > 240                                     # the bright square, from frame 12
    assert abs(bg.mean() - 8 * 3) < 6                                                       # gray level of frame 3, not another frame


def test_margin_is_clipped_to_the_frame(tmp_path):
    vid = tmp_path / "cam.avi"
    make_video(vid)
    feats = {1: {"cam_id": 0, "image_frame": 0, "image_box": (0, 0, 50, 50)}}
    out = save_tracklet_images(feats, [{"uri": str(vid)}], str(tmp_path / "i"), margin=0.5)
    h, w = cv2.imread(out[1]).shape[:2]
    assert (w, h) == (75, 75)                                                               # grows right/bottom only, no negative coords


def test_non_file_sources_are_skipped(tmp_path):
    feats = {1: {"cam_id": 0, "image_frame": 0, "image_box": (0, 0, 50, 50)}}
    assert save_tracklet_images(feats, [{"uri": "rtsp://10.0.0.1/stream"}], str(tmp_path / "i")) == {}


def obj(tid, fn, box, conf=0.9, reid=None):
    return TrackedObject(0, fn, tid, 1, box, conf, 0.9, reid=reid)


def test_tracklet_picture_is_its_best_sample_and_falls_back_to_the_most_confident_detection():
    st = TrackletStore(1920, 1080, 30.0, classes=[1])
    v = np.ones(8, np.float32)
    # track 1: clean samples at frames 10 and 14 (box moves) -> picture = frame 14
    st.update([FrameTracks(FrameInfo(0, 10, 10 / 30), [obj(1, 10, (300, 300, 100, 300), reid=v)])])
    st.update([FrameTracks(FrameInfo(0, 12, 12 / 30), [obj(1, 12, (310, 300, 100, 300))])])
    st.update([FrameTracks(FrameInfo(0, 14, 14 / 30), [obj(1, 14, (330, 300, 100, 300), reid=v)])])
    st.update([FrameTracks(FrameInfo(0, 15, 15 / 30), [obj(1, 15, (340, 300, 100, 300))])])     # later frame, no sample
    # track 2: never gets a sample -> falls back to its most confident detection
    for fn, conf in ((20, 0.5), (21, 0.8), (22, 0.6)):
        st.update([FrameTracks(FrameInfo(0, fn, fn / 30), [obj(2, fn, (500 + fn, 300, 100, 300), conf=conf)])])
    f = st.export_features()
    assert f[1]["image_kind"] == "best_sample" and f[1]["image_frame"] == 14
    assert f[1]["image_box"] == (330, 300, 430, 600)
    assert f[2]["image_kind"] == "best_conf" and f[2]["image_frame"] == 21
    assert f[2]["image_box"] == (521, 300, 621, 600) and f[2]["image_path"] is None
