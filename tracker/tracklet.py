"""Per-camera tracklet lifecycle on top of the tracker output.

The tracker gives, per frame, the currently tracked objects (track_id, box, ReID vector).
TrackletStore turns that into Tracklets that carry what the global tracker needs: when the
track entered / left, whether it touches the frame edge, and its embedding.
"""
from __future__ import annotations

import enum
from collections import deque
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from association.embedding import EmbeddingTrack
from .geometry import iou_matrix
from .types import FrameTracks, TrackedObject


def point_in_polygon(x: float, y: float, polygon) -> bool:
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


class TrackletState(enum.Enum):
    ACTIVE = "active"   # seen this frame
    LOST = "lost"       # absent, still inside the grace period (the tracker may bring it back)
    LEFT = "left"       # absent longer than the grace period - finished


class Tracklet:
    def __init__(self, source_id: int, track_id: int, frame: int, frame_num: int,
                 embedding: EmbeddingTrack):
        self.source_id = source_id
        self.track_id = track_id
        self.key = (source_id, track_id)        # unique across cameras (BoT-SORT draws ids from one counter)
        self.first_frame = frame                # entered (shared timeline: round(pts * fps))
        self.last_frame = frame                 # last frame seen
        self.left_frame: Optional[int] = None   # set when state -> LEFT (== last_frame)
        self.first_frame_num = frame_num        # same, in the source's own decoder frame numbers
        self.last_frame_num = frame_num
        self.state = TrackletState.ACTIVE
        self.tracklet_len = 0
        self.class_id = -1
        self.det_conf = 0.0
        self.tracker_conf = 0.0
        self.tlwh = np.zeros(4)
        self.bbox_history: deque = deque(maxlen=30)
        self.is_touching_edge = False
        self.entered_at_edge = False            # touching the edge on its first frame
        self.left_at_edge = False               # touching the edge on its last frame
        self.touched_edge_ever = False
        self.peer_iou = 0.0                     # max IoU with another live tracklet in this camera
        self.in_roi = True                      # bottom-centre of the box inside a ReID ROI
        self.clean = False                      # in_roi and not occluded and not touching edge (this frame)
        self.embedding = embedding              # tracklet-level moving average of its ReID samples
        # which frame / box represents this tracklet in a saved image (decoder frame_num, x1y1x2y2):
        self.sample_ref = None                  # the BEST accepted ReID sample: (frame_num, box, badness, vector)
        self.speed = 0.0                        # box heights moved per frame (motion blur proxy)
        self.aspect = 0.0                       # box width / height
        self.badness = 0.0                      # worst of speed / limit and aspect / limit (<= 1 passes); picks the picture
        self.unclean_reason = None              # first gate this frame failed: roi / edge / overlap / warmup / size / speed / aspect / norm
        self.best_ref = None                    # fallback: the most confident detection seen (frame, box, conf)
        self.new_embedding_this_frame = False   # an accepted ReID sample arrived this frame
        self.assoc_status = None                # set by the association package (NEW/PENDING/COMMITTED)
        self.t_global_id = 0
        self.t_identity_since_frame = 0

    @property
    def cam_source(self) -> int:
        return self.source_id

    def features(self) -> dict:
        """Compact per-tracklet record: the smooth feature (mean of the last N accepted ReID
        samples), the last embedding, camera, and the first / last frame the tracklet was seen.
        start/end_frame are the camera's own decoder frame numbers (the ones printed on the
        annotation video and used in its CSV)."""
        return {"cam_id": self.source_id, "track_id": self.track_id,
                "start_frame": self.first_frame_num, "end_frame": self.last_frame_num,
                "smooth_feat": self.embedding.window_mean(),
                "last_feat": None if self.embedding.last is None else self.embedding.last.copy(),
                "n_samples": self.embedding.count, "n_window": self.embedding.window_count,
                "global_pred": self.t_global_id, **self.image_ref(),
                "image_feat": None if self.sample_ref is None else self.sample_ref[3].copy(), "image_path": None}

    def image_ref(self) -> dict:
        """Which frame + box to cut the tracklet's picture from: its best accepted ReID sample (least blurred /
        least distorted; the latest one among equals), else its most confident detection."""
        if self.sample_ref is not None:
            fn, box = self.sample_ref[0], self.sample_ref[1]
            kind = "best_sample"
        elif self.best_ref is not None:
            fn, box, _ = self.best_ref
            kind = "best_conf"
        else:
            return {"image_frame": None, "image_box": None, "image_kind": None}
        return {"image_frame": int(fn), "image_box": tuple(int(round(v)) for v in box), "image_kind": kind}

    def summary(self) -> dict:
        return {"source": self.source_id, "track_id": self.track_id, "global_id": self.t_global_id,
                "entered": self.first_frame, "left": self.left_frame, "last_seen": self.last_frame,
                "state": self.state.value, "len": self.tracklet_len,
                "entered_at_edge": self.entered_at_edge, "left_at_edge": self.left_at_edge,
                "touched_edge_ever": self.touched_edge_ever,
                "embedding_samples": self.embedding.count,
                "assoc_status": self.assoc_status.value if self.assoc_status else None}


class TrackletStore:
    def __init__(self, frame_w: int, frame_h: int, frame_rate: float,
                 classes: Optional[Sequence[int]] = None, edge_margin: int = 20,
                 left_grace_frames: int = 51, peer_iou_thresh: float = 0.3,
                 rois: Optional[Dict[int, list]] = None,
                 ema_alpha: float = 0.8, smooth_window: int = 50, quality: Optional[dict] = None,
                 on_left: Optional[Callable[[Tracklet], None]] = None):
        self.frame_w, self.frame_h = frame_w, frame_h
        self.frame_rate = frame_rate
        self.classes = set(classes) if classes is not None else None
        self.edge_margin = edge_margin
        self.left_grace_frames = left_grace_frames
        self.peer_iou_thresh = peer_iou_thresh
        self.ema_alpha = ema_alpha
        self.smooth_window = smooth_window
        # image-quality gate for the samples that may feed identity (None = off). A crop is refused when the person
        # moves fast (motion blur), the box is wide (arms out / running / two people), the model's own embedding norm
        # is low, or the box is tiny. See global_tracker.yml for the values and where they come from.
        self.quality = quality
        self.on_left = on_left
        self.rois = rois or {}                         # source_id -> [polygon, ...]; missing = whole frame
        self.tracklets: Dict[tuple, Tracklet] = {}     # ACTIVE + LOST
        self.finished: List[Tracklet] = []             # LEFT (kept for logging / analysis)
        self.events: List[dict] = []
        self._live: Dict[int, List[Tracklet]] = {}

    def touching_edge(self, tlwh) -> bool:
        x, y, w, h = tlwh
        m = self.edge_margin
        return x <= m or y <= m or x + w >= self.frame_w - m or y + h >= self.frame_h - m

    def export_features(self) -> dict:
        """{track_id: Tracklet.features()} for every tracklet seen so far (finished and still
        live); cam_id is just a field of each record. Tracklets that never produced an accepted
        ReID sample have None features. BoT-SORT hands out track ids from one counter across the
        cameras, so ids are unique; if that ever stops being true this raises instead of silently
        overwriting a record."""
        out: dict = {}
        for t in self.finished + list(self.tracklets.values()):
            if t.track_id in out:
                raise ValueError(f"track_id {t.track_id} is used by cam {out[t.track_id]['cam_id']} and "
                                 f"cam {t.source_id}: cannot key features by track_id alone")
            out[t.track_id] = t.features()
        return out

    def _speed(self, tl) -> float:
        """Box heights moved per frame: median of the last three steps of the (Kalman-smoothed) box centre."""
        h = list(tl.bbox_history)[-4:]
        steps = []
        for (f0, b0), (f1, b1) in zip(h, h[1:]):
            gap = f1 - f0
            if 0 < gap <= 3 and b1[3] > 0:
                steps.append(np.hypot((b1[0] + b1[2] / 2) - (b0[0] + b0[2] / 2), (b1[1] + b1[3] / 2) - (b0[1] + b0[3] / 2)) / gap / b1[3])
        return float(np.median(steps)) if steps else 0.0

    def _quality(self, tl, t) -> Optional[str]:
        """Which image-quality gate this crop fails (None = fine). Also sets tl.speed / aspect / badness."""
        q = self.quality
        if not q:
            tl.speed = tl.aspect = tl.badness = 0.0
            return None
        w, h = tl.tlwh[2], tl.tlwh[3]
        tl.speed, tl.aspect = self._speed(tl), (w / h if h > 0 else 0.0)
        tl.badness = max(tl.speed / q["max_speed"], tl.aspect / q["max_aspect"])
        if len(tl.bbox_history) < 2:
            return "warmup"                  # a single box says nothing about motion: wait for the second frame
        if h < q.get("min_height", 0):
            return "size"
        if tl.speed > q["max_speed"]:
            return "speed"
        if tl.aspect > q["max_aspect"]:
            return "aspect"
        if q.get("min_emb_norm") and t.emb_norm is not None and t.emb_norm < q["min_emb_norm"]:
            return "norm"
        return None

    def in_roi(self, source_id: int, tlwh) -> bool:
        polys = self.rois.get(source_id)
        if not polys:
            return True
        x, y, w, h = tlwh
        return any(point_in_polygon(x + w / 2, y + h, poly) for poly in polys)

    def live(self, source_id: int) -> List[Tracklet]:
        return self._live.get(source_id, [])

    def update(self, frames: List[FrameTracks]) -> Dict[int, List[Tracklet]]:
        """Consume one batch (in frame order). Returns {source_id: live Tracklets as of that
        source's LAST frame in the batch}."""
        for ft in frames:
            frame = int(round(ft.info.pts_s * self.frame_rate))
            self._update_source(ft.info.source_id, [t for t in ft.tracks if self._keep(t)],
                                frame, ft.info.frame_num)
        return self._live

    def _keep(self, t: TrackedObject) -> bool:
        return self.classes is None or t.class_id in self.classes

    def _update_source(self, sid: int, tracks: List[TrackedObject], frame: int, frame_num: int) -> None:
        seen = set()
        current: List[Tracklet] = []
        for t in tracks:
            key = (sid, t.track_id)
            tl = self.tracklets.get(key)
            if tl is None:
                tl = Tracklet(sid, t.track_id, frame, frame_num,
                              EmbeddingTrack(self.ema_alpha, self.smooth_window))
                tl.entered_at_edge = self.touching_edge(t.bbox_tlwh)
                self.tracklets[key] = tl
                self.events.append({"event": "entered", "frame": frame, "source": sid,
                                    "track_id": t.track_id, "at_edge": tl.entered_at_edge})
            tl.state = TrackletState.ACTIVE
            tl.last_frame, tl.last_frame_num = frame, frame_num
            tl.tracklet_len += 1
            tl.class_id, tl.det_conf, tl.tracker_conf = t.class_id, t.det_confidence, t.tracker_confidence
            tl.tlwh = np.asarray(t.bbox_tlwh, dtype=np.float64)
            tl.bbox_history.append((frame, tl.tlwh.copy()))
            tl.is_touching_edge = self.touching_edge(t.bbox_tlwh)
            tl.touched_edge_ever |= tl.is_touching_edge
            tl.new_embedding_this_frame = False
            if tl.best_ref is None or t.det_confidence > tl.best_ref[2]:
                x, y, w, h = t.bbox_tlwh
                tl.best_ref = (frame_num, (x, y, x + w, y + h), t.det_confidence)
            seen.add(key)
            current.append(tl)

        # peer overlap (occlusion proxy) among this camera's live tracklets
        if len(current) >= 2:
            boxes = np.stack([tl.tlwh for tl in current])
            m = iou_matrix(boxes, boxes)
            np.fill_diagonal(m, 0.0)
            for tl, v in zip(current, m.max(axis=1)):
                tl.peer_iou = float(v)
        else:
            for tl in current:
                tl.peer_iou = 0.0

        # "clean" person: inside the ReID ROI, not occluded by another person, not touching the frame edge, and the crop
        # is not obviously degraded (see _quality). Only clean samples feed the embeddings and the association.
        for tl, t in zip(current, tracks):
            tl.in_roi = self.in_roi(sid, tl.tlwh)
            reason = ("roi" if not tl.in_roi else "edge" if tl.is_touching_edge
                      else "overlap" if tl.peer_iou > self.peer_iou_thresh else self._quality(tl, t))
            tl.unclean_reason, tl.clean = reason, reason is None
        for tl, t in zip(current, tracks):
            if t.reid is None:
                continue
            ok = tl.clean
            damping = 1.0 - min(max(t.det_confidence, 0.0), 1.0)
            tl.new_embedding_this_frame = ok and tl.embedding.update(t.reid, damping=damping)
            if tl.new_embedding_this_frame and (tl.sample_ref is None or tl.badness <= tl.sample_ref[2]):
                x, y, w, h = tl.tlwh
                tl.sample_ref = (frame_num, (x, y, x + w, y + h), tl.badness, np.array(tl.embedding.last, dtype=np.float32))

        # absent tracklets of this camera: LOST -> LEFT
        for key, tl in list(self.tracklets.items()):
            if key[0] != sid or key in seen:
                continue
            tl.new_embedding_this_frame = False
            if frame - tl.last_frame > self.left_grace_frames:
                tl.state = TrackletState.LEFT
                tl.left_frame = tl.last_frame
                tl.left_at_edge = tl.is_touching_edge
                del self.tracklets[key]
                self.finished.append(tl)
                self.events.append({"event": "left", **tl.summary()})
                if self.on_left:
                    self.on_left(tl)
            else:
                tl.state = TrackletState.LOST
        self._live[sid] = current
