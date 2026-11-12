"""BoT-SORT as the per-camera tracker, fed by a DeepStream ReID SGIE.

    nvstreammux -> nvinfer PGIE (person detector) -> nvinfer SGIE (ReID embedder) -> [probe: BoT-SORT]

The probe reads, for every person detection of a frame, its box, score and the embedding the SGIE attached
to it, runs one upstream BoT-SORT instance per camera (https://github.com/NirAharon/BoT-SORT, used
unmodified, see botsort_loader.py), and returns FrameTracks for the tracklet store and the association.

BotSortCore    pure python: boxes + scores + embeddings in, tracks out (unit-testable, no DeepStream)
BotSortBackend DeepStream glue: builds the SGIE element, reads detections / embeddings from the metadata,
               writes the track id back into the object metadata (for the on-screen labels)
"""
from __future__ import annotations

import ctypes
import os
from argparse import Namespace
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

from .botsort_loader import InjectedReID, load_botsort
from .shift_guard import ShiftGuard
from .types import UNTRACKED_ID, FrameInfo, FrameTracks, TrackedObject

DEFAULT_PARAMS = dict(
    track_high_thresh=0.6, track_low_thresh=0.1, new_track_thresh=0.6, track_buffer=30, match_thresh=0.8,
    proximity_thresh=0.5, appearance_thresh=0.25, cmc_method="none", mot20=False, frame_rate=30,
    # a track whose embeddings jump to another body and stay there is ended and a new id is minted
    split_enabled=False, split_dist=0.10, split_confirm=3, split_settle_ratio=0.8,
    split_skip_edge=True, edge_margin=20, frame_w=1920, frame_h=1080)


@dataclass
class TrackOut:
    track_id: int
    tlwh: np.ndarray
    score: float
    feat: Optional[np.ndarray]     # the detection embedding the tracker used THIS frame; None if it did not use one
    det_index: int                 # index of the input detection this track lies on (-1 if none)
    emb_norm: Optional[float] = None   # norm of that detection's raw (pre-normalisation) SGIE vector


def _iou(a_xyxy: np.ndarray, b_xyxy: np.ndarray) -> np.ndarray:
    ix = np.clip(np.minimum(a_xyxy[:, None, 2], b_xyxy[None, :, 2]) - np.maximum(a_xyxy[:, None, 0], b_xyxy[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a_xyxy[:, None, 3], b_xyxy[None, :, 3]) - np.maximum(a_xyxy[:, None, 1], b_xyxy[None, :, 1]), 0, None)
    inter = ix * iy
    area = lambda x: np.clip(x[:, 2] - x[:, 0], 0, None) * np.clip(x[:, 3] - x[:, 1], 0, None)
    union = area(a_xyxy)[:, None] + area(b_xyxy)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


class BotSortCore:
    """One upstream BoTSORT per camera, with the SGIE embeddings injected in place of its FastReID encoder.

    Camera motion compensation is off (cmc_method 'none'): the cameras are fixed, and no frame pixels are
    needed. The upstream track-id counter is class-level, so ids are unique across cameras."""

    def __init__(self, repo_root: str, params: Optional[dict] = None):
        self.bot_sort, self.basetrack, _ = load_botsort(repo_root)
        p = {**DEFAULT_PARAMS, **(params or {})}
        self.params = p
        self.args = Namespace(
            track_high_thresh=p["track_high_thresh"], track_low_thresh=p["track_low_thresh"],
            new_track_thresh=p["new_track_thresh"], track_buffer=p["track_buffer"], match_thresh=p["match_thresh"],
            proximity_thresh=p["proximity_thresh"], appearance_thresh=p["appearance_thresh"], with_reid=True,
            fast_reid_config=None, fast_reid_weights=None, device="cpu", cmc_method=p["cmc_method"],
            name="ds", ablation=False, mot20=bool(p["mot20"]))
        self._trackers: Dict[int, object] = {}
        self._last_feat: Dict[int, Dict[int, np.ndarray]] = {}      # camera -> track id -> last feature seen
        self._guards: Dict[int, Dict[int, ShiftGuard]] = {}         # camera -> track id -> shift guard
        self.events: List[dict] = []
        # BoTSORT.__init__ calls BaseTrack.clear_count(); the counter is shared by all cameras, so it must
        # only be reset once per core (here), never again when a later camera's tracker is created.
        bt = self.basetrack.BaseTrack
        if not hasattr(bt, "_orig_clear_count"):
            bt._orig_clear_count = bt.clear_count
            bt.clear_count = staticmethod(lambda: None)
        bt._count = 0

    def _tracker(self, cam: int):
        t = self._trackers.get(cam)
        if t is None:
            t = self.bot_sort.BoTSORT(self.args, frame_rate=self.params["frame_rate"])
            t.encoder = InjectedReID()
            self._trackers[cam] = t
        return t

    def update(self, cam: int, boxes_xyxy: np.ndarray, scores: np.ndarray, feats: np.ndarray) -> List[TrackOut]:
        """One frame of one camera. boxes (N,4) x1y1x2y2, scores (N,), feats (N,D) -> the active tracks."""
        tracker = self._tracker(cam)
        n = len(boxes_xyxy)
        raw_norm = [float(np.linalg.norm(f)) for f in feats]              # before the upstream code normalises in place
        dets = (np.concatenate([np.asarray(boxes_xyxy, np.float32).reshape(n, 4),
                                np.asarray(scores, np.float32).reshape(n, 1)], axis=1)
                if n else np.zeros((0, 5), np.float32))
        tracker.encoder.set(dets[:, :4], [np.array(f, dtype=np.float32) for f in feats])   # copies: upstream normalises in place
        online = tracker.update(dets, None)

        out: List[TrackOut] = []
        if online:
            tlbr = np.array([t.tlbr for t in online], dtype=np.float32)
            det_idx = np.full(len(online), -1)
            if n:
                iou = _iou(tlbr, dets[:, :4])
                rows, cols = linear_sum_assignment(1.0 - iou)
                for r, c in zip(rows, cols):
                    if iou[r, c] >= 0.3:
                        det_idx[r] = c
            seen = self._last_feat.setdefault(cam, {})
            for t, di in zip(online, det_idx):
                prev = seen.get(t.track_id)
                fresh = t.curr_feat is not None and t.curr_feat is not prev      # a new feature was fed in this frame
                if t.curr_feat is not None:
                    seen[t.track_id] = t.curr_feat
                out.append(TrackOut(t.track_id, np.asarray(t.tlwh, dtype=np.float64).copy(), float(t.score),
                                    np.array(t.curr_feat, dtype=np.float32) if fresh else None, int(di),
                                    raw_norm[di] if 0 <= di < len(raw_norm) else None))
        if self.params["split_enabled"]:
            self._apply_splits(cam, tracker, dets, out)
        if tracker.frame_id % 300 == 0:                                          # drop bookkeeping of finished tracks
            alive = {t.track_id for t in tracker.tracked_stracks + tracker.lost_stracks}
            seen = self._last_feat.get(cam, {})
            self._last_feat[cam] = {k: v for k, v in seen.items() if k in alive}
            guards = self._guards.get(cam, {})
            self._guards[cam] = {k: v for k, v in guards.items() if k in alive}
        return out

    # --- splitting a track whose embeddings moved to another body ---------------------------------------
    def _touches_edge(self, tlwh) -> bool:
        p, m = self.params, self.params["edge_margin"]
        x, y, w, h = tlwh
        return x <= m or y <= m or x + w >= p["frame_w"] - m or y + h >= p["frame_h"] - m

    def _new_guard(self) -> ShiftGuard:
        p = self.params
        return ShiftGuard(p["split_dist"], p["split_confirm"], p["split_settle_ratio"])

    def _apply_splits(self, cam: int, tracker, dets: np.ndarray, out: List[TrackOut]) -> None:
        """For every track that used a fresh embedding this frame: if its embeddings left the track's slow reference
        and stayed away, end the track and continue the detection under a NEW track id (upstream BoT-SORT otherwise
        keeps matching by IoU, which is how a box can slide from one person to another)."""
        guards = self._guards.setdefault(cam, {})
        for i, o in enumerate(out):
            if o.feat is None or o.det_index < 0:
                continue
            if self.params["split_skip_edge"] and self._touches_edge(o.tlwh):
                continue                                         # a body cut by the frame edge says little about identity
            g = guards.get(o.track_id)
            if g is None:
                g = guards[o.track_id] = self._new_guard()
            if g.step(o.feat) != "split":
                continue
            old = next((t for t in tracker.tracked_stracks if t.track_id == o.track_id), None)
            if old is None:
                continue
            det = dets[o.det_index]                              # [x1, y1, x2, y2, score]
            new = self.bot_sort.STrack(self.bot_sort.STrack.tlbr_to_tlwh(det[:4]), float(det[4]),
                                       np.array(o.feat, dtype=np.float32))
            new.activate(tracker.kalman_filter, tracker.frame_id)       # takes the next id of the shared counter
            new.is_activated = True
            old.mark_removed()
            tracker.tracked_stracks.remove(old)
            tracker.removed_stracks.append(old)
            tracker.tracked_stracks.append(new)
            guards.pop(o.track_id, None)
            ng = guards[new.track_id] = self._new_guard()
            ng.reseed(g.shifted)
            self._last_feat.setdefault(cam, {})[new.track_id] = new.curr_feat
            self.events.append(dict(event="track_split", cam=cam, frame=tracker.frame_id - 1, old_track_id=o.track_id,
                                    new_track_id=new.track_id, peak_distance=round(g.peak, 4)))
            out[i] = TrackOut(new.track_id, np.asarray(new.tlwh, dtype=np.float64).copy(), float(new.score),
                              np.array(new.curr_feat, dtype=np.float32), o.det_index, o.emb_norm)


def _embedding_of(obj_meta, gie_id: int) -> Optional[np.ndarray]:
    """The tensor the ReID SGIE attached to one detection (needs output-tensor-meta: 1)."""
    import pyds
    l_user = obj_meta.obj_user_meta_list
    while l_user is not None:
        try:
            um = pyds.NvDsUserMeta.cast(l_user.data)
        except StopIteration:
            break
        if um.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
            tm = pyds.NvDsInferTensorMeta.cast(um.user_meta_data)
            if tm.unique_id == gie_id:
                layer = pyds.get_nvds_LayerInfo(tm, 0)
                ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
                n = 1
                for i in range(layer.inferDims.numDims):
                    n *= layer.inferDims.d[i]
                return np.copy(np.ctypeslib.as_array(ptr, shape=(n,)))
        l_user = l_user.next
    return None


class BotSortBackend:
    """What DSPipeline needs from the tracker: `elements()` (the SGIE) and `process()` (BoT-SORT in the probe)."""

    def __init__(self, cfg: dict, config_dir: str = "", person_class_id: int = 1):
        def resolve(p):
            return p if os.path.isabs(p) else os.path.join(config_dir, p)

        params = cfg.get("params", {})
        if isinstance(params, str):
            with open(resolve(params)) as f:
                params = yaml.safe_load(f) or {}
        default_repo = os.environ.get("BOTSORT_REPO") or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "third_party", "BoT-SORT")
        self.core = BotSortCore(resolve(cfg.get("repo_root", default_repo)), params)
        self.sgie_config = resolve(cfg["sgie"]["config-file"])
        self.sgie_id = int(cfg["sgie"].get("unique-id", 2))
        self.person_class_id = person_class_id
        self.n_without_embedding = 0
        # BoT-SORT keeps a lost track re-findable for track_buffer frames and then re-uses the SAME id, so a
        # tracklet may only be called finished after that long (see TrackletStore.left_grace_frames)
        self.left_grace_frames = int(self.core.params["track_buffer"] * self.core.params["frame_rate"] / 30.0)

    def drain_events(self) -> List[dict]:
        ev, self.core.events = self.core.events, []
        return ev

    def elements(self) -> list:
        from gi.repository import Gst
        sgie = Gst.ElementFactory.make("nvinfer", "sgie-reid")
        if sgie is None:
            raise RuntimeError("Could not create the ReID SGIE (nvinfer)")
        sgie.set_property("config-file-path", self.sgie_config)
        return [sgie]

    def process(self, batch_meta, pgie_id: int) -> List[FrameTracks]:
        import pyds
        frames = []
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                fm = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break
            frames.append(fm)
            l_frame = l_frame.next
        frames.sort(key=lambda f: (f.source_id, f.frame_num))

        result: List[FrameTracks] = []
        for fm in frames:
            objs, boxes, scores, feats = [], [], [], []
            l_obj = fm.obj_meta_list
            while l_obj is not None:
                try:
                    om = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break
                l_obj = l_obj.next
                if om.unique_component_id != pgie_id:
                    continue
                om.object_id = UNTRACKED_ID                  # until BoT-SORT claims it
                emb = _embedding_of(om, self.sgie_id)
                if emb is None:                              # BoT-SORT needs an embedding for every detection it sees
                    self.n_without_embedding += 1
                    continue
                r = om.rect_params
                objs.append(om)
                boxes.append((r.left, r.top, r.left + r.width, r.top + r.height))
                scores.append(om.confidence)
                feats.append(emb)

            tracks = self.core.update(fm.source_id, np.array(boxes, dtype=np.float32).reshape(-1, 4),
                                      np.array(scores, dtype=np.float32), feats)
            info = FrameInfo(fm.source_id, fm.frame_num, fm.buf_pts / 1e9)
            tracked = []
            for t in tracks:
                if t.det_index >= 0:
                    objs[t.det_index].object_id = t.track_id         # lets the on-screen labels find the track
                x, y, w, h = t.tlwh
                tracked.append(TrackedObject(
                    source_id=fm.source_id, frame_num=fm.frame_num, track_id=t.track_id,
                    class_id=self.person_class_id, bbox_tlwh=(float(x), float(y), float(w), float(h)),
                    det_confidence=t.score, tracker_confidence=t.score, reid=t.feat, emb_norm=t.emb_norm))
            result.append(FrameTracks(info, tracked))
        return result
