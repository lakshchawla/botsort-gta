"""GlobalTracker: per-camera tracks -> tracklets -> cross-camera global identities."""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import yaml

from association import load_associator
from tracker.types import FrameTracks
from tracker.tracklet import Tracklet, TrackletStore

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "pipeline", "config", "global_tracker.yml")


class GlobalTracker:
    def __init__(self, frame_w: int = 1920, frame_h: int = 1080, config_path: str = DEFAULT_CONFIG,
                 rois: Optional[Dict[int, list]] = None,
                 association=None, association_config=None):
        """association: spec of the association logic (dict or 'module_or_path[:Class]'); defaults
        to the `association:` block of the config. Anything with a step() works, see
        association/base.py."""
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        base_dir = os.path.dirname(os.path.abspath(config_path))
        self.assoc = load_associator(association or cfg["association"], base_dir=base_dir,
                                     config=association_config)
        self.store = TrackletStore(
            frame_w, frame_h, cfg["frame_rate"], classes=cfg.get("classes"), rois=rois,
            on_left=lambda t: self.assoc.release(t.key), **cfg["tracklet"])
        self.frame_rate = cfg["frame_rate"]

    def update(self, frames: List[FrameTracks], on_frame=None) -> Dict[int, List[Tracklet]]:
        """One batch. Frames are processed in order: tracklet bookkeeping, then association.
        on_frame(frame_tracks, live_tracklets) is called after each frame is fully processed (the
        tracklets' t_global_id is that frame's online prediction).
        Returns {source_id: live Tracklets} with t_global_id filled where resolved."""
        for ft in frames:
            frame = int(round(ft.info.pts_s * self.frame_rate))
            sid = ft.info.source_id
            self.store.update([ft])
            for t in self.assoc.step(sid, self.store.live(sid), frame):
                self.store.events.append({"event": "identity_assigned", "frame": frame,
                                          "source": t.source_id, "track_id": t.track_id,
                                          "global_id": t.t_global_id})
            if on_frame is not None:
                on_frame(ft, self.store.live(sid))
        return self.store._live

    def drain_events(self) -> List[dict]:
        ev = self.store.events + self.assoc.drain_events()
        self.store.events = []
        return sorted(ev, key=lambda e: e.get("frame", 0))

    def save_features(self, path: str, sources: Optional[list] = None, images_dir: Optional[str] = None) -> int:
        """Save {track_id: {cam_id, start_frame, end_frame, smooth_feat, last_feat, ..., image_path}}
        as a pickled dict in an .npy file. Load with: np.load(path, allow_pickle=True).item()
        If `sources` (the pipeline's source list, with 'uri') and `images_dir` are given, one picture
        per tracklet is cut from the source videos as <images_dir>/<track_id>.jpg and its absolute
        path is stored in the record's `image_path`."""
        import numpy as np
        feats = self.store.export_features()
        if sources and images_dir:
            from tracker.tracklet_images import save_tracklet_images
            for tid, p in save_tracklet_images(feats, sources, images_dir).items():
                feats[tid]["image_path"] = p
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.save(path, feats, allow_pickle=True)
        return len(feats)

    def all_tracklets(self) -> List[Tracklet]:
        return self.store.finished + list(self.store.tracklets.values())
