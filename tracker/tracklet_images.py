"""Cut one picture per tracklet out of the source videos.

The tracklet features say WHICH frame and box represent each tracklet (Tracklet.image_ref); this
reads each camera's video once, front to back, and saves those crops as <track_id>.jpg. Working
from the source files (not from the live DeepStream buffers) leaves the inference pipeline
untouched. OpenCV's frame index equals the decoder's frame_num (checked against the annotation
video), so frame_num can be used directly.
"""
from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional
from urllib.parse import unquote, urlparse

import cv2

log = logging.getLogger("tracklet_images")


def _file_path(uri: str) -> Optional[str]:
    if uri.startswith("file://"):
        return unquote(urlparse(uri).path)
    return uri if os.path.isfile(uri) else None


def save_tracklet_images(features: Dict[int, dict], sources: List[dict], out_dir: str,
                         margin: float = 0.1, quality: int = 95) -> Dict[int, str]:
    """features: {track_id: record with cam_id, image_frame, image_box}; sources[i] has 'uri' for
    camera i. Writes out_dir/<track_id>.jpg (box grown by `margin`, clipped to the frame) and
    returns {track_id: absolute path}. Cameras that are not local files (e.g. rtsp://) are skipped."""
    os.makedirs(out_dir, exist_ok=True)
    wanted: Dict[int, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))   # cam -> frame -> [track_id]
    for tid, rec in features.items():
        if rec.get("image_frame") is not None:
            wanted[rec["cam_id"]][rec["image_frame"]].append(tid)

    saved: Dict[int, str] = {}
    for cam, frames in sorted(wanted.items()):
        path = _file_path(sources[cam]["uri"]) if cam < len(sources) else None
        cap = cv2.VideoCapture(path) if path else None
        if cap is None or not cap.isOpened():
            log.warning("camera %s: source %r is not a readable local video; no pictures for its tracklets",
                        cam, sources[cam]["uri"] if cam < len(sources) else None)
            continue
        last = max(frames)
        idx = 0
        while idx <= last and cap.grab():
            if idx in frames:
                ok, img = cap.retrieve()
                if ok:
                    h, w = img.shape[:2]
                    for tid in frames[idx]:
                        x1, y1, x2, y2 = features[tid]["image_box"]
                        mx, my = (x2 - x1) * margin, (y2 - y1) * margin
                        x1, y1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
                        x2, y2 = min(w, int(x2 + mx)), min(h, int(y2 + my))
                        if x2 - x1 < 2 or y2 - y1 < 2:
                            continue
                        out = os.path.abspath(os.path.join(out_dir, f"{tid}.jpg"))
                        cv2.imwrite(out, img[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, quality])
                        saved[tid] = out
            idx += 1
        cap.release()
    return saved
