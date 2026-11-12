"""Plain records passed from the per-camera tracker to the tracklet store (no DeepStream imports)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

# object_id of a detection that no track has claimed (DeepStream's "untracked" marker).
UNTRACKED_ID = 0xFFFFFFFFFFFFFFFF


@dataclass
class TrackedObject:
    source_id: int
    frame_num: int
    track_id: int
    class_id: int
    bbox_tlwh: tuple          # (left, top, width, height) in streammux coordinates
    det_confidence: float
    tracker_confidence: float
    reid: Optional[np.ndarray] = None  # ReID vector the tracker used for this track in this frame (float32), else None
    emb_norm: Optional[float] = None   # norm of that vector before normalisation (a model-side confidence signal)


@dataclass
class FrameInfo:
    source_id: int
    frame_num: int       # decoder frame counter of this source
    pts_s: float         # buffer PTS in seconds (shared timeline across file sources)


@dataclass
class FrameTracks:
    info: FrameInfo
    tracks: List[TrackedObject]
