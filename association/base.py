"""The association contract. Any association logic (v1, v2, an experiment ...) plugs into the
global tracker by providing ONE thing: `step`.

    step(cam_id, live_tracklets, frame) -> list of tracklets that obtained a global id this call

`live_tracklets` are tracker.tracklet.Tracklet objects (see that class for the fields: key,
embedding / embedding.value / .last, clean, tracklet_len, peer_iou, new_embedding_this_frame,
tlwh, source_id, track_id ...). The association writes its decision onto the tracklet
(`t_global_id`, optionally `assoc_status`). Everything else below is optional; sensible no-op
defaults come from AssociationBase, and the loader adds them to plain classes / modules.

    release(key)    a tracklet left the scene (key = (source_id, track_id))
    size()          number of global identities
    n_pending()     tracklets waiting for a decision
    events          list of dicts, drained by drain_events() into the JSONL log
    from_config(d)  classmethod; build the instance from a dict (default: cls(**d))
"""
from __future__ import annotations

from typing import List


class AssociationBase:
    def __init__(self):
        self.events: List[dict] = []

    def step(self, cam_id, live: list, frame: int) -> list:
        raise NotImplementedError

    def release(self, key: tuple) -> None:
        pass

    def size(self) -> int:
        return 0

    def n_pending(self) -> int:
        return 0

    def drain_events(self) -> List[dict]:
        ev, self.events = getattr(self, "events", []), []
        return ev

    @classmethod
    def from_config(cls, cfg: dict) -> "AssociationBase":
        return cls(**(cfg or {}))
