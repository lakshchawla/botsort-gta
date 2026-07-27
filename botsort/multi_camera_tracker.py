"""
Multi-camera tracking via Global Tracklet Association (GTA).

Pure orchestration - no new matching logic lives here. Cross-camera identity
matching is already provided by global_tracklet_association.py's GTA: it has
no camera concept anywhere in it, so calling step() once per camera against
ONE shared GTA instance makes a camera-B tracklet's appearance query
transparently match a camera-A-registered identity, through the exact same
code path already used for same-camera re-entry.

This depends on each camera's BoTSORT being constructed with a distinct
cam_source (see bot_sort.py - this seeds a per-instance ID_Assigner so
track_ids never collide across cameras) and the same shared `registry=`.

Assumption this design relies on: camera views are non-overlapping - a
person is never live-tracked in two cameras at the same instant.
"""

from __future__ import annotations

from typing import Optional

from .bot_sort import BoTSORT
from .global_tracklet_association import GTA


class MultiCameraTracker:
    """Thin orchestrator over N per-camera BoTSORT instances sharing one
    GTA. Does not implement any matching itself - all of it is delegated to
    BoTSORT.update() and GTA.step()."""

    def __init__(self, registry: GTA, trackers: Optional[dict] = None):
        self.registry = registry
        self.trackers: dict = dict(trackers) if trackers else {}
        # cam_id -> tracks that were just tied to a global_id on the most
        # recent update_camera() call for that camera (registry.step()'s
        # return value, see global_tracklet_association.py) - a fresh
        # registration or a re-id match, not the common "already identified"
        # case. Side channel rather than a return-value change so
        # update_camera()'s existing list[STrack] return stays backward
        # compatible.
        self.last_assigned: dict = {}

    def add_camera(self, cam_id, tracker: BoTSORT):
        """Register a per-camera BoTSORT instance. Caller is responsible for
        constructing it with a distinct cam_source and registry=self.registry."""
        self.trackers[cam_id] = tracker

    def update_camera(self, cam_id, detections, frame_id: int) -> list:
        """Update a single camera and run the registry step for it.

        Use this for asynchronous/per-source pipelines where not every
        camera has a new frame on every tick (e.g. multiple RTSP sources
        batched together) - call once per camera that has a frame this
        tick, with that camera's own frame_id (e.g. frame_meta.frame_num).
        """
        tracker = self.trackers[cam_id]
        tracks = tracker.update(detections)
        self.last_assigned[cam_id] = self.registry.step(tracker, frame_id)
        return tracks

    def update_batch(self, detections_by_cam: dict, frame_id: int) -> dict:
        """Update all registered cameras for one synchronous tick, where
        every camera's detections arrive together and share one frame_id."""
        return {
            cam_id: self.update_camera(cam_id, detections_by_cam.get(cam_id, []), frame_id)
            for cam_id in self.trackers
        }
