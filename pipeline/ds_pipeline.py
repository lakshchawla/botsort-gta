"""Raw DeepStream pipeline (not an app):

    uridecodebin x N -> nvstreammux -> nvinfer(PGIE) -> nvinfer(SGIE, ReID) -> [probe: BoT-SORT] -> sink

Consumers attach a callback that receives tracks per batch; the pipeline owns
no application logic.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import gi
gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst
import pyds
import yaml

from tracker import FrameTracks, UNTRACKED_ID
from tracker.botsort_backend import BotSortBackend

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
DEFAULT_CONFIG = os.path.join(CONFIG_DIR, "pipeline.yml")

TrackCallback = Callable[[List[FrameTracks]], None]
CLEAN_CLASS_ID = 1          # person
UNCLEAN_SENTINEL_CLASS_ID = 99


@dataclass
class Label:
    text: str
    clean: bool


LabelFn = Callable[[int, int], Optional[Label]]   # (source_id, track_id) -> OSD label


def _cb_newpad(decodebin, pad, source_bin):
    caps = pad.get_current_caps() or pad.query_caps(None)
    if "video" not in caps.get_structure(0).get_name():
        return
    if not caps.get_features(0).contains("memory:NVMM"):
        sys.stderr.write("Decodebin did not pick an NVIDIA decoder plugin.\n")
        return
    if not source_bin.get_static_pad("src").set_target(pad):
        sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")


def _child_added(child_proxy, obj, name, user_data):
    if "decodebin" in name:
        obj.connect("child-added", _child_added, user_data)
    if "source" in name:
        if obj.find_property("protocols"):
            obj.set_property("protocols", 4)
        if obj.find_property("latency"):
            obj.set_property("latency", 300)
        if obj.find_property("drop-on-latency"):
            obj.set_property("drop-on-latency", True)


def _create_source_bin(index: int, uri: str) -> Gst.Bin:
    nbin = Gst.Bin.new(f"source-bin-{index:02d}")
    dec = Gst.ElementFactory.make("uridecodebin", f"uri-decode-bin-{index}")
    if not nbin or not dec:
        raise RuntimeError("Could not create source bin elements")
    dec.set_property("uri", uri)
    dec.connect("pad-added", _cb_newpad, nbin)
    dec.connect("child-added", _child_added, nbin)
    nbin.add(dec)
    if not nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC)):
        raise RuntimeError("Failed to add ghost pad to source bin")
    return nbin


class DSPipeline:
    def __init__(self, config_path: str = DEFAULT_CONFIG,
                 on_tracks: Optional[TrackCallback] = None,
                 sink_type: Optional[str] = None,
                 label_fn: Optional[LabelFn] = None,
                 sink_sync: Optional[bool] = None,
                 snapshot_dir: Optional[str] = None, snapshot_every: int = 150,
                 video_path: Optional[str] = None, frame_overlay: bool = False,
                 tile_width: Optional[int] = None, video_bitrate: int = 12_000_000):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.config_dir = os.path.dirname(os.path.abspath(config_path))
        self.on_tracks = on_tracks
        self.label_fn = label_fn
        self.sink_sync = sink_sync
        self.snapshot_dir = snapshot_dir
        self.snapshot_every = snapshot_every
        self.video_path = video_path
        self.frame_overlay = frame_overlay
        self.tile_width = tile_width
        self.video_bitrate = video_bitrate
        self._tile = None            # (tile_w, tile_h, rows, cols) once the tiler is configured
        self.sources = self.cfg["sources"]
        self.pgie_id = self.cfg["pgie"].get("unique-id", 1)
        self.tracker = BotSortBackend(self.cfg["tracker"], self.config_dir)
        self.sink_type = sink_type or self.cfg.get("sink", {}).get("type", "fakesink")
        self.loop = None
        self.pipeline = None

    def _resolve(self, p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(self.config_dir, p)

    def build(self) -> Gst.Pipeline:
        Gst.init(None)
        n = len(self.sources)
        self.pipeline = pipeline = Gst.Pipeline.new("ds-pipeline")

        mux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
        pipeline.add(mux)
        mcfg = self.cfg.get("streammux", {})
        for key in ("width", "height", "batched-push-timeout"):
            if key in mcfg:
                mux.set_property(key, mcfg[key])
        # One frame per source per batch (so the tiler shows every frame of every camera exactly once).
        # The model's own batch size (pgie batch-size: 4, b4 engine) is independent of this.
        mux.set_property("batch-size", n)
        mux.set_property("live-source", any(s["uri"].startswith("rtsp://") for s in self.sources))

        for i, src in enumerate(self.sources):
            sb = _create_source_bin(i, src["uri"])
            pipeline.add(sb)
            if sb.get_static_pad("src").link(mux.request_pad_simple(f"sink_{i}")) != Gst.PadLinkReturn.OK:
                raise RuntimeError(f"Failed to link source {i} to streammux")

        pgie = Gst.ElementFactory.make("nvinfer", "pgie")
        pgie.set_property("config-file-path", self._resolve(self.cfg["pgie"]["config-file"]))
        tracker_elements = self.tracker.elements()

        scfg = self.cfg.get("sink", {})
        if self.sink_type == "snapshot":
            os.makedirs(self.snapshot_dir, exist_ok=True)
            sink = Gst.ElementFactory.make("multifilesink", "sink")
            sink.set_property("location", os.path.join(self.snapshot_dir, "tile_%05d.jpg"))
            sink.set_property("async", False)
        elif self.sink_type == "video":
            os.makedirs(os.path.dirname(os.path.abspath(self.video_path)), exist_ok=True)
            sink = Gst.ElementFactory.make("filesink", "sink")
            sink.set_property("location", self.video_path)
            sink.set_property("async", False)
        elif self.sink_type == "display":
            sink = Gst.ElementFactory.make("nveglglessink", "sink")
        else:
            sink = Gst.ElementFactory.make("fakesink", "sink")
        if sink is None:
            raise RuntimeError(f"Could not create sink '{self.sink_type}'")
        sink.set_property("sync", bool(scfg.get("sync", False)) if self.sink_sync is None else self.sink_sync)

        chain = [pgie] + tracker_elements
        if self.sink_type in ("display", "snapshot", "video"):
            conv = Gst.ElementFactory.make("nvvideoconvert", "conv")
            osd = Gst.ElementFactory.make("nvdsosd", "osd")
            tiler = Gst.ElementFactory.make("nvmultistreamtiler", "tiler")
            rows, cols = scfg.get("grid", [2, 2])       # fixed grid; unused tiles stay blank
            if rows * cols < n:
                raise RuntimeError(f"grid {rows}x{cols} too small for {n} sources")
            # Each tile is strictly 16:9: the tiler canvas is cols x rows tiles.
            tile_w = int(self.tile_width or scfg.get("tile-width", 960))
            tile_h = tile_w * 9 // 16
            self._tile = (tile_w, tile_h, rows, cols)
            tiler.set_property("rows", rows)
            tiler.set_property("columns", cols)
            tiler.set_property("width", tile_w * cols)
            tiler.set_property("height", tile_h * rows)
            chain += [tiler, conv, osd]
            if self.sink_type == "video":
                conv2 = Gst.ElementFactory.make("nvvideoconvert", "conv2")
                caps = Gst.ElementFactory.make("capsfilter", "caps2")
                caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12"))
                enc = Gst.ElementFactory.make("nvv4l2h264enc", "h264enc")
                enc.set_property("bitrate", self.video_bitrate)
                enc.set_property("iframeinterval", 30)
                parse = Gst.ElementFactory.make("h264parse", "h264parse")
                container = Gst.ElementFactory.make("qtmux", "mp4mux")
                if not all((conv2, caps, enc, parse, container)):
                    raise RuntimeError("Could not create the H.264/MP4 encoder chain")
                chain += [conv2, caps, enc, parse, container]
            elif self.sink_type == "snapshot":
                conv2 = Gst.ElementFactory.make("nvvideoconvert", "conv2")
                caps = Gst.ElementFactory.make("capsfilter", "caps2")
                caps.set_property("caps", Gst.Caps.from_string("video/x-raw, format=I420"))
                enc = Gst.ElementFactory.make("jpegenc", "jpegenc")
                chain += [conv2, caps, enc]
        chain.append(sink)

        for el in chain:
            pipeline.add(el)
        mux.link(chain[0])
        for a, b in zip(chain, chain[1:]):
            a.link(b)

        if self.sink_type == "snapshot":     # keep only every Nth frame
            cnt = [0]
            def keep_nth(pad, info, _):
                cnt[0] += 1
                return Gst.PadProbeReturn.OK if cnt[0] % self.snapshot_every == 0 else Gst.PadProbeReturn.DROP
            sink.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, keep_nth, None)

        # last tracker-side element's src pad: the detections / track ids are final here.
        tracker_elements[-1].get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, self._probe, None)

        self.loop = GLib.MainLoop()
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._bus_call)
        return pipeline

    def _probe(self, pad, info, _):
        buf = info.get_buffer()
        if not buf:
            return Gst.PadProbeReturn.OK
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        if self.on_tracks:
            self.on_tracks(self.tracker.process(batch_meta, self.pgie_id))
        if self.label_fn:
            self._apply_labels(batch_meta)
        if self.frame_overlay:
            self._apply_frame_overlay(batch_meta)
        return Gst.PadProbeReturn.OK

    def _apply_frame_overlay(self, batch_meta):
        """'cam<N> f<frame_num>' at the top-left of every tile (frame_num = that source's own
        decoder frame counter, the same number the CSV uses)."""
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                fm = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break
            dm = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            dm.num_labels = 1
            tp = dm.text_params[0]
            tp.display_text = f"cam{fm.source_id + 1}  f{fm.frame_num}"
            tp.x_offset, tp.y_offset = 10, 8
            tp.font_params.font_name = "Serif"
            tp.font_params.font_size = 18
            tp.font_params.font_color.set(1.0, 1.0, 0.0, 1.0)
            tp.set_bg_clr = 1
            tp.text_bg_clr.set(0.0, 0.0, 0.0, 0.8)
            pyds.nvds_add_display_meta_to_frame(fm, dm)
            l_frame = l_frame.next

    def _apply_labels(self, batch_meta):
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                fm = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break
            l_obj = fm.obj_meta_list
            while l_obj is not None:
                try:
                    om = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break
                if om.object_id != UNTRACKED_ID:
                    lab = self.label_fn(fm.source_id, om.object_id)
                    if lab is not None:
                        self._style(om, lab)
                else:                                           # a detection that is not (yet) part of any track
                    om.rect_params.border_width = 0
                    om.text_params.display_text = ""
                l_obj = l_obj.next
            l_frame = l_frame.next

    @staticmethod
    def _style(om, lab: Label):
        """Clean persons: cyan box + 'g<global> t<local>'. Unclean ones (outside ROI / occluded /
        on the edge) are id-mutated to a sentinel class and drawn faint, as in the old pipeline."""
        rp, tp = om.rect_params, om.text_params
        rp.has_bg_color = 0
        if lab.clean:
            om.class_id = CLEAN_CLASS_ID
            rp.border_color.set(0.0, 1.0, 1.0, 1.0)
            rp.border_width = 2
        else:
            om.class_id = UNCLEAN_SENTINEL_CLASS_ID
            rp.border_color.set(1.0, 1.0, 1.0, 0.25)
            rp.border_width = 1
        tp.display_text = lab.text
        tp.font_params.font_name = "Serif"
        tp.font_params.font_size = 10
        tp.font_params.font_color.set(1.0, 1.0, 1.0, 1.0 if lab.clean else 0.5)
        tp.set_bg_clr = 1
        tp.text_bg_clr.set(0.0, 0.0, 0.0, 0.7)

    def finish(self):
        """Graceful stop: send EOS so the encoder/muxer finalise the file; the EOS message on the
        bus then ends the main loop."""
        self.pipeline.send_event(Gst.Event.new_eos())

    def _bus_call(self, bus, msg):
        t = msg.type
        if t == Gst.MessageType.EOS:
            sys.stdout.write("End of stream\n")
            self.loop.quit()
        elif t == Gst.MessageType.WARNING:
            err, _ = msg.parse_warning()
            sys.stderr.write(f"WARNING {msg.src.get_name()}: {err.message}\n")
        elif t == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            sys.stderr.write(f"ERROR {msg.src.get_name()}: {err.message}\n{dbg or ''}\n")
            self.loop.quit()
        return True

    def run(self):
        if self.pipeline is None:
            self.build()
        self.pipeline.set_state(Gst.State.PLAYING)
        try:
            self.loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            self.pipeline.set_state(Gst.State.NULL)
