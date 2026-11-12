"""Run the DeepStream pipeline (PGIE -> ReID SGIE -> BoT-SORT).

    python3 run.py --global [--display] [--fast] [--max-seconds N] [--log-dir logs]
    python3 run.py --record annotation [--csv-interval 5] [--tile-width 960]   # video + CSV to annotate

Without --global: per-camera tracks only (counts). With --global: tracklets ->
GlobalTracker -> cross-camera global ids, JSONL event log, live invariant checks.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import DSPipeline, DEFAULT_CONFIG  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--global", dest="use_global", action="store_true", help="enable cross-camera global tracking")
    ap.add_argument("--display", action="store_true", help="render tiled output instead of fakesink")
    ap.add_argument("--snapshots", default=None, help="write the tiled OSD output as JPEGs to this dir (headless check)")
    ap.add_argument("--snapshot-every", type=int, default=150)
    ap.add_argument("--fast", action="store_true", help="don't pace to native fps (sink sync off)")
    ap.add_argument("--max-seconds", type=float, default=0, help="stop after N wall seconds (0 = until EOS)")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--record", default=None, metavar="DIR",
                    help="annotation run: save the tiled output (frame numbers on top) as MP4 and a "
                         "ground-truth CSV in DIR; implies --global --fast")
    ap.add_argument("--assoc", default=None, metavar="SPEC",
                    help="association logic: 'pkg.module:Class' or '/path/file.py:Class' (or a module/file with a "
                         "step() function). Default: the association block of global_tracker.yml")
    ap.add_argument("--assoc-config", default=None, help="yaml file (or leave unset) passed to the association class")
    ap.add_argument("--features-out", default=None, metavar="FILE.npy",
                    help="save per-tracklet features (smooth feat, last feat, cam, start/end frame) to this .npy")
    ap.add_argument("--tracklet-images", default=None, metavar="DIR",
                    help="where to save one picture per tracklet (<track_id>.jpg); default: tracklet_images/ "
                         "next to the features file")
    ap.add_argument("--no-tracklet-images", action="store_true", help="save the features without pictures")
    ap.add_argument("--csv-interval", type=int, default=5, help="CSV gets rows for frames where frame %% N == 0")
    ap.add_argument("--tile-width", type=int, default=None, help="per-camera tile width of the tiled video (16:9)")
    args = ap.parse_args()
    if args.record:
        args.use_global = args.fast = True

    gt = None
    labels = {}
    from pipeline.ds_pipeline import Label
    if args.use_global:
        from global_tracker import GlobalTracker
        import yaml
        with open(args.config) as f:
            rois = {i: src["roi"] for i, src in enumerate(yaml.safe_load(f)["sources"]) if src.get("roi")}
        gt = GlobalTracker(rois=rois, association=args.assoc, association_config=args.assoc_config)
        os.makedirs(args.log_dir, exist_ok=True)
        log_path = os.path.join(args.log_dir, f"events_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
        log_f = open(log_path, "w")

    csv_f = csv_w = None
    video_path = None
    if args.record:
        import csv
        os.makedirs(args.record, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(args.record, f"annotation_{stamp}.mp4")
        csv_path = os.path.join(args.record, f"annotation_{stamp}.csv")
        args.features_out = args.features_out or os.path.join(args.record, f"annotation_{stamp}_tracklet_features.npy")
        args.tracklet_images = args.tracklet_images or os.path.join(args.record, f"annotation_{stamp}_tracklet_images")
        csv_f = open(csv_path, "w", newline="")
        csv_w = csv.writer(csv_f)
        # tl = top-left, br = bottom-right (pixels in the 1920x1080 camera frame); frame = that camera's
        # own frame number (the number printed on the video tile); global_gt is left blank for you.
        csv_w.writerow(["frame", "tl_x", "tl_y", "br_x", "br_y", "cam_source", "tr_id", "global_pred", "global_gt"])
    n_rows = [0]

    def write_rows(ft, live):
        fn = ft.info.frame_num
        if csv_w is None or fn % args.csv_interval:
            return
        for t in live:
            x, y, w, h = t.tlwh
            csv_w.writerow([fn, round(x), round(y), round(x + w), round(y + h),
                            t.source_id, t.track_id, t.t_global_id, ""])
            n_rows[0] += 1

    seen = defaultdict(set)
    frames_n = defaultdict(int)
    state = {"t0": time.monotonic(), "last": time.monotonic(), "violations": 0, "same_cam": 0, "cross_cam": 0}
    pipe = None

    def on_tracks(frames):
        for ft in frames:
            sid = ft.info.source_id
            frames_n[sid] += 1
            seen[sid].update(t.track_id for t in ft.tracks)
        if gt is not None:
            live = gt.update(frames, on_frame=write_rows if csv_w else None)
            holders = defaultdict(list)                   # invariant: one gid, one live tracklet
            labels.clear()
            for sid, tls in live.items():
                for t in tls:
                    gid = f"g{t.t_global_id}" if t.t_global_id else "g?"
                    labels[(sid, t.track_id)] = Label(f"{gid} t{t.track_id}", t.clean)
                    if t.t_global_id:
                        holders[t.t_global_id].append(t)
            # violation = one gid worn by two different tracklets seen at the same instant (+-2 frames)
            for ts in holders.values():
                for i, a in enumerate(ts):
                    for b in ts[i + 1:]:
                        if a.key != b.key and abs(a.last_frame - b.last_frame) <= 2:
                            state["same_cam" if a.source_id == b.source_id else "cross_cam"] += 1
                            state.setdefault("examples", {}).setdefault((a.t_global_id, a.key, b.key), []).append(
                                (a.last_frame, b.last_frame, a.t_identity_since_frame, b.t_identity_since_frame))
            state["violations"] = state["same_cam"] + state["cross_cam"]
            for e in gt.drain_events() + pipe.tracker.drain_events():
                log_f.write(json.dumps(e, default=str) + "\n")
        now = time.monotonic()
        if now - state["last"] >= 2.0:
            state["last"] = now
            line = " ".join(f"src{s}:f={frames_n[s]},ids={len(seen[s])}" for s in sorted(frames_n))
            if gt is not None:
                line += f" | identities={gt.assoc.size()} pending={gt.assoc.n_pending()} violations={state['violations']}"
            print(line, flush=True)
        if args.max_seconds and now - state["t0"] > args.max_seconds:
            (pipe.finish if args.record else pipe.loop.quit)()

    label_fn = (lambda sid, tid: labels.get((sid, tid))) if gt is not None else None
    pipe = DSPipeline(args.config, on_tracks=on_tracks, sink_type="video" if args.record else ("snapshot" if args.snapshots else ("display" if args.display else None)),
                      snapshot_dir=args.snapshots, snapshot_every=args.snapshot_every,
                      video_path=video_path, frame_overlay=bool(args.record), tile_width=args.tile_width,
                      label_fn=label_fn, sink_sync=False if args.fast else None)
    if gt is not None and gt.store.left_grace_frames < pipe.tracker.left_grace_frames:
        print(f"[run] tracklets are kept for {pipe.tracker.left_grace_frames} frames after their last sighting "
              f"(BoT-SORT may bring the same id back that late; global_tracker.yml had "
              f"{gt.store.left_grace_frames})")
        gt.store.left_grace_frames = pipe.tracker.left_grace_frames
    pipe.run()

    if csv_f:
        csv_f.close()
        print(f"annotation video: {video_path}\nannotation csv:   {csv_path}  ({n_rows[0]} rows, every {args.csv_interval} frames)")
    print("frames:", {s: frames_n[s] for s in sorted(frames_n)}, "track ids:", {s: len(seen[s]) for s in sorted(seen)})
    if gt is not None:
        for e in gt.drain_events() + pipe.tracker.drain_events():
            log_f.write(json.dumps(e, default=str) + "\n")
        tls = gt.all_tracklets()
        for t in tls:
            log_f.write(json.dumps({"event": "tracklet_summary", **t.summary()}) + "\n")
        log_f.close()
        if args.features_out:
            import yaml
            with open(args.config) as f:
                sources = yaml.safe_load(f)["sources"]
            img_dir = None if args.no_tracklet_images else (
                args.tracklet_images or os.path.join(os.path.dirname(os.path.abspath(args.features_out)), "tracklet_images"))
            n = gt.save_features(args.features_out, sources=sources, images_dir=img_dir)
            print(f"tracklet features: {args.features_out}  ({n} tracklets)"
                  + (f"\ntracklet images:   {img_dir}" if img_dir else ""))
        per_cam = defaultdict(lambda: [0, 0])
        for t in tls:
            per_cam[t.source_id][0] += 1
            per_cam[t.source_id][1] += bool(t.t_global_id)
        cross = defaultdict(set)
        for t in tls:
            if t.t_global_id:
                cross[t.t_global_id].add(t.source_id)
        print("tracklets (total, with_gid) per cam:", dict(per_cam))
        print(f"global identities: {gt.assoc.size()}  spanning >1 camera: {sum(len(v) > 1 for v in cross.values())}")
        print(f"one gid on >1 simultaneous tracklet (tracklet-pair-frames): same camera={state['same_cam']} "
              f"(tracker fragments of one person), CROSS camera={state['cross_cam']}")
        for (gid, ka, kb), rows in list(state.get("examples", {}).items())[:4]:
            print(f"  violation gid={gid} {ka} vs {kb}: {len(rows)} frames, first={rows[0]} "
                  f"(last_a,last_b,since_a,since_b)")
        print("event log:", log_path)


if __name__ == "__main__":
    main()
