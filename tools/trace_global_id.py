"""Record everything the pipeline knows about ONE global id, frame by frame, for its whole life.

    python3 tools/trace_global_id.py --gid 1 --out traces/gid1            # pipeline + annotated mp4 + trace
    python3 tools/trace_global_id.py --gid 1 --out traces/gid1 --no-video  # faster, no video

It is an OBSERVER. The real run.py is executed unchanged; this script only wraps (monkey-patches, in this
process) the calls where the interesting numbers are produced, copies them, and passes everything through:

    BotSortCore.update        BoT-SORT stage: current embedding, smooth_feat before / after, neighbours
    TrackletStore._update_source   tracklet embedding (moving average) before this frame's update
    FastCommit.decide         association: fast-commit decisions (reason, distance, margin)
    GlobalEntry.observe       association: sample-to-entry distance, accepted / rejected, mismatch_streak
    Associator._emit          association: fast_commit / new_identity / revoked events
    GlobalTracker.update      adds an on_frame hook, called after every frame is fully processed

Output  <out>_trace.pkl : {"meta", "rows", "vec", "entry_vecs", "events"}   (load with pickle; see gmm_probe.ipynb)
        <out>_run/      : annotated mp4 (frame numbers + g<id> labels) and the usual csv / features of the same run

One row = one frame in which a tracklet wore the id (or took part in a decision / observation about it).
Never import this inside a notebook kernel: it imports DeepStream (gi / pyds), after which matplotlib aborts.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import pickle
import sys
import traceback
from collections import defaultdict, deque

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

NAN = float("nan")


def cos(a, b):
    return NAN if a is None or b is None else float(np.dot(a, b))


def f16(v, d):
    return np.full(d, np.nan, np.float16) if v is None else np.asarray(v, np.float32).astype(np.float16)


class Recorder:
    VEC_NAMES = ("curr", "ema_pre", "ema_post", "last", "win_mean", "nb1", "nb2")

    def __init__(self, gid: int):
        self.gid = gid
        self.rows, self.events = [], []
        self.vecs = defaultdict(list)
        self.entry_vecs, self._ver_key = [], None
        self.bs_q, self.pre_q = defaultdict(deque), defaultdict(deque)     # per camera, FIFO (same frame order)
        self.obs, self.dec = [], []
        self.errors, self.error_example = 0, None
        self.gt = None
        self._new_events = []
        self.botsort_params = None
        self.dim = None

    # ------------------------------------------------------------------ observers (called from the wrappers)
    def after_botsort(self, core, cam, tr, pre, out):
        from tracker.botsort_backend import _iou
        self.botsort_params = dict(core.params)
        by_id = {t.track_id: t for t in tr.tracked_stracks}
        info = {}
        if out:
            xyxy = np.array([[o.tlwh[0], o.tlwh[1], o.tlwh[0] + o.tlwh[2], o.tlwh[1] + o.tlwh[3]] for o in out], np.float32)
            iou = _iou(xyxy, xyxy)
            np.fill_diagonal(iou, 0.0)
        for i, o in enumerate(out):
            st = by_id.get(o.track_id)
            post = None if st is None or st.smooth_feat is None else st.smooth_feat
            nbs = []
            for j in np.argsort(-iou[i])[:2]:
                if iou[i, j] > 0:
                    sj = by_id.get(out[j].track_id)
                    nbs.append(dict(id=out[j].track_id, iou=float(iou[i, j]),
                                    vec=None if sj is None or sj.curr_feat is None else np.array(sj.curr_feat, np.float32),
                                    fresh=out[j].feat is not None))
            info[o.track_id] = dict(score=o.score, curr=o.feat, fresh=o.feat is not None,
                                    cos_pre=cos(pre.get(o.track_id), o.feat), cos_post=cos(post, o.feat),
                                    max_iou=float(iou[i].max()) if len(out) > 1 else 0.0, nbs=nbs)
        self.bs_q[cam].append(info)

    def on_frame(self, gt, ft, live):
        self.gt = gt
        cam, fn = ft.info.source_id, ft.info.frame_num
        new_events, self._new_events = self._new_events, []              # events emitted while this frame was processed
        bs = self.bs_q[cam].popleft() if self.bs_q[cam] else {}
        pre = self.pre_q[cam].popleft() if self.pre_q[cam] else {}
        obs, dec = self.obs, self.dec
        self.obs, self.dec = [], []
        entry = gt.assoc.entry(self.gid)
        if entry is None:
            return
        gid_of = {t.track_id: t.t_global_id for t in live}
        involved = {o["key"] for o in obs} | {d["key"] for d in dec if d["gid"] == self.gid or d["cand"] == self.gid}
        group = [t for t in live if t.t_global_id == self.gid or t.key in involved]
        if not group:
            return
        if entry.embedding.count != self._ver_key:                                  # entry vector: stored once per change
            self.entry_vecs.append(np.array(entry.embedding.value, np.float32))
            self._ver_key = entry.embedding.count
        ver = len(self.entry_vecs) - 1
        for tl in group:
            info = bs.get(tl.track_id, {})
            ema_pre, ema_post = pre.get(tl.key), tl.embedding.value
            curr = info.get("curr")
            self.dim = self.dim or (len(ema_post) if ema_post is not None else None)
            d = self.dim or 1280
            # brute-force nearest entries by the tracklet's moving average (does not touch the FAISS index)
            ranked = []
            if ema_post is not None:
                ranked = sorted(((g, 1.0 - float(np.dot(e.embedding.value, ema_post)))
                                 for g, e in gt.assoc.gallery.entries.items() if not e.embedding.empty), key=lambda x: x[1])[:3]
            ranked += [(None, NAN)] * (3 - len(ranked))
            o = next((x for x in obs if x["key"] == tl.key), None)
            dc = next((x for x in dec if x["key"] == tl.key), None)
            nbs = info.get("nbs", []) + [None, None]
            ev = [e for e in new_events if e.get("track_id") == tl.track_id and e.get("source") == cam]
            row = dict(
                gid=self.gid, cam=cam, frame_num=fn, pts=ft.info.pts_s, tl_frame=int(round(ft.info.pts_s * gt.frame_rate)),
                track_id=tl.track_id, wears=tl.t_global_id == self.gid, tracklet_gid=tl.t_global_id,
                x=float(tl.tlwh[0]), y=float(tl.tlwh[1]), w=float(tl.tlwh[2]), h=float(tl.tlwh[3]),
                score=info.get("score", NAN), fresh=info.get("fresh", False),
                cos_pre_smooth=info.get("cos_pre", NAN), cos_post_smooth=info.get("cos_post", NAN),
                max_iou_other=info.get("max_iou", NAN),
                nb1_id=nbs[0]["id"] if nbs[0] else None, nb1_iou=nbs[0]["iou"] if nbs[0] else NAN,
                nb1_gid=gid_of.get(nbs[0]["id"]) if nbs[0] else None, nb1_fresh=nbs[0]["fresh"] if nbs[0] else None,
                nb2_id=nbs[1]["id"] if nbs[1] else None, nb2_iou=nbs[1]["iou"] if nbs[1] else NAN,
                nb2_gid=gid_of.get(nbs[1]["id"]) if nbs[1] else None, nb2_fresh=nbs[1]["fresh"] if nbs[1] else None,
                clean=tl.clean, in_roi=tl.in_roi, edge=tl.is_touching_edge, peer_iou=tl.peer_iou,
                tracklet_len=tl.tracklet_len, assoc_status=tl.assoc_status.value if tl.assoc_status else None,
                new_emb=tl.new_embedding_this_frame, emb_count=tl.embedding.count,
                d_ema_pre_curr=1.0 - cos(ema_pre, curr) if ema_pre is not None and curr is not None else NAN,
                d_ema_post_entry=entry.embedding.distance(ema_post) if ema_post is not None else NAN,
                obs_d=o["d"] if o else NAN, obs_accepted=o["accepted"] if o else None,
                mismatch_streak=entry.mismatch_streak, entry_count=entry.embedding.count,
                entry_wearer=str(entry.active_key), entry_last_cam=entry.last_cam, entry_ver=ver,
                dec_reason=dc["reason"] if dc else None, dec_dist=dc["dist"] if dc else NAN,
                dec_margin=dc["margin"] if dc else NAN, dec_gid=dc["gid"] if dc else None, dec_cand=dc["cand"] if dc else None,
                top1_gid=ranked[0][0], top1_d=ranked[0][1], top2_gid=ranked[1][0], top2_d=ranked[1][1],
                top3_gid=ranked[2][0], top3_d=ranked[2][1],
                events="|".join(e["event"] + ("(forced)" if e.get("forced") else "") for e in ev),
            )
            self.rows.append(row)
            self.vecs["curr"].append(f16(curr, d))
            self.vecs["ema_pre"].append(f16(ema_pre, d))
            self.vecs["ema_post"].append(f16(ema_post, d))
            self.vecs["last"].append(f16(tl.embedding.last, d))
            self.vecs["win_mean"].append(f16(tl.embedding.window_mean(), d))
            self.vecs["nb1"].append(f16(nbs[0]["vec"] if nbs[0] else None, d))
            self.vecs["nb2"].append(f16(nbs[1]["vec"] if nbs[1] else None, d))


    # ------------------------------------------------------------------ output
    def save(self, path, extra_meta):
        gt = self.gt
        entry = gt.assoc.entry(self.gid) if gt else None
        meta = dict(extra_meta, gid=self.gid, n_rows=len(self.rows), dim=self.dim,
                    observer_errors=self.errors, observer_error_example=self.error_example,
                    botsort_params=self.botsort_params,
                    associator_cfg=dataclasses.asdict(gt.assoc.inner.cfg if hasattr(gt.assoc, "inner") else gt.assoc.cfg) if gt else None,
                    wearers={f"{k[0]}:{k[1]}": dict(first=m.first_frame, last=m.last_frame, cam=m.cam)
                             for k, m in (entry.members.items() if entry else [])},
                    final_gallery_size=gt.assoc.size() if gt else None)
        out = dict(meta=meta, rows=self.rows,
                   vec={n: (np.stack(v) if v else np.zeros((0, self.dim or 1280), np.float16)) for n, v in self.vecs.items()},
                   entry_vecs=(np.stack(self.entry_vecs).astype(np.float16) if self.entry_vecs else np.zeros((0, self.dim or 1280), np.float16)),
                   events=self.events)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        return meta


def safe(rec, fn):
    """Run an observer; an exception in it must never reach the pipeline."""
    def run(*a, **k):
        try:
            return fn(*a, **k)
        except Exception:
            rec.errors += 1
            rec.error_example = rec.error_example or traceback.format_exc()
    return run


def install(rec: Recorder):
    from association.associator import Associator
    from association.fast_commit import FastCommit
    from association.global_entry import GlobalEntry
    from global_tracker import GlobalTracker
    from tracker.botsort_backend import BotSortCore
    from tracker.tracklet import TrackletStore

    o_core = BotSortCore.update

    def core_update(self, cam, boxes, scores, feats):
        tr = self._tracker(cam)
        pre = {}
        try:
            pre = {t.track_id: (None if t.smooth_feat is None else np.array(t.smooth_feat, np.float32))
                   for t in tr.tracked_stracks + tr.lost_stracks}
        except Exception:
            rec.errors += 1
        out = o_core(self, cam, boxes, scores, feats)
        safe(rec, rec.after_botsort)(self, cam, tr, pre, out)
        return out
    BotSortCore.update = core_update

    o_us = TrackletStore._update_source

    def update_source(self, sid, tracks, frame, frame_num):
        pre = {k: (None if tl.embedding.value is None else tl.embedding.value.copy())
               for k, tl in self.tracklets.items() if k[0] == sid}
        r = o_us(self, sid, tracks, frame, frame_num)
        rec.pre_q[sid].append(pre)
        return r
    TrackletStore._update_source = update_source

    o_dec = FastCommit.decide

    def decide(self, query, key, frame, cam=None, live_keys=None):
        d = o_dec(self, query, key, frame, cam, live_keys)
        rec.dec.append(dict(key=key, frame=frame, reason=d.reason, gid=d.gid, cand=d.candidate, dist=d.dist, margin=d.margin))
        return d
    FastCommit.decide = decide

    o_obs = GlobalEntry.observe

    def observe(self, vec, outlier_thresh, damping=0.0):
        dist = self.embedding.distance(vec)
        key = self.active_key
        res = o_obs(self, vec, outlier_thresh, damping)
        if self.global_id == rec.gid:
            rec.obs.append(dict(key=key, d=dist, accepted=res, streak=self.mismatch_streak))
        return res
    GlobalEntry.observe = observe

    o_emit = Associator._emit

    def emit(self, kind, frame, t, **fields):
        o_emit(self, kind, frame, t, **fields)
        ev = self.events[-1]
        if fields.get("global_id") == rec.gid or t.t_global_id == rec.gid:
            rec.events.append(dict(ev))
            rec._new_events.append(dict(ev))
    Associator._emit = emit

    o_upd = GlobalTracker.update

    def gt_update(self, frames, on_frame=None):
        def both(ft, live):
            if on_frame is not None:
                on_frame(ft, live)
            safe(rec, rec.on_frame)(self, ft, live)
        return o_upd(self, frames, on_frame=both)
    GlobalTracker.update = gt_update


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gid", type=int, default=1)
    ap.add_argument("--out", default="traces/gid1", help="prefix: <out>_trace.pkl and the run folder <out>_run/")
    ap.add_argument("--no-video", action="store_true", help="skip the annotated mp4 (faster)")
    ap.add_argument("--max-seconds", type=float, default=0)
    args = ap.parse_args()

    rec = Recorder(args.gid)
    install(rec)

    import run                                             # the real entry point, unchanged
    run_dir = args.out + "_run"
    argv = ["run.py", "--record", run_dir, "--no-tracklet-images"] if not args.no_video else ["run.py", "--global", "--fast"]
    if args.max_seconds:
        argv += ["--max-seconds", str(args.max_seconds)]
    sys.argv = argv
    run.main()

    import yaml
    from pipeline import DEFAULT_CONFIG
    with open(DEFAULT_CONFIG) as f:
        sources = yaml.safe_load(f)["sources"]
    path = args.out + "_trace.pkl"
    meta = rec.save(path, dict(sources=sources, video_dir=None if args.no_video else run_dir, argv=argv))
    print(f"\n[trace] gid {args.gid}: {meta['n_rows']} rows, {len(rec.events)} events, "
          f"{len(meta['wearers'])} wearer tracklets, observer errors: {meta['observer_errors']}  ->  {path}")
    if meta["observer_errors"]:
        print(meta["observer_error_example"])


if __name__ == "__main__":
    main()
