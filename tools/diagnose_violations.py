"""Record where one global id is held by >1 live tracklet at once (same batch)."""
import os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import DSPipeline, DEFAULT_CONFIG
from global_tracker import GlobalTracker

import yaml
_rois = {i: s["roi"] for i, s in enumerate(yaml.safe_load(open(DEFAULT_CONFIG))["sources"]) if s.get("roi")}
gt = GlobalTracker(rois=_rois)
allev = []
pairs = collections.Counter()
examples = {}
N = int(os.environ.get("SECS", "25"))
import time; t0 = time.monotonic()

def cb(frames):
    live = gt.update(frames)
    allev.extend(gt.drain_events())
    holders = collections.defaultdict(list)
    for sid, tls in live.items():
        for t in tls:
            if t.t_global_id:
                holders[t.t_global_id].append(t)
    for gid, ts in holders.items():
        if len(ts) > 1:
            k = (gid, tuple(sorted(t.key for t in ts)))
            pairs[k] += 1
            examples.setdefault(k, [(t.key, [int(x) for x in t.tlwh], t.t_identity_since_frame,
                                     t.first_frame, t.tracklet_len) for t in ts])
    if time.monotonic() - t0 > N:
        p.loop.quit()

p = DSPipeline(DEFAULT_CONFIG, on_tracks=cb, sink_sync=False)
p.run()
for k, n in pairs.most_common(12):
    print(k, "batches:", n)
    for e in examples[k]:
        print("    ", e)
print("distinct violating combos:", len(pairs))
same_cam = sum(1 for (g, ks) in pairs if len({c for c, _ in ks}) < len(ks))
print("combos with two tracklets in the SAME camera:", same_cam)

for (gid, keys) in pairs:
    print("events for gid", gid)
    for e in allev:
        if e.get("global_id") == gid or e.get("old_global_id") == gid or any(list(k) in [list(x) for x in e.get("keys", [])] for k in keys):
            print("   ", {k: v for k, v in e.items()})
