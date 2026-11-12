"""FastCommit: the stateless decision 'does this tracklet embedding belong to an existing entry?'

Commit to the nearest global_entry iff
  * its cosine distance d1 < thresh,
  * nobody else is wearing it right now (one body, one place),
  * the runner-up is at least `min_margin` further away (d2 - d1 >= min_margin; d2 = 1.0 if alone).
Reasons for NOT committing: empty | too_far | worn_elsewhere  -> the tracklet is novel (may mint);
                            ambiguous                         -> near-tie between entries (must NOT
                            mint: that is how duplicate entries snowball); `candidate` is the nearest.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .gallery import FaissGallery


@dataclass
class Decision:
    gid: Optional[int]       # chosen identity, None = no commit
    reason: str              # 'match' | 'empty' | 'too_far' | 'worn_elsewhere' | 'ambiguous'
    dist: float = 2.0
    margin: float = 0.0
    candidate: Optional[int] = None   # nearest entry (set for 'ambiguous')


class FastCommit:
    def __init__(self, gallery: FaissGallery, thresh: float, min_margin: float, co_tol: int):
        self.gallery = gallery
        self.thresh, self.min_margin, self.co_tol = thresh, min_margin, co_tol

    def decide(self, query, key: tuple, frame: int, cam=None, live_keys=None) -> Decision:
        hits = self.gallery.search(query, k=2)
        if not hits:
            return Decision(None, "empty")
        gid, d1 = hits[0]
        d2 = hits[1][1] if len(hits) > 1 else 1.0
        margin = d2 - d1
        if d1 >= self.thresh:
            return Decision(None, "too_far", d1, margin)
        if self.gallery.entries[gid].worn_by_other(key, frame, self.co_tol, cam, live_keys):
            return Decision(None, "worn_elsewhere", d1, margin)
        if margin < self.min_margin:
            return Decision(None, "ambiguous", d1, margin, candidate=gid)
        return Decision(gid, "match", d1, margin, candidate=gid)
