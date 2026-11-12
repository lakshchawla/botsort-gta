"""FaissGallery: FAISS inner-product index over the global_entry embeddings (one row per entry)."""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import faiss
import numpy as np

from .global_entry import GlobalEntry

log = logging.getLogger("association.gallery")


class FaissGallery:
    def __init__(self, use_gpu: bool = False, views_in_search: bool = False):
        self.entries: Dict[int, GlobalEntry] = {}
        self.views_in_search = views_in_search           # also index every entry's varied_views
        self._use_gpu = use_gpu
        self._index = None
        self._dim = None
        self._row_gid: List[int] = []
        self._dirty = False

    def __len__(self) -> int:
        return len(self.entries)

    def add(self, entry: GlobalEntry) -> None:
        self.entries[entry.global_id] = entry
        self._dirty = True

    def mark_dirty(self) -> None:
        """An entry's embedding changed: the index is rebuilt lazily on the next search."""
        self._dirty = True

    def search(self, vec, k: int = 2) -> List[Tuple[int, float]]:
        """[(global_id, cosine_distance)] best first, at most k entries (an entry's best row counts)."""
        self._sync()
        if self._index is None or self._index.ntotal == 0:
            return []
        q = np.ascontiguousarray(np.asarray(vec, dtype=np.float32).reshape(1, -1))
        rows = self._index.ntotal if self.views_in_search else min(k, self._index.ntotal)
        sims, idxs = self._index.search(q, rows)
        best: Dict[int, float] = {}
        for s, i in zip(sims[0], idxs[0]):
            if i >= 0:
                best.setdefault(self._row_gid[int(i)], 1.0 - float(s))          # rows come best first
        return sorted(best.items(), key=lambda kv: kv[1])[:k]

    def _sync(self) -> None:
        if not self._dirty:
            return
        self._dirty = False
        gids = [g for g, e in self.entries.items() if not e.embedding.empty]
        if not gids:
            self._index, self._row_gid = None, []
            return
        rows, row_gid = [], []
        for g in gids:
            e = self.entries[g]
            rows.append(e.embedding.value); row_gid.append(g)
            if self.views_in_search:
                for view in e.varied_views:
                    rows.append(view); row_gid.append(g)
        mat = np.ascontiguousarray(np.stack(rows).astype(np.float32))
        if self._index is None or self._dim != mat.shape[1]:
            idx = faiss.IndexFlatIP(mat.shape[1])
            if self._use_gpu:
                try:
                    idx = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, idx)
                except Exception as exc:
                    log.warning("GPU FAISS unavailable (%s); using CPU index", exc)
            self._index, self._dim = idx, mat.shape[1]
        else:
            self._index.reset()
        self._index.add(mat)
        self._row_gid = row_gid
