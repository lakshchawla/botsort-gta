"""Import the upstream BoT-SORT repo (https://github.com/NirAharon/BoT-SORT) without modifying it.

Three things stand in the way of a plain `import tracker.bot_sort`:
  * the repo's package is called `tracker`, exactly like OUR package (and has no __init__.py, so Python would
    prefer ours): while the repo is imported, an explicit package object pointing at its folder is registered
    under that name; afterwards it is renamed to `botsort_repo` and ours is put back;
  * bot_sort.py imports the FastReID interface (torch, fastreid ...) at module level: a stub takes its
    place, because here the embeddings come from the DeepStream SGIE, not from the repo's encoder;
  * bot_sort.py / gmc.py import matplotlib.pyplot at module level (unused, or only for a debug plot), and loading
    matplotlib's C extension after the GStreamer / DeepStream libraries aborts the process: a stub is used for
    the duration of the import;
  * the repo uses np.float / np.int / np.bool, removed in numpy >= 1.24: aliases are restored.
"""
from __future__ import annotations

import importlib
import os
import sys
import types
from typing import Tuple

import numpy as np

_CACHE: dict = {}


class InjectedReID:
    """Stands in for FastReIDInterface. BoTSORT.update() calls `encoder.inference(img, dets)`; this returns
    the embeddings that were computed by the SGIE for exactly those detections (looked up by box bytes)."""

    def __init__(self, *args, **kwargs):
        self._by_box: dict = {}

    def set(self, boxes_xyxy: np.ndarray, feats: np.ndarray) -> None:
        self._by_box = {np.ascontiguousarray(b, dtype=np.float32).tobytes(): f for b, f in zip(boxes_xyxy, feats)}

    def inference(self, img, dets):
        return [self._by_box[np.ascontiguousarray(d, dtype=np.float32).tobytes()] for d in dets]


def load_botsort(repo_root: str) -> Tuple[types.ModuleType, types.ModuleType, types.ModuleType]:
    """Returns (bot_sort module, basetrack module, matching module) of the upstream repo."""
    repo_root = os.path.abspath(os.path.expanduser(repo_root))
    if repo_root in _CACHE:
        return _CACHE[repo_root]
    if not os.path.isfile(os.path.join(repo_root, "tracker", "bot_sort.py")):
        raise FileNotFoundError(f"{repo_root} does not look like a BoT-SORT checkout (tracker/bot_sort.py missing)")

    for alias, builtin in (("float", float), ("int", int), ("bool", bool)):
        if alias not in vars(np):
            setattr(np, alias, builtin)

    ours = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "tracker" or k.startswith("tracker.")}
    stub_names = ("fast_reid", "fast_reid.fast_reid_interfece", "matplotlib", "matplotlib.pyplot")
    stubbed = {name: sys.modules.get(name) for name in stub_names}
    mpl = types.ModuleType("matplotlib")
    plt = types.ModuleType("matplotlib.pyplot")

    def _no_plotting(*a, **k):
        raise RuntimeError("matplotlib is stubbed while BoT-SORT is imported (debug plots are not available)")
    plt.figure = plt.imshow = plt.show = _no_plotting
    mpl.pyplot = plt
    sys.modules["matplotlib"], sys.modules["matplotlib.pyplot"] = mpl, plt
    fr = types.ModuleType("fast_reid")
    fri = types.ModuleType("fast_reid.fast_reid_interfece")
    fri.FastReIDInterface = InjectedReID
    fr.fast_reid_interfece = fri
    sys.modules["fast_reid"], sys.modules["fast_reid.fast_reid_interfece"] = fr, fri
    pkg = types.ModuleType("tracker")                   # the repo's folder, as a package called `tracker`
    pkg.__path__ = [os.path.join(repo_root, "tracker")]
    pkg.__package__ = "tracker"
    sys.modules["tracker"] = pkg
    try:
        bot_sort = importlib.import_module("tracker.bot_sort")
        basetrack = importlib.import_module("tracker.basetrack")
        matching = importlib.import_module("tracker.matching")
    finally:
        for k in [k for k in sys.modules if k == "tracker" or k.startswith("tracker.")]:
            sys.modules["botsort_repo" + k[len("tracker"):]] = sys.modules.pop(k)
        sys.modules.update(ours)
        for name, old in stubbed.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
    _CACHE[repo_root] = (bot_sort, basetrack, matching)
    return _CACHE[repo_root]
