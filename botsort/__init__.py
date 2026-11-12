"""Entry point of the repo: `import botsort` puts the repo root on sys.path and exposes the tested
global tracker (per-camera BoT-SORT tracklets -> cross-camera global ids).

    from botsort import GlobalTracker, DSPipeline
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from global_tracker import GlobalTracker  # noqa: E402

try:                                        # needs GStreamer / DeepStream (pyds)
    from pipeline import DSPipeline, DEFAULT_CONFIG  # noqa: E402,F401
except ImportError:
    DSPipeline = DEFAULT_CONFIG = None

__all__ = ["GlobalTracker", "DSPipeline", "DEFAULT_CONFIG"]
