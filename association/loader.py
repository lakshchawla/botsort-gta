"""Load an association implementation from a spec, so logic can be swapped without touching the
pipeline.

spec (dict, or the string 'module_or_path[:Class]'):
    module: dotted import path          e.g. association.associator
    path:   a .py file                  e.g. /home/me/experiments/assoc_v2.py
    cls:    class to instantiate        e.g. Associator   (optional)
    config: yaml file path (relative to base_dir) or an inline dict, passed to the class
If `cls` is omitted the module itself must define a function `step(cam_id, live, frame)`
(optionally `release(key)`, `size()`, `n_pending()`); module-level globals hold its state.
The class may subclass AssociationBase or be any object with a step method.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from typing import Any, List, Optional, Union

import yaml

from .base import AssociationBase


class _ModuleAssociator(AssociationBase):
    """Wraps a module that only exposes functions."""

    def __init__(self, module):
        super().__init__()
        self._m = module

    def step(self, cam_id, live, frame):
        return self._m.step(cam_id, live, frame) or []

    def release(self, key):
        if hasattr(self._m, "release"):
            self._m.release(key)

    def size(self):
        return self._m.size() if hasattr(self._m, "size") else 0

    def n_pending(self):
        return self._m.n_pending() if hasattr(self._m, "n_pending") else 0


class _Compat(AssociationBase):
    """Gives any object with a step() method the full optional interface."""

    def __init__(self, obj):
        super().__init__()
        self._o = obj
        self.inner = obj

    def step(self, cam_id, live, frame):
        return self._o.step(cam_id, live, frame) or []

    def release(self, key):
        if hasattr(self._o, "release"):
            self._o.release(key)

    def size(self):
        return self._o.size() if hasattr(self._o, "size") else 0

    def n_pending(self):
        return self._o.n_pending() if hasattr(self._o, "n_pending") else 0

    def drain_events(self):
        if hasattr(self._o, "drain_events"):
            return self._o.drain_events()
        ev, self._o.events = (getattr(self._o, "events", []), []) if hasattr(self._o, "events") else ([], None)
        return ev


def parse_spec(text: str) -> dict:
    """'pkg.module:Class' | '/path/file.py:Class' | 'pkg.module' | '/path/file.py'"""
    target, _, cls = text.rpartition(":") if ":" in text else (text, "", "")
    if not target:
        target, cls = cls, ""
    spec: dict = {"path" if (target.endswith(".py") or os.sep in target) else "module": target}
    if cls:
        spec["cls"] = cls
    return spec


def _import(spec: dict, base_dir: str):
    if spec.get("path"):
        path = spec["path"] if os.path.isabs(spec["path"]) else os.path.join(base_dir, spec["path"])
        if not os.path.isfile(path):
            raise FileNotFoundError(f"association file not found: {path}")
        name = "assoc_" + os.path.splitext(os.path.basename(path))[0] + f"_{abs(hash(path)) % 10**6}"
        mod_spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(mod_spec)
        sys.modules[name] = module
        mod_spec.loader.exec_module(module)
        return module
    if spec.get("module"):
        return importlib.import_module(spec["module"])
    raise ValueError("association spec needs 'module' or 'path'")


def _load_config(cfg: Union[None, str, dict], base_dir: str) -> Optional[dict]:
    if cfg is None or isinstance(cfg, dict):
        return cfg
    path = cfg if os.path.isabs(cfg) else os.path.join(base_dir, cfg)
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_associator(spec: Union[str, dict], base_dir: str = ".", config: Union[None, str, dict] = None):
    """Return an object that has step/release/size/n_pending/drain_events."""
    if isinstance(spec, str):
        spec = parse_spec(spec)
    spec = dict(spec)
    if config is not None:
        spec["config"] = config
    module = _import(spec, base_dir)
    cfg = _load_config(spec.get("config"), base_dir)

    if spec.get("cls"):
        cls = getattr(module, spec["cls"], None)
        if cls is None:
            raise AttributeError(f"{module.__name__} has no class {spec['cls']!r}")
        obj = cls.from_config(cfg) if hasattr(cls, "from_config") else (cls(**cfg) if cfg else cls())
    elif callable(getattr(module, "step", None)):
        obj = _ModuleAssociator(module)
    else:
        raise ValueError(f"{module.__name__}: give 'cls', or define a module-level step(cam_id, live, frame)")
    if not callable(getattr(obj, "step", None)):
        raise TypeError(f"{type(obj).__name__} has no step(cam_id, live, frame)")
    return obj if isinstance(obj, AssociationBase) else _Compat(obj)
