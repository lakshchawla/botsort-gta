import textwrap

import numpy as np
import pytest

from association import AssociationBase, Associator, load_associator, parse_spec
from association.embedding import EmbeddingTrack
from tracker.tracklet import Tracklet


def tl(cam=0, tid=1):
    t = Tracklet(cam, tid, 0, 0, EmbeddingTrack())
    t.clean, t.tracklet_len, t.new_embedding_this_frame = True, 99, True
    return t


def test_parse_spec_forms():
    assert parse_spec("association.associator:Associator") == {"module": "association.associator", "cls": "Associator"}
    assert parse_spec("/tmp/x/assoc_v2.py:V2") == {"path": "/tmp/x/assoc_v2.py", "cls": "V2"}
    assert parse_spec("pkg.mod") == {"module": "pkg.mod"}
    assert parse_spec("exp/step_only.py") == {"path": "exp/step_only.py"}


def test_load_builtin_by_module_with_inline_config():
    a = load_associator({"module": "association.associator", "cls": "Associator",
                         "config": {"min_embeddings": 1, "mint_min_embeddings": 1, "min_tracklet_len": 1}})
    assert isinstance(a, Associator) and a.cfg.min_embeddings == 1
    t = tl(); t.embedding.update(np.ones(8))
    assert a.step(0, [t], 1) == [t] and t.t_global_id == 1 and a.size() == 1


def test_load_class_from_a_file_and_minimal_interface(tmp_path):
    f = tmp_path / "assoc_v2.py"
    f.write_text(textwrap.dedent('''
        class V2:                                   # NOT a subclass of anything: only step() is required
            def __init__(self, offset=100):
                self.offset = offset
            def step(self, cam_id, live, frame):
                for t in live:
                    t.t_global_id = self.offset + t.track_id
                return list(live)
    '''))
    a = load_associator({"path": str(f), "cls": "V2", "config": {"offset": 7}})
    t = tl(tid=3)
    assert a.step(0, [t], 1) == [t] and t.t_global_id == 10
    # the optional parts exist with safe defaults
    a.release((0, 3)); assert a.size() == 0 and a.n_pending() == 0 and a.drain_events() == []


def test_module_with_only_a_step_function(tmp_path):
    f = tmp_path / "fn_only.py"
    f.write_text("calls = []\n"
                 "def step(cam_id, live, frame):\n"
                 "    calls.append(frame)\n"
                 "    for t in live:\n"
                 "        t.t_global_id = 42\n"
                 "    return list(live)\n"
                 "def release(key):\n"
                 "    calls.append(('rel', key))\n")
    a = load_associator(str(f))                  # string spec: path, no class
    t = tl()
    assert a.step(1, [t], 5) == [t] and t.t_global_id == 42
    a.release((1, 1))
    assert a._m.calls == [5, ("rel", (1, 1))]


def test_events_are_drained_from_a_plain_class(tmp_path):
    f = tmp_path / "ev.py"
    f.write_text("class E:\n    def __init__(self):\n        self.events = []\n"
                 "    def step(self, c, live, fr):\n        self.events.append({'event': 'x', 'frame': fr})\n        return []\n")
    a = load_associator(f"{f}:E")
    a.step(0, [], 3)
    assert a.drain_events() == [{"event": "x", "frame": 3}] and a.drain_events() == []


def test_bad_specs_fail_loudly(tmp_path):
    with pytest.raises(ValueError):
        load_associator({"cls": "X"})
    with pytest.raises(FileNotFoundError):
        load_associator({"path": str(tmp_path / "nope.py")})
    f = tmp_path / "nostep.py"
    f.write_text("class A:\n    pass\n")
    with pytest.raises(TypeError):
        load_associator({"path": str(f), "cls": "A"})
    g = tmp_path / "plain.py"
    g.write_text("x = 1\n")
    with pytest.raises(ValueError):
        load_associator({"path": str(g)})
    with pytest.raises(AttributeError):
        load_associator({"path": str(g), "cls": "Missing"})


def test_global_tracker_accepts_a_swapped_association(tmp_path):
    from global_tracker import GlobalTracker
    f = tmp_path / "tag.py"
    f.write_text("def step(cam_id, live, frame):\n    for t in live:\n        t.t_global_id = 900 + t.track_id\n    return list(live)\n")
    gt = GlobalTracker(association=str(f))
    assert gt.assoc.size() == 0 and gt.drain_events() == []
