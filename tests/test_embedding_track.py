import numpy as np
import pytest

from association.embedding import EmbeddingTrack, unit

rng = np.random.default_rng(0)


def rv(d=64):
    return rng.normal(size=d).astype(np.float32)


def test_update_normalises_and_rejects_zero():
    e = EmbeddingTrack()
    assert e.empty and e.distance(rv()) == 2.0
    assert not e.update(np.zeros(8))
    assert not e.update(np.array([np.nan] * 8))
    assert e.update(rv() * 9)
    assert np.linalg.norm(e.value) == pytest.approx(1.0, abs=1e-5) and e.count == 1


def test_ema_converges_to_repeated_vector():
    e, target = EmbeddingTrack(alpha=0.5), unit(rv())
    e.update(rv())
    for _ in range(40):
        e.update(target)
    assert e.distance(target) < 1e-3 and e.count == 41
    assert np.allclose(e.last, target)


def test_damping_slows_the_average():
    a, b = EmbeddingTrack(alpha=0.5), EmbeddingTrack(alpha=0.5)
    first, new = rv(), rv()
    a.update(first); b.update(first)
    a.update(new, damping=0.0); b.update(new, damping=1.0)
    assert b.distance(first) < a.distance(first)
    assert b.distance(first) < 1e-4                     # damping=1 freezes the average


def test_distance_to_other_track_and_copy_independent():
    a = EmbeddingTrack(); a.update(rv())
    c = a.copy(); c.update(rv())
    assert a.distance(a) == pytest.approx(0.0, abs=1e-5)
    assert a.count == 1 and c.count == 2
    assert a.distance(c) > 0


def test_window_mean_is_plain_mean_of_last_n():
    e = EmbeddingTrack(alpha=0.8, window=3)
    assert e.window_mean() is None
    vs = [unit(rv()) for _ in range(5)]
    for v in vs:
        e.update(v)
    assert e.window_count == 3
    assert np.allclose(e.window_mean(), unit(np.mean(vs[-3:], axis=0)), atol=1e-5)
    assert np.allclose(e.last, vs[-1])
    c = e.copy(); c.update(unit(rv()))
    assert e.window_count == 3 and not np.allclose(e.window_mean(), c.window_mean())


def test_window_off_by_default():
    e = EmbeddingTrack(); e.update(rv())
    assert e.window_count == 0 and e.window_mean() is None
