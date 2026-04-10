"""Tests for HeatmapAccumulator."""

import numpy as np
import pytest

from eye_tracking.heatmap import HeatmapAccumulator, HeatmapConfig


def _make_heatmap(sigma: float = 2.0, decay: float = 0.5) -> HeatmapAccumulator:
    cfg = HeatmapConfig(height=72, width=128, sigma=sigma, decay=decay)
    return HeatmapAccumulator(cfg)


def test_initial_map_is_zeros() -> None:
    hm = _make_heatmap()
    assert np.all(hm.map == 0.0)


def test_update_with_none_does_not_raise() -> None:
    hm = _make_heatmap()
    hm.update(None, confidence=0.0, dt=0.016)
    assert np.all(hm.map == 0.0)


def test_update_adds_energy_to_map() -> None:
    hm = _make_heatmap()
    point = np.array([0.5, 0.5], dtype=np.float32)
    hm.update(point, confidence=1.0, dt=0.016)
    assert hm.map.max() > 0.0


def test_map_decays_over_time() -> None:
    hm = _make_heatmap(decay=10.0)
    point = np.array([0.5, 0.5], dtype=np.float32)
    hm.update(point, confidence=1.0, dt=0.016)
    peak_before = hm.map.max()
    # update with no new point but large dt
    hm.update(None, confidence=0.0, dt=1.0)
    assert hm.map.max() < peak_before


def test_reset_clears_map() -> None:
    hm = _make_heatmap()
    hm.update(np.array([0.5, 0.5], dtype=np.float32), confidence=1.0, dt=0.016)
    hm.reset()
    assert np.all(hm.map == 0.0)


def test_out_of_bounds_point_is_ignored() -> None:
    hm = _make_heatmap()
    hm.update(np.array([-1.0, 2.0], dtype=np.float32), confidence=1.0, dt=0.016)
    assert np.all(hm.map == 0.0)