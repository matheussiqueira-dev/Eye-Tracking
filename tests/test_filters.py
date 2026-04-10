"""Tests for temporal filters."""

import math

import numpy as np

from eye_tracking.config import RuntimeConfig
from eye_tracking.filters import GazeStabilizer, LowPassFilter, OneEuroFilter


class TestLowPassFilter:
    def test_first_value_is_returned_unchanged(self) -> None:
        f = LowPassFilter()
        result = f.apply(0.5, alpha=0.8)
        assert result == pytest.approx(0.5)

    def test_filter_moves_toward_new_value(self) -> None:
        import pytest
        f = LowPassFilter()
        f.apply(0.0, alpha=0.5)
        result = f.apply(1.0, alpha=0.5)
        assert 0.0 < result < 1.0

    def test_alpha_one_tracks_input_exactly(self) -> None:
        import pytest
        f = LowPassFilter()
        f.apply(0.0, alpha=1.0)
        assert f.apply(0.8, alpha=1.0) == pytest.approx(0.8)


class TestOneEuroFilter:
    def test_static_signal_converges(self) -> None:
        import pytest
        f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        val = 0.5
        for _ in range(100):
            val = f.apply(0.5, dt=0.016)
        assert val == pytest.approx(0.5, abs=1e-3)

    def test_ramp_signal_tracks_with_lag(self) -> None:
        f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        target = 0.0
        for i in range(60):
            target = i / 60.0
            out = f.apply(target, dt=0.016)
        # Some lag expected but should be tracking ballpark
        assert abs(out - target) < 0.3


class TestGazeStabilizer:
    def test_returns_clipped_output(self) -> None:
        import pytest
        config = RuntimeConfig()
        stab = GazeStabilizer(config)
        meas = np.array([1.5, -0.2], dtype=np.float32)
        result = stab.update(meas, dt=0.016, confidence=0.9)
        assert result.shape == (2,)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_none_measurement_uses_prediction(self) -> None:
        config = RuntimeConfig()
        stab = GazeStabilizer(config)
        # Prime with a valid measurement first
        stab.update(np.array([0.5, 0.5], dtype=np.float32), dt=0.016, confidence=0.9)
        result = stab.update(None, dt=0.016, confidence=0.0)
        assert result.shape == (2,)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)