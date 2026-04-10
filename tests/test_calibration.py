"""Tests for LinearCalibrator."""

import numpy as np
import pytest

from eye_tracking.calibration import CalibrationRoutine, LinearCalibrator


def test_calibrator_not_ready_before_fit() -> None:
    cal = LinearCalibrator()
    assert not cal.is_ready


def test_fit_fails_with_too_few_samples() -> None:
    cal = LinearCalibrator()
    for i in range(5):
        cal.add_sample(np.zeros(6, dtype=np.float32), np.array([0.5, 0.5], dtype=np.float32))
    assert not cal.fit()
    assert not cal.is_ready


def test_fit_succeeds_with_sufficient_samples() -> None:
    cal = LinearCalibrator()
    rng = np.random.default_rng(42)
    for i in range(9):
        feat = rng.standard_normal(6).astype(np.float32)
        target = rng.random(2).astype(np.float32)
        cal.add_sample(feat, target)
    assert cal.fit()
    assert cal.is_ready


def test_predict_returns_values_in_unit_range() -> None:
    cal = LinearCalibrator()
    rng = np.random.default_rng(0)
    for _ in range(9):
        cal.add_sample(rng.standard_normal(6).astype(np.float32), rng.random(2).astype(np.float32))
    cal.fit()
    pred = cal.predict(np.zeros(6, dtype=np.float32))
    assert pred.shape == (2,)
    assert np.all(pred >= 0.0)
    assert np.all(pred <= 1.0)


def test_predict_raises_when_not_fitted() -> None:
    cal = LinearCalibrator()
    with pytest.raises(RuntimeError, match="not fitted"):
        cal.predict(np.zeros(6, dtype=np.float32))


def test_reset_clears_state() -> None:
    cal = LinearCalibrator()
    rng = np.random.default_rng(1)
    for _ in range(9):
        cal.add_sample(rng.standard_normal(6).astype(np.float32), rng.random(2).astype(np.float32))
    cal.fit()
    assert cal.is_ready
    cal.reset()
    assert not cal.is_ready