"""Tests for RuntimeConfig validation."""

import pytest

from eye_tracking.config import RuntimeConfig


def test_default_config_is_valid() -> None:
    cfg = RuntimeConfig()
    assert cfg.camera_id == 0
    assert cfg.frame_width == 1280
    assert cfg.frame_height == 720
    assert cfg.target_fps == 120
    assert 0.0 < cfg.min_confidence <= 1.0
    assert cfg.heatmap_sigma > 0.0


def test_process_every_n_must_be_at_least_one() -> None:
    with pytest.raises(ValueError, match="process_every_n"):
        RuntimeConfig(process_every_n=0)


def test_min_confidence_must_be_positive() -> None:
    with pytest.raises(ValueError, match="min_confidence"):
        RuntimeConfig(min_confidence=0.0)


def test_min_confidence_must_not_exceed_one() -> None:
    with pytest.raises(ValueError, match="min_confidence"):
        RuntimeConfig(min_confidence=1.1)


def test_heatmap_sigma_must_be_positive() -> None:
    with pytest.raises(ValueError, match="heatmap_sigma"):
        RuntimeConfig(heatmap_sigma=0.0)


def test_custom_config_values_are_stored() -> None:
    cfg = RuntimeConfig(camera_id=2, frame_width=640, frame_height=480)
    assert cfg.camera_id == 2
    assert cfg.frame_width == 640
    assert cfg.frame_height == 480