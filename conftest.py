"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the src directory is on the path when running pytest from the repo root.
SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture()
def default_config():
    """Return a default RuntimeConfig instance."""
    from eye_tracking.config import RuntimeConfig

    return RuntimeConfig()
