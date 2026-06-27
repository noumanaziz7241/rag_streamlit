"""Shared pytest fixtures."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_db_path() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        yield str(Path(tmp) / "test.db")
