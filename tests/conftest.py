"""Pytest configuration for TBC tests."""

import os
import tempfile
from collections.abc import Generator

import pytest


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_file(temp_dir: str) -> str:
    """Create a sample file for testing."""
    filepath = os.path.join(temp_dir, "sample.txt")
    with open(filepath, "w") as f:
        f.write("test content")
    return filepath


@pytest.fixture
def sample_model_file(temp_dir: str) -> str:
    """Create a sample model file with TFlite magic bytes."""
    filepath = os.path.join(temp_dir, "model.tflite")
    with open(filepath, "wb") as f:
        f.write(b"\x14\x00\x00\x00TFL3testdata")
    return filepath


@pytest.fixture
def sample_apk_file(temp_dir: str) -> str:
    """Create a sample APK file."""
    filepath = os.path.join(temp_dir, "app.apk")
    with open(filepath, "wb") as f:
        f.write(b"PK\x03\x04" + b"\x00" * 100)
    return filepath
