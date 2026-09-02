"""Filesystem path helpers."""
from __future__ import annotations

from pathlib import Path
import os


def get_project_root() -> Path:
    """Return the project root that contains the knowledge package."""
    return Path(__file__).resolve().parents[2]


def get_front_page_dir() -> str:
    """Return the frontend static directory."""
    return str(get_project_root() / "knowledge" / "front")


def get_temp_data_dir() -> str:
    """Return the local temp data directory."""
    return os.getenv("TEMP_DATA_DIR", str(get_project_root() / "temp_data"))
