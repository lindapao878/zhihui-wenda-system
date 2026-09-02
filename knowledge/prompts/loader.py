"""Prompt file loader."""
from __future__ import annotations

from pathlib import Path

_PROMPT_ROOT = Path(__file__).resolve().parent


def load_prompt(*parts: str) -> str:
    path = _PROMPT_ROOT.joinpath(*parts)
    return path.read_text(encoding="utf-8").strip()
