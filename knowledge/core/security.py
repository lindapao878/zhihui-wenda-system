"""API key authentication and CORS configuration helpers."""
from __future__ import annotations

import os
from typing import List

from fastapi import HTTPException, Request

from knowledge.utils.logger_util import logger

_WARNED_NO_KEY = False


def _is_exempt_path(path: str) -> bool:
    """Paths that never require auth (probes + static front pages)."""
    if path in ("/health", "/ready", "/"):
        return True
    # static front pages served at known routes
    front_pages = (
        "/import", "/import.html", "/chat.html",
        "/front", "/front/",
    )
    if path in front_pages:
        return True
    if path.startswith("/front/"):
        return True
    return False


def get_app_api_key() -> str:
    return os.getenv("APP_API_KEY", "").strip()


def verify_api_key(request: Request) -> None:
    """FastAPI dependency: validate X-API-Key header against APP_API_KEY env.

    When APP_API_KEY is empty (local dev), auth is skipped with a one-time
    startup warning. In production the key must be set explicitly.
    """
    global _WARNED_NO_KEY
    expected = get_app_api_key()

    if not expected:
        if not _WARNED_NO_KEY:
            logger.warning("APP_API_KEY 未设置，接口鉴权已跳过（仅限本地开发；生产环境必须设置）")
            _WARNED_NO_KEY = True
        return

    if _is_exempt_path(request.url.path):
        return

    provided = request.headers.get("X-API-Key", "")
    if provided != expected:
        raise HTTPException(status_code=401, detail="无效或缺失的 API Key")


def get_allowed_origins() -> List[str]:
    raw = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000,http://localhost:8001")
    return [o.strip() for o in raw.split(",") if o.strip()]
