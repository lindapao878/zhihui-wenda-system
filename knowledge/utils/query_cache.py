"""In-memory structured query cache keyed by query and dataset version.

The dataset version is stored in MongoDB and shared by import/query processes.
When an import succeeds and increments the version, old in-process cache keys
become unreachable without requiring a cross-process clear() call.
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Union

from knowledge.utils.dataset_version_util import get_dataset_version

_DEFAULT_TTL = int(os.getenv("QUERY_CACHE_TTL_SECONDS", "300"))
_DEFAULT_MAX_ITEMS = int(os.getenv("QUERY_CACHE_MAX_ITEMS", "200"))
_CACHE_FIELDS = ("answer", "image_urls", "source_refs", "item_names", "related_entities")


class QueryCache:
    def __init__(self, ttl_seconds: Optional[int] = None, max_items: int = _DEFAULT_MAX_ITEMS):
        self.ttl_seconds = _DEFAULT_TTL if ttl_seconds is None else ttl_seconds
        self._max_items = max_items
        self._data: "OrderedDict[str, tuple[Dict[str, Any], float]]" = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _key(query: str, dataset_version: Optional[int] = None) -> str:
        version = get_dataset_version() if dataset_version is None else int(dataset_version)
        raw_key = f"v{version}:{query}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_value(value: Union[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if isinstance(value, str):
            value = {"answer": value}
        if not isinstance(value, dict):
            return None
        answer = value.get("answer")
        if not isinstance(answer, str) or not answer:
            return None
        payload: Dict[str, Any] = {}
        for field in _CACHE_FIELDS:
            payload[field] = value.get(field, [] if field != "answer" else answer)
        payload["answer"] = answer
        return payload

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        if not query:
            return None
        key = self._key(query)
        now = time.time()
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            payload, timestamp = item
            if now - timestamp > self.ttl_seconds:
                del self._data[key]
                return None
            return dict(payload)

    def set(self, query: str, value: Union[str, Dict[str, Any]]) -> None:
        if not query:
            return
        payload = self._normalize_value(value)
        if payload is None:
            return
        key = self._key(query)
        with self._lock:
            self._data[key] = (dict(payload), time.time())
            self._data.move_to_end(key)
            while len(self._data) > self._max_items:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


query_cache = QueryCache()
