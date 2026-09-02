"""In-memory query result cache keyed by rewritten_query hash with TTL.

Skips the RAG pipeline when the same rewritten query is seen again within the TTL.
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import OrderedDict
from typing import Optional

_DEFAULT_TTL = int(os.getenv("QUERY_CACHE_TTL_SECONDS", "300"))
_DEFAULT_MAX_ITEMS = int(os.getenv("QUERY_CACHE_MAX_ITEMS", "200"))


class QueryCache:
    def __init__(self, ttl_seconds: Optional[int] = None, max_items: int = _DEFAULT_MAX_ITEMS):
        self.ttl_seconds = _DEFAULT_TTL if ttl_seconds is None else ttl_seconds
        self._max_items = max_items
        self._data: "OrderedDict[str, tuple[str, float]]" = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _key(query: str) -> str:
        return hashlib.sha256(query.encode("utf-8")).hexdigest()

    def get(self, query: str) -> Optional[str]:
        if not query:
            return None
        key = self._key(query)
        now = time.time()
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            answer, ts = item
            if now - ts > self.ttl_seconds:
                del self._data[key]
                return None
            return answer

    def set(self, query: str, answer: str) -> None:
        if not query or not answer:
            return
        key = self._key(query)
        with self._lock:
            self._data[key] = (answer, time.time())
            self._data.move_to_end(key)
            while len(self._data) > self._max_items:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


query_cache = QueryCache()
