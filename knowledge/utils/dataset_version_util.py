"""Dataset version used to invalidate query cache entries."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from knowledge.utils.logger_util import logger

VERSION_DOCUMENT_ID = "dataset_version"
_metadata_collection = None
_fallback_version = 1


def _get_metadata_collection():
    """Return kb_metadata collection, or None when MongoDB is unavailable."""
    global _metadata_collection
    if _metadata_collection is not None:
        return _metadata_collection
    try:
        from knowledge.utils.mongo_history_util import get_mongo_client

        client = get_mongo_client()
        if client is None:
            return None
        db_name = os.getenv("MONGO_DB_NAME", "kb001")
        _metadata_collection = client[db_name]["kb_metadata"]
        return _metadata_collection
    except Exception as exc:
        logger.warning("dataset_version MongoDB 连接失败，使用进程内降级版本: {}", exc)
        return None


def get_dataset_version() -> int:
    """Read the shared dataset version; default to 1 before first import."""
    global _fallback_version
    collection = _get_metadata_collection()
    if collection is None:
        return _fallback_version
    try:
        document: Optional[Dict[str, Any]] = collection.find_one({"_id": VERSION_DOCUMENT_ID})
        if not document:
            return 1
        return int(document.get("version", 1))
    except Exception as exc:
        logger.warning("读取 dataset_version 失败: {}", exc)
        return _fallback_version


def increment_dataset_version() -> int:
    """Increment the shared version after a successful dataset mutation."""
    global _fallback_version
    collection = _get_metadata_collection()
    if collection is None:
        _fallback_version += 1
        return _fallback_version
    try:
        from pymongo import ReturnDocument

        document = collection.find_one_and_update(
            {"_id": VERSION_DOCUMENT_ID},
            {"$inc": {"version": 1}},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return int(document.get("version", 1))
    except Exception as exc:
        logger.warning("递增 dataset_version 失败: {}", exc)
        _fallback_version += 1
        return _fallback_version
