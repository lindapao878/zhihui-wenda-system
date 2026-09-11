"""Mongo document registry for content-hash duplicate detection."""
from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

from knowledge.utils.logger_util import logger

_collection = None


def _get_collection():
    global _collection
    if _collection is not None:
        return _collection
    try:
        from knowledge.utils.mongo_history_util import get_mongo_client

        client = get_mongo_client()
        if client is None:
            return None
        db_name = os.getenv("MONGO_DB_NAME", "kb001")
        _collection = client[db_name]["kb_documents"]
        return _collection
    except Exception as exc:
        logger.warning("kb_documents MongoDB 连接失败，内容 hash 去重降级: {}", exc)
        return None


def is_registry_available() -> bool:
    return _get_collection() is not None


def find_active_document(content_hash: str) -> Optional[Dict[str, Any]]:
    if not content_hash:
        return None
    collection = _get_collection()
    if collection is None:
        return None
    try:
        return collection.find_one({"_id": content_hash, "status": "active"})
    except Exception as exc:
        logger.warning("查询 kb_documents 失败: {}", exc)
        return None


def register_importing(content_hash: str, file_title: str, task_id: str) -> bool:
    if not content_hash:
        return False
    collection = _get_collection()
    if collection is None:
        return False
    now = time.time()
    document = {
        "_id": content_hash,
        "content_hash": content_hash,
        "file_title": file_title,
        "status": "importing",
        "task_id": task_id,
        "created_at": now,
        "updated_at": now,
    }
    try:
        collection.replace_one({"_id": content_hash}, document, upsert=True)
        return True
    except Exception as exc:
        logger.warning("写入 kb_documents importing 失败: {}", exc)
        return False


def mark_active(content_hash: str, file_title: str, task_id: str) -> bool:
    if not content_hash:
        return False
    collection = _get_collection()
    if collection is None:
        return False
    try:
        collection.update_one(
            {"_id": content_hash},
            {
                "$set": {
                    "content_hash": content_hash,
                    "file_title": file_title,
                    "status": "active",
                    "task_id": task_id,
                    "updated_at": time.time(),
                }
            },
            upsert=True,
        )
        return True
    except Exception as exc:
        logger.warning("更新 kb_documents active 失败: {}", exc)
        return False


def mark_failed(content_hash: str, file_title: str, task_id: str) -> bool:
    if not content_hash:
        return False
    collection = _get_collection()
    if collection is None:
        return False
    try:
        collection.update_one(
            {"_id": content_hash},
            {
                "$set": {
                    "file_title": file_title,
                    "status": "failed",
                    "task_id": task_id,
                    "updated_at": time.time(),
                }
            },
        )
        return True
    except Exception as exc:
        logger.warning("更新 kb_documents failed 失败: {}", exc)
        return False


def supersede_title(file_title: str, except_content_hash: str) -> int:
    collection = _get_collection()
    if collection is None or not file_title:
        return 0
    try:
        result = collection.update_many(
            {
                "file_title": file_title,
                "status": "active",
                "_id": {"$ne": except_content_hash},
            },
            {"$set": {"status": "superseded", "updated_at": time.time()}},
        )
        return int(result.modified_count)
    except Exception as exc:
        logger.warning("标记旧 file_title 为 superseded 失败: {}", exc)
        return 0


def delete_by_title(file_title: str) -> int:
    collection = _get_collection()
    if collection is None:
        return 0
    try:
        result = collection.delete_many({"file_title": file_title})
        return int(result.deleted_count)
    except Exception as exc:
        logger.warning("删除 kb_documents 失败: {}", exc)
        return 0
