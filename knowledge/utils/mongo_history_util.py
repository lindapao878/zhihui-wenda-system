"""MongoDB chat history helpers."""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from knowledge.utils.logger_util import logger

load_dotenv()

_mongo_client = None


def get_mongo_client():
    global _mongo_client
    if _mongo_client is not None:
        return _mongo_client

    try:
        from pymongo import MongoClient

        uri = os.getenv("MONGO_URL", "mongodb://localhost:27017")
        kwargs: Dict[str, Any] = {"serverSelectionTimeoutMS": 3000}
        user = os.getenv("MONGO_USER", "")
        password = os.getenv("MONGO_PASSWORD", "")
        if user:
            kwargs["username"] = user
        if password:
            kwargs["password"] = password

        _mongo_client = MongoClient(uri, **kwargs)
    except Exception as exc:
        logger.error("创建 MongoDB 客户端失败: {}", exc)
        return None

    return _mongo_client


def _history_collection():
    client = get_mongo_client()
    if client is None:
        return None
    db_name = os.getenv("MONGO_DB_NAME", "kb001")
    return client[db_name]["chat_history"]


def save_chat_message(
    session_id: str,
    role: str,
    text: str,
    rewritten_query: str = "",
    item_names: Optional[List[str]] = None,
) -> Optional[str]:
    collection = _history_collection()
    if collection is None:
        return None

    try:
        result = collection.insert_one(
            {
                "session_id": session_id,
                "role": role,
                "text": text,
                "rewritten_query": rewritten_query,
                "item_names": item_names or [],
                "ts": time.time(),
            }
        )
        return str(result.inserted_id)
    except Exception as exc:
        logger.warning("保存聊天记录失败: {}", exc)
        return None


def get_recent_messages(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    collection = _history_collection()
    if collection is None:
        return []

    try:
        cursor = collection.find({"session_id": session_id}).sort("ts", -1).limit(limit)
        records = list(cursor)
        records.reverse()
        return records
    except Exception as exc:
        logger.warning("读取聊天记录失败: {}", exc)
        return []


def update_message_item_names(ids: List[str], item_names: List[str]) -> None:
    collection = _history_collection()
    if collection is None or not ids:
        return

    try:
        from bson import ObjectId

        object_ids = []
        for raw_id in ids:
            try:
                object_ids.append(ObjectId(raw_id))
            except Exception:
                object_ids.append(raw_id)

        collection.update_many(
            {"_id": {"$in": object_ids}},
            {"$set": {"item_names": item_names}},
        )
    except Exception as exc:
        logger.warning("回填历史 item_names 失败: {}", exc)


def clear_history(session_id: str) -> int:
    collection = _history_collection()
    if collection is None:
        return 0

    try:
        result = collection.delete_many({"session_id": session_id})
        return result.deleted_count
    except Exception as exc:
        logger.warning("清空聊天记录失败: {}", exc)
        return 0
