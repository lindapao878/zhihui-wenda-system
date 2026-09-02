"""MongoDB-backed task state store with graceful in-memory fallback.

Document shape in kb_tasks collection:
    { _id: task_id, status, running_nodes: [], done_nodes: [], result: {}, updated_at }

If MongoDB is unreachable at first access, all operations silently
degrade to an in-memory dict so the API still responds.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from knowledge.utils.logger_util import logger

_collection = None
_memory: Dict[str, Dict[str, Any]] = {}


def _get_collection():
    """Return the kb_tasks collection, or None to signal memory fallback."""
    global _collection
    if _collection is not None:
        return _collection
    try:
        from knowledge.utils.mongo_history_util import get_mongo_client

        client = get_mongo_client()
        if client is None:
            return None
        # verify connectivity before committing to Mongo mode
        client.admin.command("ping")  # serverSelectionTimeoutMS is on MongoClient
        db_name = os.getenv("MONGO_DB_NAME", "kb001")
        _collection = client[db_name]["kb_tasks"]
        logger.info("task_store 已连接 MongoDB (集合 kb_tasks)")
        return _collection
    except Exception as exc:
        logger.warning("task_store MongoDB 连接失败，降级为内存模式: {}", exc)
        return None


def _now() -> float:
    return time.time()


def _mem_doc(task_id: str) -> Dict[str, Any]:
    return _memory.setdefault(
        task_id,
        {"status": "pending", "running_nodes": [], "done_nodes": [], "result": {}},
    )


def mark_processing_as_interrupted() -> int:
    """On startup, flip all 'processing' tasks to 'failed' (interrupted)."""
    col = _get_collection()
    if col is None:
        return 0
    try:
        result = col.update_many(
            {"status": "processing"},
            {"$set": {"status": "failed", "result.interrupted": True, "updated_at": _now()}},
        )
        if result.modified_count:
            logger.warning(
                "启动时发现 {} 个 processing 状态任务，已标记为 failed(interrupted)",
                result.modified_count,
            )
        return result.modified_count
    except Exception as exc:
        logger.warning("清理 processing 任务失败: {}", exc)
        return 0


def add_running(task_id: str, node_name: str) -> None:
    col = _get_collection()
    if col is None:
        doc = _mem_doc(task_id)
        if node_name not in doc["running_nodes"]:
            doc["running_nodes"].append(node_name)
        doc["done_nodes"] = [n for n in doc["done_nodes"] if n != node_name]
        return
    try:
        col.update_one(
            {"_id": task_id},
            {
                "$addToSet": {"running_nodes": node_name},
                "$pull": {"done_nodes": node_name},
                "$setOnInsert": {"status": "pending", "result": {}},
                "$set": {"updated_at": _now()},
            },
            upsert=True,
        )
    except Exception as exc:
        logger.warning("task_store add_running 失败: {}", exc)


def add_done(task_id: str, node_name: str) -> None:
    col = _get_collection()
    if col is None:
        doc = _mem_doc(task_id)
        doc["running_nodes"] = [n for n in doc["running_nodes"] if n != node_name]
        if node_name not in doc["done_nodes"]:
            doc["done_nodes"].append(node_name)
        return
    try:
        col.update_one(
            {"_id": task_id},
            {
                "$pull": {"running_nodes": node_name},
                "$addToSet": {"done_nodes": node_name},
                "$setOnInsert": {"status": "pending", "result": {}},
                "$set": {"updated_at": _now()},
            },
            upsert=True,
        )
    except Exception as exc:
        logger.warning("task_store add_done 失败: {}", exc)


def update_status(task_id: str, status: str) -> None:
    col = _get_collection()
    if col is None:
        _mem_doc(task_id)["status"] = status
        return
    try:
        col.update_one(
            {"_id": task_id},
            {"$set": {"status": status, "updated_at": _now()}},
            upsert=True,
        )
    except Exception as exc:
        logger.warning("task_store update_status 失败: {}", exc)


def get_status(task_id: str) -> str:
    col = _get_collection()
    if col is None:
        return _mem_doc(task_id)["status"]
    try:
        doc = col.find_one({"_id": task_id}, {"status": 1})
        return doc.get("status", "pending") if doc else "pending"
    except Exception as exc:
        logger.warning("task_store get_status 失败: {}", exc)
        return "pending"


def get_running_list(task_id: str) -> List[str]:
    col = _get_collection()
    if col is None:
        return list(_mem_doc(task_id)["running_nodes"])
    try:
        doc = col.find_one({"_id": task_id}, {"running_nodes": 1})
        return list(doc.get("running_nodes", [])) if doc else []
    except Exception as exc:
        logger.warning("task_store get_running_list 失败: {}", exc)
        return []


def get_done_list(task_id: str) -> List[str]:
    col = _get_collection()
    if col is None:
        return list(_mem_doc(task_id)["done_nodes"])
    try:
        doc = col.find_one({"_id": task_id}, {"done_nodes": 1})
        return list(doc.get("done_nodes", [])) if doc else []
    except Exception as exc:
        logger.warning("task_store get_done_list 失败: {}", exc)
        return []


def set_result(task_id: str, key: str, value: Any) -> None:
    col = _get_collection()
    if col is None:
        _mem_doc(task_id)["result"][key] = value
        return
    try:
        col.update_one(
            {"_id": task_id},
            {"$set": {f"result.{key}": value, "updated_at": _now()}},
            upsert=True,
        )
    except Exception as exc:
        logger.warning("task_store set_result 失败: {}", exc)


def get_result(task_id: str, key: str, default: Any = None) -> Any:
    col = _get_collection()
    if col is None:
        return _mem_doc(task_id)["result"].get(key, default)
    try:
        doc = col.find_one({"_id": task_id}, {"result": 1})
        if not doc:
            return default
        return doc.get("result", {}).get(key, default)
    except Exception as exc:
        logger.warning("task_store get_result 失败: {}", exc)
        return default


def clear(task_id: str) -> None:
    col = _get_collection()
    if col is None:
        _memory.pop(task_id, None)
        return
    try:
        col.delete_one({"_id": task_id})
    except Exception as exc:
        logger.warning("task_store clear 失败: {}", exc)
