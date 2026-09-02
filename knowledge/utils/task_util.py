"""Task tracking for import and query services.

Public API is unchanged from the in-memory era; internally delegates to
task_store (MongoDB-backed with in-memory fallback). This lets task state
survive process restarts while keeping all call sites untouched.
"""
from __future__ import annotations

from typing import Any, List

from knowledge.utils import task_store

TASK_STATUS_PENDING = "pending"
TASK_STATUS_PROCESSING = "processing"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"


def init_on_startup() -> None:
    """Call once at application startup to clean up stale tasks."""
    task_store.mark_processing_as_interrupted()


def add_running_task(task_id: str, node_name: str) -> None:
    task_store.add_running(task_id, node_name)


def add_done_task(task_id: str, node_name: str) -> None:
    task_store.add_done(task_id, node_name)


def update_task_status(task_id: str, status: str) -> None:
    task_store.update_status(task_id, status)


def get_task_status(task_id: str) -> str:
    return task_store.get_status(task_id)


def get_running_task_list(task_id: str) -> List[str]:
    return task_store.get_running_list(task_id)


def get_done_task_list(task_id: str) -> List[str]:
    return task_store.get_done_list(task_id)


def set_task_result(task_id: str, key: str, value: Any) -> None:
    task_store.set_result(task_id, key, value)


def get_task_result(task_id: str, key: str, default: Any = None) -> Any:
    return task_store.get_result(task_id, key, default)


def clear_task(task_id: str) -> None:
    task_store.clear(task_id)