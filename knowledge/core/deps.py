"""FastAPI dependency providers."""
from __future__ import annotations

from functools import lru_cache

from knowledge.services.file_import_service import ImportFileService
from knowledge.services.query_service import QueryService
from knowledge.services.task_service import TaskService


@lru_cache
def get_import_file_service() -> ImportFileService:
    return ImportFileService()


@lru_cache
def get_query_service() -> QueryService:
    return QueryService()


@lru_cache
def get_task_service() -> TaskService:
    return TaskService()
