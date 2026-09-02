"""Base node for query flow."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, TypeVar

from knowledge.processor.query_process.config import QueryConfig, get_config
from knowledge.utils.task_util import add_done_task, add_running_task
from knowledge.utils.sse_util import SSEEvent, push_sse_event
from knowledge.utils.task_util import get_done_task_list, get_running_task_list
from knowledge.utils.logger_util import logger

T = TypeVar("T")


class BaseNode(ABC):
    name: str = "base_node"

    def __init__(self, config: Optional[QueryConfig] = None):
        self.config = config or get_config()

    def __call__(self, state: T) -> T:
        task_id = state.get("task_id", "")
        try:
            logger.info("--- {} 开始 ---", self.name)
            if task_id:
                add_running_task(task_id, self.name)
                self._push_progress(task_id, state)

            result = self.process(state)

            logger.info("--- {} 完成 ---", self.name)
            if task_id:
                add_done_task(task_id, self.name)
                self._push_progress(task_id, state)
            return result
        except Exception as exc:
            logger.error("{} 执行失败: {}", self.name, exc)
            raise

    @staticmethod
    def _push_progress(task_id, state):
        if state.get("is_stream"):
            push_sse_event(task_id, SSEEvent.PROGRESS, {
                "status": "processing",
                "done_list": get_done_task_list(task_id),
                "running_list": get_running_task_list(task_id),
            })

    @abstractmethod
    def process(self, state: T) -> T:
        pass

    def log_step(self, step_name: str, message: str = ""):
        log_msg = f"[{step_name}]"
        if message:
            log_msg += f" {message}"
        logger.info(log_msg)

    def _apply_min_score_filter(self, hits: list) -> list:
        """Filter hits by milvus_min_cosine_score, logging before/after counts."""
        threshold = self.config.milvus_min_cosine_score
        before = len(hits)
        if threshold <= 0 or before == 0:
            return hits
        kept = [h for h in hits if (h.get("distance") or 0) >= threshold]
        logger.info(
            "{} 最低分数过滤: before={} after={} threshold={:.3f}",
            self.name, before, len(kept), threshold,
        )
        return kept

