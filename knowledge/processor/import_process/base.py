"""Base node for document import flow."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, TypeVar

from knowledge.processor.import_process.config import ImportConfig, get_config
from knowledge.processor.import_process.exceptions import ImportProcessError
from knowledge.utils.task_util import add_done_task, add_running_task
from knowledge.utils.logger_util import logger

T = TypeVar("T")


class BaseNode(ABC):
    name: str = "base_node"

    def __init__(self, config: Optional[ImportConfig] = None):
        self.config = config or get_config()

    def __call__(self, state: T) -> T:
        task_id = state.get("task_id", "")
        try:
            logger.info("--- {} 开始 ---", self.name)
            if task_id:
                add_running_task(task_id, self.name)

            result = self.process(state)

            logger.info("--- {} 完成 ---", self.name)
            if task_id:
                add_done_task(task_id, self.name)
            return result
        except Exception as exc:
            logger.error("{} 执行失败: {}", self.name, exc)
            raise ImportProcessError(message=str(exc), node_name=self.name, cause=exc)

    @abstractmethod
    def process(self, state: T) -> T:
        pass

    def log_step(self, step_name: str, message: str = ""):
        log_msg = f"[{step_name}]"
        if message:
            log_msg += f" {message}"
        logger.info(log_msg)

