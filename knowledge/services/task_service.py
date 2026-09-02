"""Task status service."""
from __future__ import annotations

from knowledge.utils.task_util import add_done_task, add_running_task, get_done_task_list, get_running_task_list, get_task_status, update_task_status
from knowledge.utils.task_util import get_task_result


class TaskService:
    def mark_node_running(self, task_id: str, node_name: str):
        add_running_task(task_id, node_name)

    def mark_node_done(self, task_id: str, node_name: str):
        add_done_task(task_id, node_name)

    def update_task_status(self, task_id: str, status: str):
        update_task_status(task_id, status)

    def get_task_info(self, task_id: str):
        return {
            "status": get_task_status(task_id),
            "done_list": get_done_task_list(task_id),
            "running_list": get_running_task_list(task_id),
            "error": get_task_result(task_id, "error"),
        }
