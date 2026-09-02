"""Knowledge query service."""
from __future__ import annotations

import uuid
from typing import Any, Dict, List

from knowledge.processor.query_process.main_graph import query_app
from knowledge.utils.mongo_history_util import clear_history, get_recent_messages
from knowledge.utils.sse_util import create_sse_queue, push_sse_event
from knowledge.utils.task_util import TASK_STATUS_COMPLETED, TASK_STATUS_PROCESSING, TASK_STATUS_FAILED, get_done_task_list, get_running_task_list, get_task_result, get_task_status, set_task_result, update_task_status
from knowledge.utils.logger_util import logger



class QueryService:
    def generate_session_id(self) -> str:
        return str(uuid.uuid4())

    def generate_task_id(self) -> str:
        return str(uuid.uuid4())

    def submit_query(self, task_id: str, is_stream: bool):
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        if is_stream:
            create_sse_queue(task_id)

    def run_query_graph(self, task_id: str, session_id: str, user_query: str, is_stream: bool):
        state = {
            "original_query": user_query,
            "session_id": session_id,
            "task_id": task_id,
            "is_stream": is_stream,
        }
        try:
            query_app.invoke(state)
        except Exception as exc:
            logger.exception('查询流程执行失败: {}', exc)
            set_task_result(task_id, 'error', str(exc))
            update_task_status(task_id, TASK_STATUS_FAILED)
        finally:
            set_task_result(task_id, 'done_list', get_done_task_list(task_id))
            set_task_result(task_id, 'running_list', get_running_task_list(task_id))
            if get_task_status(task_id) != TASK_STATUS_FAILED:
                update_task_status(task_id, TASK_STATUS_COMPLETED)
            if is_stream:
                push_sse_event(task_id, "progress", {
                    "status": get_task_status(task_id),
                    "done_list": get_done_task_list(task_id),
                    "running_list": get_running_task_list(task_id),
                })

    def get_answer(self, task_id: str) -> str:
        return get_task_result(task_id, "answer", "")

    def get_task_info(self, task_id: str) -> Dict[str, Any]:
        return {
            'status': get_task_status(task_id),
            'done_list': get_task_result(task_id, 'done_list', get_done_task_list(task_id)),
            'running_list': get_task_result(task_id, 'running_list', get_running_task_list(task_id)),
            'answer': self.get_answer(task_id),
            'error': get_task_result(task_id, 'error'),
            'image_urls': get_task_result(task_id, 'image_urls'),
        }

    def get_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        records = get_recent_messages(session_id, limit=limit)
        return [
            {
                "_id": str(record.get("_id", "")),
                "session_id": record.get("session_id", ""),
                "role": record.get("role", ""),
                "text": record.get("text", ""),
                "rewritten_query": record.get("rewritten_query", ""),
                "item_names": record.get("item_names", []),
                "ts": record.get("ts"),
            }
            for record in records
        ]

    def clear_history(self, session_id: str) -> int:
        return clear_history(session_id)



