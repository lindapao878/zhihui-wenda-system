"""SSE event queue utilities."""
from __future__ import annotations

import asyncio
import json
import queue
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import Request


class SSEEvent:
    READY = "ready"
    PROGRESS = "progress"
    DELTA = "delta"
    FINAL = "final"


_task_stream: Dict[str, queue.Queue] = {}


def get_sse_queue(task_id: str) -> Optional[queue.Queue]:
    return _task_stream.get(task_id)


def create_sse_queue(task_id: str) -> queue.Queue:
    q = queue.Queue()
    _task_stream[task_id] = q
    return q


def remove_sse_queue(task_id: str) -> None:
    _task_stream.pop(task_id, None)


def _sse_pack(event: str, data: Dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def push_sse_event(task_id: str, event: str, data: Dict[str, Any]) -> None:
    stream_queue = get_sse_queue(task_id)
    if stream_queue:
        stream_queue.put({"event": event, "data": data})


async def sse_generator(task_id: str, request: Request) -> AsyncGenerator[str, None]:
    stream_queue = get_sse_queue(task_id)
    if stream_queue is None:
        return

    loop = asyncio.get_running_loop()

    try:
        yield _sse_pack(SSEEvent.READY, {})

        while True:
            if await request.is_disconnected():
                break

            try:
                msg = await loop.run_in_executor(None, stream_queue.get, True, 1.0)
            except queue.Empty:
                continue

            event = msg.get("event")
            data = msg.get("data")
            yield _sse_pack(event, data)

            if event == SSEEvent.FINAL:
                break

    except (ConnectionResetError, BrokenPipeError):
        return
    except asyncio.CancelledError:
        raise
    finally:
        remove_sse_queue(task_id)
