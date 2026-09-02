"""Query service FastAPI application."""
from __future__ import annotations

import os

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from knowledge.core.deps import get_query_service
from knowledge.core.paths import get_front_page_dir
from knowledge.core.security import get_allowed_origins, verify_api_key
from knowledge.schema.query_schema import QueryRequest, QueryResponse, StreamSubmitResponse
from knowledge.services.query_service import QueryService
from knowledge.utils.health_util import readiness_check
from knowledge.utils.logger_util import logger
from knowledge.utils.sse_util import sse_generator
from knowledge.utils.task_util import init_on_startup


def register_routes(app: FastAPI):
    @app.get("/")
    @app.get("/chat.html")
    async def chat_page():
        return FileResponse(os.path.join(get_front_page_dir(), "chat.html"))

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready():
        result = readiness_check()
        if not result["ready"]:
            return JSONResponse(status_code=503, content=result)
        return result

    @app.post("/query")
    async def query(
        request: QueryRequest,
        background_tasks: BackgroundTasks,
        service: QueryService = Depends(get_query_service),
        _auth: None = Depends(verify_api_key),
    ):
        session_id = request.session_id or service.generate_session_id()
        task_id = service.generate_task_id()
        service.submit_query(task_id, request.is_stream)

        if request.is_stream:
            background_tasks.add_task(service.run_query_graph, task_id, session_id, request.query, True)
            return StreamSubmitResponse(message="Query submitted", session_id=session_id, task_id=task_id)

        service.run_query_graph(task_id, session_id, request.query, False)
        info = service.get_task_info(task_id)
        return QueryResponse(
            message="处理完成",
            session_id=session_id,
            answer=info.get("answer", ""),
            done_list=info.get("done_list", []),
            running_list=info.get("running_list", []),
            error=info.get("error"),
            image_urls=info.get("image_urls"),
        )

    @app.get("/status/{task_id}")
    async def get_query_status(
        task_id: str,
        service: QueryService = Depends(get_query_service),
        _auth: None = Depends(verify_api_key),
    ):
        return service.get_task_info(task_id)

    @app.get("/stream/{task_id}")
    async def stream(task_id: str, request: Request):
        return StreamingResponse(sse_generator(task_id, request), media_type="text/event-stream")

    @app.get("/history/{session_id}")
    async def get_history(
        session_id: str,
        limit: int = 50,
        service: QueryService = Depends(get_query_service),
        _auth: None = Depends(verify_api_key),
    ):
        items = service.get_history(session_id, limit)
        return {"session_id": session_id, "items": items}

    @app.delete("/history/{session_id}")
    async def clear_chat_history(
        session_id: str,
        service: QueryService = Depends(get_query_service),
        _auth: None = Depends(verify_api_key),
    ):
        count = service.clear_history(session_id)
        return {"message": "History cleared", "deleted_count": count}


def create_app() -> FastAPI:
    app = FastAPI(title="Query Service", description="知识库查询服务")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_allowed_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    front_page_dir = get_front_page_dir()
    if front_page_dir and os.path.exists(front_page_dir):
        app.mount("/front", StaticFiles(directory=front_page_dir), name="front")

    register_routes(app)

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("未处理异常: {} | path={}", exc, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "内部服务器错误"})

    init_on_startup()
    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run("knowledge.api.query_router:app", host="0.0.0.0", port=8001, reload=False)