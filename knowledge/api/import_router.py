"""Import service FastAPI application."""
from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from knowledge.core.deps import get_import_file_service, get_task_service
from knowledge.core.paths import get_front_page_dir
from knowledge.core.security import get_allowed_origins, verify_api_key
from knowledge.schema.task_schema import TaskStatusResponse
from knowledge.schema.upload_schema import UploadResponse
from knowledge.services.file_import_service import ImportFileService
from knowledge.services.task_service import TaskService
from knowledge.utils.health_util import readiness_check
from knowledge.utils.logger_util import logger
from knowledge.utils.task_util import init_on_startup


def register_router(app: FastAPI):
    @app.get("/")
    @app.get("/import")
    @app.get("/import.html")
    async def import_root():
        return FileResponse(path=os.path.join(get_front_page_dir(), "import.html"))

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready():
        result = readiness_check()
        if not result["ready"]:
            return JSONResponse(status_code=503, content=result)
        return result

    @app.post("/upload", response_model=UploadResponse)
    async def upload_file_endpoint(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        service: ImportFileService = Depends(get_import_file_service),
        _auth: None = Depends(verify_api_key),
    ):
        allowed_extensions = {".pdf", ".md", ".docx"}
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"不支持的文件类型: {suffix or '未知'}")
        if service.check_duplicate_file(file):
            raise HTTPException(status_code=409, detail="文件已导入，跳过重复导入")
        task_id, file_dir, import_file_path = service.process_upload_file(file)
        background_tasks.add_task(service.run_import_graph, task_id, file_dir, import_file_path)
        return UploadResponse(message="文件上传成功", task_id=task_id)

    @app.get("/status/{task_id}", response_model=TaskStatusResponse)
    async def get_status_endpoint(
        task_id: str,
        task_service: TaskService = Depends(get_task_service),
        _auth: None = Depends(verify_api_key),
    ):
        task_info = task_service.get_task_info(task_id)
        return TaskStatusResponse(**task_info)

    @app.delete("/document/{file_title}")
    async def delete_document_endpoint(
        file_title: str,
        service: ImportFileService = Depends(get_import_file_service),
        _auth: None = Depends(verify_api_key),
    ):
        try:
            return service.delete_document(file_title)
        except Exception as exc:
            logger.error("删除文档失败: {}", exc)
            raise HTTPException(status_code=404, detail=str(exc))


def create_app() -> FastAPI:
    app = FastAPI(title="Import Service", description="知识库导入服务")
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

    register_router(app)

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("未处理异常: {} | path={}", exc, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "内部服务器错误"})

    init_on_startup()
    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run("knowledge.api.import_router:app", host="0.0.0.0", port=8000, reload=False)