"""File upload response schema."""
from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    message: str = Field(..., description="上传结果")
    task_id: str = Field(..., description="任务ID")
