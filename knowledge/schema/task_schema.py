"""Task status schemas."""
from typing import List
from typing import Optional

from pydantic import BaseModel, Field


class TaskStatusResponse(BaseModel):
    status: str = Field(..., description="任务状态")
    done_list: List[str] = Field(..., description="已完成节点列表")
    running_list: List[str] = Field(..., description="正在运行节点列表")
    error: Optional[str] = Field(None, description="失败原因")
