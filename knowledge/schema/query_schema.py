"""Query request/response schemas."""
from typing import List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="查询内容")
    session_id: Optional[str] = Field(None, description="会话ID")
    is_stream: bool = Field(False, description="是否流式返回")


class QueryResponse(BaseModel):
    message: str
    session_id: str
    answer: str
    done_list: List[str] = Field(default_factory=list)
    running_list: List[str] = Field(default_factory=list)
    error: Optional[str] = Field(default=None)
    image_urls: Optional[List[str]] = Field(default=None)


class StreamSubmitResponse(BaseModel):
    message: str
    session_id: str
    task_id: str


