from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ── 请求 ──────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000, description="用户问题")
    session_id: Optional[str] = Field(
        default=None, description="会话 ID,传入可持续多轮对话;不传则新建"
    )


# ── 响应 ──────────────────────────────────────────

class ToolInfo(BaseModel):
    """工具元信息"""
    name: str
    description: str


class SourceInfo(BaseModel):
    """回答引用的来源"""
    case_number: str = ""
    year: str = ""
    snippet: str = ""


class QueryResponse(BaseModel):
    query: str
    session_id: str
    final_answer: str
    sources: List[SourceInfo] = Field(default_factory=list)
    tool_calls: List[str] = Field(default_factory=list)
    reasoning: List[str] = Field(default_factory=list)



