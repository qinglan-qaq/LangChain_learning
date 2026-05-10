from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from lawApp_LangGraph.state import EvaluationResult

# 基础问题信息结构
class Message(BaseModel):
    role: str
    content: str

# 请求体模型
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None  # 作为 thread_id 实现短期记忆,不传则自动生成
    message_id: Optional[str] = None
    timestamp: Optional[str] = None


# 响应体模型
class QueryResponse(BaseModel):
    query: str
    final_answer: str
    session_id: str = ""  # 当前会话 ID,客户端可保存用于多轮对话
    messages: List[Message] = Field(default_factory=list)
    crag_context: Optional[str] = None
    pdf_path: Optional[str] = None
    is_law_questions: bool = False
    is_simple_questions: bool = False
    evaluation: EvaluationResult = Field(default_factory=EvaluationResult)
    web_search_results: List[str] = Field(default_factory=list)
    output_format: str = "text"
