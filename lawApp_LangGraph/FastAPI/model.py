from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# 基础问题信息结构
class Message(BaseModel):
    role: str
    content: str

# 检索到的文档结构
class RetrievedDoc(BaseModel):
    id: Optional[str] = None
    chunk_text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    rerank_score: Optional[float] = None
    score: Optional[float] = None

# 评估结果结构
class EvaluationResult(BaseModel):
    correct: List[RetrievedDoc] = Field(default_factory=list)
    ambiguous: List[RetrievedDoc] = Field(default_factory=list)
    incorrect: List[RetrievedDoc] = Field(default_factory=list)

# 请求体模型
class QueryRequest(BaseModel):
    query: str
    message_id: Optional[str] = None
    timestamp: Optional[str] = None


# 响应体模型
class QueryResponse(BaseModel):
    query: str
    final_answer: str
    messages: List[Message] = Field(default_factory=list)
    crag_context: Optional[str] = None
    pdf_path: Optional[str] = None
    is_law_questions: bool = False
    is_simple_questions: bool = False
    evaluation: EvaluationResult = Field(default_factory=EvaluationResult)
    web_search_results: List[str] = Field(default_factory=list)
    output_format: str = "text"
