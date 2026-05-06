from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str
    content: str


class RetrievedDoc(BaseModel):
    id: Optional[str] = None
    chunk_text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    rerank_score: Optional[float] = None
    score: Optional[float] = None


class EvaluationResult(BaseModel):
    correct: List[RetrievedDoc] = Field(default_factory=list)
    ambiguous: List[RetrievedDoc] = Field(default_factory=list)
    incorrect: List[RetrievedDoc] = Field(default_factory=list)


class QueryRequest(BaseModel):
    query: str


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