"""
lawApp_LangGraph 统一数据模型

三层 Pydantic 模型体系：
A. 工具返回层 — RetrievedDocument / EvaluationResult
B. 计划执行层 — PlanStep / ToolCallRecord
C. 顶层 — AgentState(Plan & Execute Agent 状态)

所有模块统一从此文件导入模型,保证整个项目的返回格式一致。
"""

from __future__ import annotations
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


#  A. 工具返回层 — Tool Output Models


class RetrievedDocument(BaseModel):
    """单条检索到的法律案例文档块"""

    rank: int = 0
    id: str = ""
    rerank_score: float = 0.0
    hybrid_score: float = 0.0
    year: str = ""
    case_number: str = ""
    case_cause: str = ""
    chunk_text: str = ""


class EvaluationResult(BaseModel):
    """evaluate_case_relevance 工具返回 — CRAG 三档评估"""

    total: int = 0
    correct_count: int = 0
    ambiguous_count: int = 0
    incorrect_count: int = 0
    quality_verdict: str = ""
    correct: List[RetrievedDocument] = Field(default_factory=list)
    ambiguous: List[RetrievedDocument] = Field(default_factory=list)
    incorrect: List[RetrievedDocument] = Field(default_factory=list)
    error: Optional[str] = None


#  B. 计划执行层 — Plan & Execute Models


class PlanStep(BaseModel):
    """计划中的单个步骤"""

    # 步骤编号(从1开始)
    step_id: int
    # 步骤描述LLM 生成的自然语言描述
    description: str
    # 计划执行的工具名称
    tool_name: Optional[str] = None
    # 步骤状态: pending(未执行) | doing(执行中) | done(成功) | failed(失败)
    status: str = "pending"
    # 失败重试次数
    retry_count: int = 0


class ToolCallRecord(BaseModel):
    """单次工具调用的记录"""
    step_id: int
    tool_name: str
    tool_input: Dict[str, Any] = Field(default_factory=dict)
    output: Any = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


#  C. 顶层 — AgentState(Plan & Execute Agent)

class AgentState(BaseModel):
    """Plan & Execute Agent 的全局状态"""

    class Config:
        arbitrary_types_allowed = True

    # 会话标识
    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    user_id: Optional[str] = None

    # 当前请求
    query: str = ""
    # 是否输出为 PDF
    is_pdf_output: bool = False

    # 对话历史(多轮)
    messages: List[Any] = Field(default_factory=list)

    # 计划与执行
    plan: List[PlanStep] = Field(default_factory=list)
    current_step_index: int = 0
    replan_needed: bool = False
    replan_reason: Optional[str] = None

    # 最终回答结果
    final_answer: str = ""

    # 工具调用跟踪
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)

    #  思考链(Chain of Thought)
    reasoning: List[str] = Field(default_factory=list)

    # RAG检索结果
    rag_documents: List[RetrievedDocument] = Field(default_factory=list)

    # 评估结果
    evaluation: EvaluationResult = Field(default_factory=EvaluationResult)

    # 网络检索结果
    web_search_results: List[str] = Field(default_factory=list)

    # 拼装后的 CRAG 上下文
    crag_context: str = ""

    # 扩展搜索与知识
    statute_results: List[Dict[str, Any]] = Field(default_factory=list)

    # 长期记忆检索结果
    memory_results: List[Dict[str, Any]] = Field(default_factory=list)

    # 长期记忆写入确认
    memory_update: Optional[Dict[str, Any]] = None

    # CRAG 管线兼容字段(LangGraph 路由用)
    is_law_questions: bool = False
    is_simple_questions: bool = False
    final_prompts: str = ""
    pdf_path: Optional[str] = None

    # 流程控制
    should_continue: bool = True
    error: Optional[str] = None
