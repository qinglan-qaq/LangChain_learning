"""
这里存放节点

TODO:
大模型判断节点
RAG检索节点
llm普通对话节点
路由器少样本llm
CRAG流程:
    RAG检索
    llm评估

"""
import os

from langchain_openai import ChatOpenAI
from langgraph.graph import END
from typing import Annotated
from langchain_core.messages import SystemMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field, ConfigDict

# 初始化 LLM
llm = ChatOpenAI(
    model=os.getenv('DEEPSEEK_MODEL', 'deepseek-chat'),
    openai_api_key=os.getenv('DEEPSEEK_API_KEY'),
    openai_api_base=os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com'),
    temperature=0.3
)


class AgentState(BaseModel):
    # 消息列表，支持追加消息
    messages: Annotated[list, add_messages] = Field(default_factory=list)
    # 用户id
    user_id: str = ""
    # 置信度
    confidence_score: float = 0.0

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


# 设计类 实现llm的调用
class LLM_Invoke:
    def __init__(self, llm, prompt: str):
        self.llm = llm
        self.prompt = prompt

    # state 参数由 LangGraph 自动传入
    def __call__(self, state: AgentState) -> dict:
        message = [
            SystemMessage(content=self.prompt),
            *state.messages
        ]
        response = self.llm.invoke(message)
        return {"messages": [response]}


# 路由跳转节点
def My_router(state: AgentState):
    last_message = state.messages[-1].content

    # 判断是否存在工具调用
    if last_message.tool_calls:
        if last_message.tool_calls[0]["name"] == "google_search_tool":
            return "search_node"
        if last_message.tool_calls[0]["name"] == "google_search_tool":
            pass
        return "none"

    return END


# 设置简单的大模型询问节点
def Simple_llm_node(state: AgentState) -> dict:
    SYSTEM_PROMPT = """
    你是一个智能、可靠、友善的AI助手。请遵循以下原则：

    1. 准确优先：不确定时不编造，明确说“我不知道”或“需要进一步信息”。
    2. 简洁清晰：直接回答问题核心，避免啰嗦和无关内容。
    3. 安全中立：不生成暴力、色情、仇恨或歧视性内容；不提供违法建议。
    4. 结构友好：当需要分点、列表或步骤时，使用清晰的结构（如序号或短横）。
    5. 仅输出回答内容：不要添加“作为AI模型…”之类的开场白或免责声明，除非用户明确询问你的身份。
    """
    # 创建实例,注意参数的顺序
    llm_invoke = LLM_Invoke(llm, SYSTEM_PROMPT)
    # 实例化后传入state,得到dict结果
    response = llm_invoke(state)
    return response


def Evaluate_llm(state: AgentState) -> dict:
    # 带内部分级标准的提示词
    system_prompt = """You are a document relevance evaluator. Assess whether the retrieved document can help answer the user's question.

    Scoring criteria:
    - 1.0: Document directly and specifically answers the question with detailed information
    - 0.7-0.9: Document is moderately relevant with useful related information
    - 0.4-0.6: Document is slightly relevant but contains some useful context
    - 0.0-0.3: Document is irrelevant and unrelated to the question

    Output format: Return a JSON with 'score' (float 0.0-1.0) and 'reason' (brief explanation)."""

    # 创建实例,注意参数的顺序
    llm_invoke = LLM_Invoke(llm, system_prompt)

    # 注意这里可能不需要前者的信息
    response = llm_invoke(state)
    return response


# 告别节点
def farewell(state: AgentState) -> str:
    return "使用完毕,再见~"
