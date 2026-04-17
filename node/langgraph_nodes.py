"""
这里存放节点

TODO:
大模型判断节点
RAG检索节点
llm普通对话节点


"""
from typing import Annotated
from langchain_core.messages import SystemMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field, ConfigDict


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
