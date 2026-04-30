from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

from lawApp_LangGraph.node.langgraph_nodes import Simple_llm_node, AgentState, Router_node
from lawApp_LangGraph.tools.tools import get_google_search, send_email, markdown_to_pdf

"""

这里主要实现RouterRAG的流程
用户提问 ->
路由器分发判断 ->
(简单问题) -> LLM直接回答
(法律问题) -> CRAG流程

需要:
路由节点(少样本LLM构建)
LLM直接回答节点
RAG检索节点
CRAG节点
条件边
"""

# 获取环境变量
load_dotenv()

tools = [get_google_search, send_email, markdown_to_pdf]





builder = StateGraph(AgentState)

# 添加节点
builder.add_node("Simple_chat_node", Simple_llm_node)
builder.add_node("Router", Router_node)
builder.add_node("RAG", RAG_node)

# Router RAG形式
builder.add_edge(START, "Simple_chat_node")

# 条件跳转边, 对应不同难度的问题
builder.add_conditional_edges(
    "Simple_chat_node",
    Evalue_func,
    {
        "simple":"llm_node",
        "normal":"RAG_node",
        "hard":"CRAG_node"
    }
)
builder.add_edge("Simple_chat_node", END)

# 编译图
graph = builder.compile()

# 执行流式调用
final_state = None
for chunk in graph.stream({"messages": [HumanMessage(content="泥嚎,我有点累了")]}):
    print(chunk)
    final_state = chunk  # 不断更新，循环结束后 final_state 就是最后一个块

# 从最终状态中提取消息和用量
reply = final_state['messages'][-1].content
usage = final_state['messages'][-1].response_metadata.get('token_usage',
                                                          {})  # usage_metadata 可能不存在，这里从 response_metadata 取

print(f"AI回答: {reply}")
print(f"AI用量: {usage}")
