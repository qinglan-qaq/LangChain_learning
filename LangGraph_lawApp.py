import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from node import AgentState,Simple_llm_node


load_dotenv()

# 初始化 LLM
llm = ChatOpenAI(
    model=os.getenv('DEEPSEEK_MODEL', 'deepseek-chat'),
    openai_api_key=os.getenv('DEEPSEEK_API_KEY'),
    openai_api_base=os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com'),
    temperature=0.3
)




builder = StateGraph(AgentState)

# 添加节点
builder.add_node("Simple_chat_node", Simple_llm_node)

builder.add_edge(START, "Simple_chat_node")
builder.add_edge("Simple_chat_node", END)

# 编译图
graph = builder.compile()

response = graph.stream({"messages": [HumanMessage(content="泥嚎,我有点累了")]})


reply = response['messages'][-1].content
usage = response['messages'][-1].usage_metadata


print(f"AI回答:{reply}")
print(f"AI用量:{usage}")











