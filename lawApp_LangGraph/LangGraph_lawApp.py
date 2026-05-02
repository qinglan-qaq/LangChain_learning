from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

from lawApp_LangGraph.node.langgraph_nodes import (
    AgentState,
    start_node,
    create_router_node,
    simple_llm_node,
    create_retrieval_node,
    create_evaluate_node,
    create_web_search_node,
    create_analysis_node,
    create_postprocess_node,
    memory_update_node,
    route_by_difficulty,
    route_after_evaluation,
    llm,
)
from lawApp_LangGraph.tools.tools import get_google_search, send_email, markdown_to_pdf

"""

这里主要实现RouterRAG的流程
用户提问 ->
路由器分发判断 ->
(简单问题) -> LLM直接回答
(法律问题) -> CRAG流程 (RAG检索 -> 评估 -> 条件路由 -> 网络搜索/分析 -> PDF生成)

工作流节点说明:
1. start: 初始化状态，加入对话历史
2. router: 判断是否为法律问题及复杂度
3. simple_llm: 简单问题直接回答
4. retrieval: RAG检索法律案例
5. evaluate: 评估检索结果质量
6. web_search: 网络搜索补充资料
7. analysis: 综合分析生成答案
8. postprocess: PDF后处理
9. memory_update: 更新记忆，循环控制
"""

# 获取环境变量
load_dotenv()

tools = [get_google_search, send_email, markdown_to_pdf]

# ===== 初始化RAG服务和PDF生成器 =====
# TODO: 需要根据实际项目注入这些依赖
# rag_service = RAGService(...)  
# pdf_generator = PDFGenerator(...)

# 为演示，先定义占位符
class MockRAGService:
    """占位符 RAG 服务，实际应替换为真实实现"""
    def search_withDenseSparse(self, query, namespace, top_k, rerank_top_n, alpha):
        return []

class MockPDFGenerator:
    """占位符 PDF 生成器"""
    def __call__(self, content, filename):
        return f"/path/to/{filename}"

def mock_search_func(query: str, num: int):
    """占位符网络搜索函数"""
    return [f"搜索结果 {i} for '{query}'" for i in range(num)]

rag_service = MockRAGService()
pdf_generator = MockPDFGenerator()
search_func = mock_search_func

# ===== 创建节点实例（工厂函数需要参数） =====
router_node = create_router_node(llm)
retrieval_node = create_retrieval_node(rag_service)
evaluate_node = create_evaluate_node()
web_search_node = create_web_search_node(llm, search_func)
analysis_node = create_analysis_node(llm)
postprocess_node = create_postprocess_node(pdf_generator)

# simple_llm_node 需要适配（原始函数需要额外的llm参数）
def simple_llm_node_wrapper(state: AgentState) -> dict:
    """LangGraph适配包装：simple_llm_node"""
    simple_llm_node(state, llm)
    return {
        "final_answer": state.final_answer,
        "messages": state.messages
    }




# ===== 构建 LangGraph 图 =====
builder = StateGraph(AgentState)

# 添加所有节点
builder.add_node("start", start_node)
builder.add_node("router", router_node)
builder.add_node("simple_llm", simple_llm_node_wrapper)
builder.add_node("retrieval", retrieval_node)
builder.add_node("evaluate", evaluate_node)
builder.add_node("web_search", web_search_node)
builder.add_node("analysis", analysis_node)
builder.add_node("postprocess", postprocess_node)
builder.add_node("memory_update", memory_update_node)

# ===== 添加边 =====
# 1. 入口：START -> start（初始化状态）
builder.add_edge(START, "start")

# 2. start -> router（进行路由判断）
builder.add_edge("start", "router")

# 3. router 之后的条件分支：判断是否简单问题
builder.add_conditional_edges(
    "router",
    route_by_difficulty,  # 路由函数
    {
        "simple_llm": "simple_llm",     # 简单问题 -> LLM直接回答
        "crag_retrieval": "retrieval"   # 复杂问题 -> RAG检索
    }
)

# 4. simple_llm 直接到后处理
builder.add_edge("simple_llm", "postprocess")

# 5. retrieval -> evaluate（评估检索结果）
builder.add_edge("retrieval", "evaluate")

# 6. evaluate 之后的条件分支：判断是否需要网络搜索
builder.add_conditional_edges(
    "evaluate",
    route_after_evaluation,  # 路由函数
    {
        "analysis": "analysis",      # 文档足够 -> 直接分析
        "web_search": "web_search"   # 文档不足 -> 网络搜索补充
    }
)

# 7. web_search -> analysis（获取补充资料后进行分析）
builder.add_edge("web_search", "analysis")

# 8. analysis -> postprocess（分析完成后后处理）
builder.add_edge("analysis", "postprocess")

# 9. postprocess -> memory_update（后处理完成后更新记忆）
builder.add_edge("postprocess", "memory_update")

# 10. memory_update -> END（完成流程）
builder.add_edge("memory_update", END)

# ===== 编译图 =====
graph = builder.compile()

# ===== 执行示例 =====
if __name__ == "__main__":
    # 测试查询
    test_query = "泥嚎,我有点累了"
    
    print(f"输入问题: {test_query}\n")
    
    # 流式执行
    final_state = None
    for chunk in graph.stream({"query": test_query, "messages": []}):
        print(f"步骤输出: {chunk}\n")
        final_state = chunk
    
    # 提取最终答案
    if final_state and "final_answer" in final_state:
        print(f"\n最终答案: {final_state['final_answer']}")
    
    print("\n工作流执行完成！")
