"""
RAG Agent 工具集
将 CRAG 流程拆解为 3 个 @tool,供 Plan & Execute Agent 自主调用

    retrieve_legal_knowledge → evaluate_case_relevance → analyze_legal_issue
                ↑                        ↑                       ↑
            混合检索+重排序            三档质量评估           LLM 法律分析生成

Agent 可据此自主决策:检索 → 评估 → (如需)联网搜索 → 生成分析

所有工具直接返回 dict,调用方按需自行转换.
"""

import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


load_dotenv()

# 懒加载单例 只在 tool 第一次被 invoke 时才初始化

_rag_service = None
_llm = None


# 私有方法新建一个单例 RAG_service 实例,供 retrieve_legal_knowledge 工具调用
def _get_rag_service():
    global _rag_service
    if _rag_service is None:
        from lawApp_LangGraph.RAG_service.RAG_program import RAG_service

        _rag_service = RAG_service(
            # TODO: 生产环境改为从安全配置中心获取,不要直接用环境变量 需要预先配置好
            index_name=os.getenv("PINECONE_INDEX_NAME", "pinecone-test-lawapp"),
            api_key=os.getenv("PINECONE_API_KEY"),
            cloud=os.getenv("PINECONE_CLOUD", "aws"),
            region=os.getenv("PINECONE_REGION", "us-east-1"),
        )
    return _rag_service


# 私有方法新建一个单例 LLM 实例,供 analyze_legal_issue 工具调用
def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=os.getenv("DEEPSEEK_MODEL"),
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            openai_api_base=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            temperature=0.4,
        )
    return _llm


# Tool 1: 法律案例检索


@tool
def retrieve_legal_knowledge(
    query: str,
    top_k: int = 30,
    rerank_top_n: int = 10,
    alpha: float = 0.5,
    namespace: str = "legal_cases",
) -> dict:
    """从法律案例库中检索相关判例.使用混合检索(语义向量 + BM25 关键词匹配)与
    CrossEncoder 重排序,返回最相关的案例内容及其相关性评分.

    适用场景:
    - 查找与特定法律问题相关的历史判例
    - 获取类似案件的裁判要旨
    - 法律问题需要案例支撑时

    参数:
    query: 法律问题查询语句,中文
    top_k: 初始召回数量,默认 30
    rerank_top_n: 重排序后返回数量,默认 10
    alpha: 混合检索中的权重参数,默认 0.5
    namespace: 检索的命名空间,默认 "legal_cases"
    返回:
    结构化 dict,含 status / count / results 字段,
    每个 result 为 RetrievedDocument 格式(case_number / case_cause / rerank_score / chunk_text 等)
    """
    service = _get_rag_service()
    matches = service.search_withDenseSparse(
        query=query,
        namespace= namespace ,
        top_k=top_k,
        rerank_top_n=rerank_top_n,
        alpha=alpha,
    )

    if not matches:
        return {"status": "empty", "message": "未检索到相关案例"}

    results = []
    for i, match in enumerate(matches):
        meta = match.metadata or {}
        results.append(
            {
                "rank": i + 1,
                "id": match.id,
                "rerank_score": round(getattr(match, "rerank_score", 0.0), 4),
                "hybrid_score": round(match.score, 4)
                if hasattr(match, "score")
                else 0.0,
                "year": meta.get("year", ""),
                "case_number": meta.get("case_number", ""),
                "case_cause": meta.get("case_cause", ""),
                "chunk_text": meta.get("chunk_text", "")[:500],
            }
        )
    return {"status": "success", "count": len(results), "results": results}


# Tool 2: 检索质量评估 (CRAG 三档)

CORRECT_THRESHOLD = 0.7
INCORRECT_THRESHOLD = 0.3
MIN_QUALITY_DOCS = 3


@tool
def evaluate_case_relevance(
    documents: List[Dict[str, Any]],
) -> dict:
    """评估检索到的案例与用户问题的相关程度,按评分分为三档:
    - correct (高质量):     rerank_score >= 0.7,可直接用于法律分析
    - ambiguous (中等质量): 0.3 <= score < 0.7,可参考但需谨慎
    - incorrect (低质量):   score < 0.3,不建议使用

    评估报告会明确告知检索质量是否「充足」或「不足,建议进行网络搜索补充」.
    Agent 应据此决定是否调用 get_google_search 进行联网补充.

    参数:
    documents: retrieve_legal_knowledge 返回结果中的 results 列表
    每项含 rerank_score / chunk_text / case_number 等字段

    返回:
    结构化 dict,含 correct/ambiguous/incorrect 分类及 quality_verdict
    """
    if not documents:
        return {"error": "输入为空,没有可评估的文档"}

    correct, ambiguous, incorrect = [], [], []

    for doc in documents:
        score = doc.get("rerank_score", 0.0)
        if score >= CORRECT_THRESHOLD:
            correct.append(doc)
        elif score >= INCORRECT_THRESHOLD:
            ambiguous.append(doc)
        else:
            incorrect.append(doc)

    total_usable = len(correct) + len(ambiguous)
    quality_verdict = (
        "充足"
        if len(correct) >= MIN_QUALITY_DOCS or total_usable >= MIN_QUALITY_DOCS
        else "不足,建议进行网络搜索补充"
    )

    return {
        "total": len(documents),
        "correct_count": len(correct),
        "ambiguous_count": len(ambiguous),
        "incorrect_count": len(incorrect),
        "quality_verdict": quality_verdict,
        "correct": correct,
        "ambiguous": ambiguous,
        "incorrect": incorrect,
    }


# Tool 3: 法律分析生成

LEGAL_ANALYSIS_PROMPT = PromptTemplate.from_template(
    "你是资深法律顾问.请根据以下资料回答用户问题.\n\n"
    "资料:\n{context}\n\n"
    "用户问题: {query}\n\n"
    "请进行详细分析,包括:\n"
    "1. 涉及的法律概念和法条\n"
    "2. 相关案例的参考价值\n"
    "3. 法律建议和注意事项\n"
    "4. 如需进一步确认的事项"
)


@tool
def analyze_legal_issue(
    query: str,
    correct_cases: Optional[List[Dict[str, Any]]] = None,
    ambiguous_cases: Optional[List[Dict[str, Any]]] = None,
    web_results: Optional[List[str]] = None,
) -> dict:
    """基于法律案例和网络资料,生成专业的法律分析和建议.
    整合高质量案例、中等相关案例和外部网络资料作为分析依据.

    典型调用流程:
    1. 先调用 retrieve_legal_knowledge 获取案例列表
    2. 再调用 evaluate_case_relevance 评估质量
    3. 如 quality_verdict 为"不足",则调用 get_google_search 联网补充
    4. 最后调用本工具,传入 correct_cases / ambiguous_cases / web_results 生成最终分析

    参数:
    query: 用户的法律问题
    correct_cases: 评估为 high-quality 的案例列表(来自 evaluate_case_relevance 的 correct 字段)
    ambiguous_cases: 评估为 medium-quality 的案例列表(来自 evaluate_case_relevance 的 ambiguous 字段)
    web_results: 网络搜索结果的文本列表(来自 get_google_search 的返回值),可选

    返回:
    结构化 dict,含 final_answer / crag_context / sources
    """
    llm = _get_llm()

    correct_cases = correct_cases or []
    ambiguous_cases = ambiguous_cases or []
    web_results = web_results or []

    parts = []
    sources = []
    # 优先使用高质量案例,其次是中等相关案例,最后是网络资料 解析返回结果组装上下文内容
    for doc in correct_cases:
        cn = doc.get("case_number", "")
        yr = doc.get("year", "")
        parts.append(f"[高相关案例 | 案号:{cn} | {yr}年]\n{doc.get('chunk_text', '')}")
        if cn:
            sources.append(f"案例: {cn} ({yr})")
    for doc in ambiguous_cases:
        cn = doc.get("case_number", "")
        yr = doc.get("year", "")
        parts.append(
            f"[中等相关案例 | 案号:{cn} | {yr}年]\n{doc.get('chunk_text', '')}"
        )
        if cn:
            sources.append(f"案例: {cn} ({yr})")

    for i, snippet in enumerate(web_results, 1):
        content = (
            snippet
            if isinstance(snippet, str)
            else snippet.get("content", str(snippet))
        )
        parts.append(f"[外部网络资料{i}]\n{content}")
        sources.append(f"网络资料{i}")

    context = "\n\n---\n\n".join(parts) if parts else "暂无相关资料"

    chain = LEGAL_ANALYSIS_PROMPT | llm | StrOutputParser()

    # 结合上下文内容和用户问题进行法律分析生成
    answer = chain.invoke({"context": context, "query": query})

    return {
        "final_answer": answer,
        "crag_context": context,
        "sources": sources,
    }
