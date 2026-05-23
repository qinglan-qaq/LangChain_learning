"""
RAG Agent 工具集
将 CRAG 流程拆解为 3 个 @tool,供 Plan & Execute Agent 自主调用

    retrieve_legal_knowledge → evaluate_case_relevance → analyze_legal_issue
                ↑                        ↑                       ↑
            混合检索+重排序            三档质量评估           LLM 法律分析生成

Agent 可据此自主决策:检索 → 评估 → (如需)联网搜索 → 生成分析

"""

import os
import time
from typing import Any, List, Optional
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from lawApp_LangGraph.FastAPI.logging import (
    tool as tool_log,
    rag as rag_log,
    system as sys_log,
)
from lawApp_LangGraph.state import (
    LawsResult,
    WebSearchResult,
    simpleRetrievedDocument,
    PromptsRecord,
)
from langsmith import traceable


load_dotenv()

# 懒加载单例 只在 tool 第一次被 invoke 时才初始化

_rag_service = None
_llm = None


# 私有方法新建一个单例 RAG_service 实例,供 retrieve_legal_knowledge 工具调用
@traceable(run_type="llm", name="RAG_Service初始化")
def _get_rag_service():
    global _rag_service
    if _rag_service is None:
        from lawApp_LangGraph.RAG_service.RAG_program import RAG_service

        sys_log.info(
            "初始化 RAG_service (冷启动)",
            detail="首次加载嵌入模型 + 重排序模型 + BM25 编码器",
        )
        _rag_service = RAG_service(
            # TODO: 生产环境改为从安全配置中心获取,不要直接用环境变量 需要预先配置好
            index_name=os.getenv("PINECONE_INDEX_NAME", "pinecone-test-lawapp"),
            api_key=os.getenv("PINECONE_API_KEY"),  # type: ignore
            cloud=os.getenv("PINECONE_CLOUD", "aws"),
            region=os.getenv("PINECONE_REGION", "us-east-1"),
        )

        # RAG_service 初始化时会自动创建索引并加载模型,这里等待索引创建完成并返回状态
        _rag_service.create_index(wait_for_completion=True)
        _rag_service.get_index_stats()

        sys_log.info("RAG_service 初始化完成", result="嵌入 + 重排序 + BM25 模型已就绪")

    return _rag_service


# 私有方法新建一个单例 LLM 实例,供 analyze_legal_issue 工具调用
@traceable(run_type="llm", name="LLM_实例获取")
def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=os.getenv("DEEPSEEK_FLASH_MODEL"),
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            openai_api_base=os.getenv("DEEPSEEK_BASE_URL"),
            temperature=0.4,
        )
    return _llm


# Tool 1: 法律案例检索

@tool
@traceable(run_type="tool", name="tool_法律知识检索")
def retrieve_legal_knowledge(
    query: str,
    top_k: int = 20,
    rerank_top_n: int = 5,
    alpha: float = 0.4,
    namespace: str = "law_cases",
) -> dict:
    """从法律案例库中检索相关判例.使用混合检索(语义向量 + BM25 关键词匹配)与
    CrossEncoder 重排序,返回最相关的案例内容及其相关性评分.

    适用场景:
    - 查找与特定法律问题相关的历史判例
    - 获取类似案件的裁判要旨
    - 法律问题需要案例支撑时

    参数:
    query: 法律问题查询语句,中文
    top_k: 初始召回数量,默认 20
    rerank_top_n: 重排序后返回数量,默认 5 (从 top_k 中选出最相关的 5 条)
    alpha: 混合检索中的权重参数,默认 0.4 (越接近 1 越重视语义匹配,越接近 0 越重视关键词匹配)
    namespace: 检索的命名空间,默认 "law_cases"
    返回:
    结构化 dict,含 status / rag_documents 字段,
    每个文档为 RetrievedDocument 格式(case_number / case_cause / rerank_score / chunk_text 等)
    """
    t0 = time.time()
    tool_log.info(
        "→ 调用工具: retrieve_legal_knowledge",
        detail=f"query={query[:60]} | top_k={top_k} | alpha={alpha} | ns={namespace}",
    )

    service = _get_rag_service()
    matches = service.search_withDenseSparse(
        query=query,
        namespace=namespace,
        top_k=top_k,
        rerank_top_n=rerank_top_n,
        alpha=alpha,
    )

    if not matches:
        tool_log.info(
            "← 工具返回: retrieve_legal_knowledge",
            detail="未检索到相关案例",
            result=f"elapsed={time.time() - t0:.2f}s",
        )
        return {"status": "empty", "message": "未检索到相关案例", "rag_documents": []}

    results = []
    for match in matches:
        meta = match.metadata or {}
        results.append(
            {
                "rank": 0,  # 排序后重新分配
                "id": match.id,
                "hybrid_score": round(match.score, 4)
                if hasattr(match, "score")
                else 0.0,
                "year": meta.get("year", ""),
                "case_number": meta.get("case_number", ""),
                "case_cause": meta.get("case_cause", ""),
                "chunk_text": meta.get("chunk_text", "")[:500],
            }
        )

    # 按 hybrid_score 降序排列,重新分配 rank
    results.sort(key=lambda r: r["hybrid_score"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    top_score = results[0]["hybrid_score"] if results else 0
    
    tool_log.info(
        "← 工具返回: retrieve_legal_knowledge",
        detail=f"返回{len(results)}条案例 | top_score={top_score:.3f}",
        result=f"elapsed={time.time() - t0:.2f}s",
    )
    return {"status": "success", "count": len(results), "rag_documents": results}


# Tool 2: 检索质量评估 (CRAG 三档)

CORRECT_THRESHOLD = 0.6
INCORRECT_THRESHOLD = 0.3
MIN_QUALITY_DOCS = 3


@tool
@traceable(run_type="tool", name="tool_案例相关性评估")
def evaluate_case_relevance(
    documents: list[dict[str, Any]],
) -> dict:
    """评估检索到的案例与用户问题的相关程度,按评分分为三档:
    - correct (高质量):     rerank_score >= 0.7,可直接用于法律分析
    - ambiguous (中等质量): 0.3 <= score < 0.7,可参考但需谨慎
    - incorrect (低质量):   score < 0.3,不建议使用

    评估报告会明确告知检索质量是否「充足」或「不足,建议进行网络搜索补充」.
    Agent 应据此决定是否调用 get_google_search 进行联网补充.

    参数:
    documents: retrieve_legal_knowledge 返回结果中的 rag_documents 列表
    每项含 rerank_score / chunk_text / case_number 等字段

    返回:
    结构化 dict,含 evaluation 键,其值为 correct/ambiguous/incorrect 分类及 quality_verdict
    """

    t0 = time.time()
    tool_log.info(
        "→ 调用工具: evaluate_case_relevance",
        detail=f"input_docs={len(documents)}",
    )

    if not documents:
        tool_log.info(
            "← 工具返回: evaluate_case_relevance",
            detail="输入为空",
            result="verdict=不足",
        )
        return {
            "evaluation": {
                "error": "输入为空,没有可评估的文档",
                "total": 0,
                "correct_count": 0,
                "ambiguous_count": 0,
                "incorrect_count": 0,
                "quality_verdict": "不足,建议进行网络搜索补充",
                "correct": [],
                "ambiguous": [],
                "incorrect": [],
            }
        }

    correct, ambiguous, incorrect = [], [], []

    for doc in documents:
        score = doc.get("rerank_score")
        sdoc = simpleRetrievedDocument(
            id=doc.get("id", ""),
            year=str(doc.get("year", "")),
            case_number=doc.get("case_number", ""),
            case_cause=doc.get("case_cause", ""),
            chunk_text=doc.get("chunk_text", ""),
        )
        if score >= CORRECT_THRESHOLD:
            correct.append(sdoc)
        elif score >= INCORRECT_THRESHOLD:
            ambiguous.append(sdoc)
        else:
            incorrect.append(sdoc)

    total_usable = len(correct) + len(ambiguous)
    quality_verdict = (
        "充足"
        if len(correct) >= MIN_QUALITY_DOCS or total_usable >= MIN_QUALITY_DOCS
        else "不足,建议进行网络搜索补充"
    )

    tool_log.info(
        "← 工具返回: evaluate_case_relevance",
        detail=f"correct={len(correct)} | ambiguous={len(ambiguous)} | incorrect={len(incorrect)}",
        result=f"verdict={quality_verdict} | elapsed={time.time() - t0:.2f}s",
    )
    return {
        "evaluation": {
            "total": len(documents),
            "correct_count": len(correct),
            "ambiguous_count": len(ambiguous),
            "incorrect_count": len(incorrect),
            "quality_verdict": quality_verdict,
            "correct": correct,
            "ambiguous": ambiguous,
            "incorrect": incorrect,
        }
    }


# Tool 3: 法律分析生成


# 风骚律师中小金(KIM)的人设提示词
LEGAL_ANALYSIS_PROMPT_Kim = PromptTemplate.from_template(
    """
    # Role: kim Wexler (《风骚律师》中的冷静理智的资深律师)

    ## Profile
    你是一位经验丰富、务实沉稳的法律顾问,精通中国法律体系.
    你的当事人带着真实的法律困惑来找你,他们可能是普通人,不懂法条、容易焦虑.
    你的职责是用专业和冷静帮他们看清局面、找到出路.

    ## Tone and Style
    1. 清醒且坚定:首先自我介绍,简短表达强大的业务能力,面对当事人的情绪宣泄或抱怨,给予简短有力的共情,随后立刻切入法律事实和可行方案.
    2. 极度务实:直击痛点,不谈虚无缥缈的道德评判,只谈证据、权利、程序、风险.
    3. 沉稳的掌控感:逻辑严密,用专业度给当事人安全感.用普通人听得懂的大白话来分析问题.

    ## 分析要求
    1. 明确法律定性:一句话点出用户问题涉及的核心法律关系(如合同纠纷、侵权、婚姻财产分割、劳动争议等).
    2. 引用法条依据:如参考材料中有「相关法条」,优先引用具体法条原文作为法律依据,明确告知出处(法规名称+条款号).
    3. 引用参考案例:从提供的参考资料中提取相关判例,用案例说明法院的裁判思路,不要照搬原文,概括要点,部分引用案例细节(如案情、争议焦点、法院观点)来佐证分析.
    4. 引用参考案例:从提供的参考资料中提取相关判例,用案例说明法院的裁判思路,不要照搬原文,概括要点.
    5. 指出关键风险:用户可能没意识到的法律陷阱、证据短板、时效问题.
    6. 给出可行建议:具体的下一步行动,可以做什么、应该注意什么、可以找谁.
    7. 区分确定与不确定:明确哪些结论有充分依据,哪些还需进一步核实.
    8. 使用完整的Markdown格式输出,结构清晰,层次分明.
    9. 最后给出具体参考了那些资料(如「参考了3条案例和2条法条」),并列出它们的编号或标题.

    ## 参考材料
    {context}

    ## 用户问题
    {query}

    ## 你的回答
    """
)


# Saul Goodman 的人设提示词
LEGAL_ANALYSIS_PROMPT_Saul = PromptTemplate.from_template(
    """
    # Role: Saul Goodman (《风骚律师》中的传奇边缘律师)

    ## Profile
    你现在是 Saul Goodman(曾用名 Jimmy McGill).你不是那种坐在高档写字楼里按小时收费的刻板律师,你是专门帮陷入绝境的普通人解决“难言之隐”的街头斗士.你精通人性弱点,深谙法律漏洞.在婚姻家庭纠纷中,你信奉“进攻是最好的防守”,擅长用极其戏剧化、边缘化、甚至不择手段的奇招,在最短时间内逼对方崩溃并割肉和解.

    ## Tone and Style
    1. 市侩、幽默且极具煽动性:说话语速快,充满美式俚语、夸张的比喻和黑色幽默.常挂着自信、甚至带点无赖的微笑.
    2. 江湖气的安抚:喜欢用“Honey”、“My friend”、“Pal”等亲昵称呼.用一种“天塌下来有哥们替你顶着”的江湖气让当事人放松.
    3. 金句频出:标志性口号“Better Call Saul!”(这句口号不能翻译成其他语言)挂在嘴边.说话极具感染力,擅长把严肃的法律条文解构成通俗的利益博弈.

    ## Domain Expertise & Logic (Saul的婚姻案件办事逻辑)
    1. 声誉勒索(抓住软肋):对方有钱、有地位、有公司？太好了,这说明他输不起.针对他的软肋(如税务问题、商业机密、个人名誉)做文章,逼他主动求和.
    2. “演一出好戏”(边缘手段):规规矩矩查财产太慢.指导User如何去“设局”——比如雇人去勾引对方拍下更劲爆的证据,或者利用媒体、大字报、公司闹事等手段,让对方在法庭之外先社会性死亡.
    3. 灰色信息搜集:暗示或明示User去用一些非常规手段(如翻看垃圾桶、偷偷安装定位、找黑客查云端账号),只要能拿到线索,Saul就有办法把它包装成“合法呈堂证供”.
    4. 实用主义分账:极其看重实际到手的利益.只要能多拿钱,面子不重要,程序正义也不重要.

    ## Workflow (咨询与审理分析步骤)
    1. 痛点放大:听完User的哭诉后,先用夸张的语气痛骂对方,瞬间和User结成情感同盟.
    2. 寻找“阿喀琉斯之踵”:略过死板的法律条文,直接问对方最怕失去什么(钱、公司、名声、还是情妇).
    3. 抛出“惊天计划”(The Scheme):拿出一个听起来疯狂、游走在法律边缘、但绝对有效的反击方案.
    4. 洗脑式结语:用充满诱惑力和绝对自信的话,让User觉得跟着你就能稳操胜券.

    ## 参考材料
    {context}

    ## 用户问题
    {query}

    ## 你的回答
"""
)


def _extract_doc_fields(doc) -> tuple:
    """从 dict 或 simpleRetrievedDocument 中提取案号、年份、文本."""
    if isinstance(doc, dict):
        return doc.get("case_number", ""), str(doc.get("year", "")), doc.get("chunk_text", "")
    return (
        getattr(doc, "case_number", ""),
        str(getattr(doc, "year", "")),
        getattr(doc, "chunk_text", ""),
    )


def _to_simple_doc(doc) -> simpleRetrievedDocument:
    """将 dict 或对象转换为 simpleRetrievedDocument."""
    if isinstance(doc, simpleRetrievedDocument):
        return doc
    if isinstance(doc, dict):
        return simpleRetrievedDocument(
            id=doc.get("id", ""),
            year=str(doc.get("year", "")),
            case_number=doc.get("case_number", ""),
            case_cause=doc.get("case_cause", ""),
            chunk_text=doc.get("chunk_text", ""),
        )
    return simpleRetrievedDocument(
        id=getattr(doc, "id", ""),
        year=str(getattr(doc, "year", "")),
        case_number=getattr(doc, "case_number", ""),
        case_cause=getattr(doc, "case_cause", ""),
        chunk_text=getattr(doc, "chunk_text", ""),
    )


@tool
@traceable(run_type="tool", name="tool_法律问题分析")
def analyze_legal_issue(
    query: str,
    correct_cases: Optional[list[dict[str, Any]]] = None,
    ambiguous_cases: Optional[list[dict[str, Any]]] = None,
    web_results: Optional[List[str]] = None,
    law_results: Optional[List[dict[str, Any]]] = None,
) -> dict:
    """基于法律案例、法律条文和网络资料,生成专业的法律分析和建议.
    整合高质量案例、中等相关案例、相关法条和外部网络资料作为分析依据.

    典型调用流程:
    1. 先调用 retrieve_legal_knowledge 获取案例列表
    2. 再调用 evaluate_case_relevance 评估质量
    3. 如需要法律条文依据,调用 fetch_laws 获取相关法条原文
    4. 如 quality_verdict 为"不足",则调用 get_google_search 联网补充
    5. 最后调用本工具,传入 correct_cases / ambiguous_cases / law_results / web_results 并生成最终分析

    参数:
    query: 用户的法律问题
    correct_cases: 评估为 high-quality 的案例列表(来自 evaluate_case_relevance 的 correct 字段)
    ambiguous_cases: 评估为 medium-quality 的案例列表(来自 evaluate_case_relevance 的 ambiguous 字段)
    web_results: 网络搜索结果的文本列表(来自 get_google_search 的返回值),可选
    law_results: 相关法律条文列表(来自 fetch_laws 的 law_results 字段)

    返回:
    结构化 dict,含 final_answer / crag_context / sources / prompts_record
    """
    t0 = time.time()
    correct_n = len(correct_cases or [])
    ambig_n = len(ambiguous_cases or [])
    web_n = len(web_results or [])
    law_n = len(law_results or [])
    tool_log.info(
        "→ 调用工具: analyze_legal_issue",
        detail=f"query={query[:60]} | correct={correct_n} | ambiguous={ambig_n} | web={web_n} | law={law_n}",
    )

    llm = _get_llm()

    correct_cases = correct_cases or []
    ambiguous_cases = ambiguous_cases or []
    web_results = web_results or []
    law_results = law_results or []

    parts = []
    sources = []

    # 法律条文 (优先展示,作为权威依据)
    laws_for_record: list[LawsResult] = []
    for i, law in enumerate(law_results, 1):
        if isinstance(law, dict):
            title = law.get("law_title", "")
            article_number = law.get("article_number", "")
            content = law.get("content", "")
            laws_for_record.append(LawsResult(
                law_title=title,
                chapter=law.get("chapter", ""),
                article_number=article_number,
                content=content,
            ))
        else:
            title = getattr(law, "law_title", "")
            article_number = getattr(law, "article_number", "")
            content = getattr(law, "content", "")
            laws_for_record.append(law)
        parts.append(f"法条: {title} 第{article_number}条\n{content}")
        sources.append(f"法条: {title} 第{article_number}条")

    # 案例部分,按照质量分档展示,高质量的案例会被 LLM 优先关注
    eval_docs_for_record: list[simpleRetrievedDocument] = []
    for doc in correct_cases:
        if doc:
            cn, yr, chunk_text = _extract_doc_fields(doc)
            parts.append(
                f"[高相关案例 | 案号:{cn} | {yr}年]\n{chunk_text}"
            )
            if cn:
                sources.append(f"案例: {cn} ({yr})")
            eval_docs_for_record.append(_to_simple_doc(doc))
    # 中等相关的案例也可以参考,但要明确标注质量较低
    for doc in ambiguous_cases:
        if doc:
            cn, yr, chunk_text = _extract_doc_fields(doc)
            parts.append(
                f"[中等相关案例 | 案号:{cn} | {yr}年]\n{chunk_text}"
            )
            if cn:
                sources.append(f"案例: {cn} ({yr})")
            eval_docs_for_record.append(_to_simple_doc(doc))

    web_for_record: list[WebSearchResult] = []
    for i, snippet in enumerate(web_results, 1):
        if isinstance(snippet, str):
            content = snippet
        elif isinstance(snippet, dict):
            content = snippet.get("snippet", snippet.get("content", str(snippet)))
            web_for_record.append(WebSearchResult(
                title=snippet.get("title", ""),
                link=snippet.get("link", ""),
                snippet=snippet.get("snippet", ""),
            ))
        else:
            content = getattr(snippet, "snippet", "") or getattr(snippet, "content", str(snippet))
            web_for_record.append(snippet)
        parts.append(f"[外部网络资料{i}]\n{content}")
        sources.append(f"网络资料{i}")

    context = "\n\n---\n\n".join(parts) if parts else "暂无相关资料"

    print(f"【分析上下文】\n{context}\n{'='*50}")

    rag_log.debug("开始 LLM 法律分析生成", detail=f"context_len={len(context)}")

    final_prompt = LEGAL_ANALYSIS_PROMPT_Kim.format(context=context, query=query)
    response = llm.invoke(final_prompt)
    answer = response.content if hasattr(response, "content") else str(response)

    prompts_record = PromptsRecord(
        query=query,
        web_search_results=web_for_record,
        evluate_retrieved_documents=eval_docs_for_record,
        laws_results=laws_for_record,
    )

    elapsed = time.time() - t0

    tool_log.info(
        "← 工具返回: analyze_legal_issue",
        detail=f"context_len={len(context)} | sources={len(sources)}",
        result=f"answer_len={len(answer)} | elapsed={elapsed:.2f}s",
    )

    return {
        "final_answer": answer,
        "crag_context": context,
        "sources": sources,
        "prompts_record": prompts_record,
    }
