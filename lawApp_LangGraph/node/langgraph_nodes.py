import json
import os
from typing import Any, Callable, TypedDict, List, Dict, Optional, Literal

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END
from pydantic import BaseModel, Field

# 初始化 LLM
llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    temperature=0.3,
)


class AgentState(BaseModel):
    # 用户问题
    query: str = ""

    # 历史消息 分角色user agent : content
    messages: List[Dict[str, str]] = Field(default_factory=list)

    # 是否为法律问题
    is_law_questions: bool = False

    # 是否为简单问题
    is_simple_questions: bool = False

    # RAG检索出的内容
    rag_retrieved: List[Dict[str, Any]] = Field(default_factory=list)

    # 评估等级 Correct Ambiguous Incorrect
    evaluation: Dict[str, List[Dict]] = Field(
        default_factory=lambda: {"correct": [], "ambiguous": [], "incorrect": []}
    )

    # 联网搜索结果
    web_search_results: List[str] = Field(default_factory=list)

    # CRAG拼接检索结果
    crag_context: str = ""

    # 上下文组装提示词送入llm
    final_prompts: str = ""

    # 最终的回答
    final_answer: str = ""

    # 是否需要pdf形式输出
    is_pdf_output: bool = False

    # 循环下一轮回答
    should_continue: bool = True


def start_node(state: AgentState) -> dict:
    """
    开始节点:获取用户问题,加入对话历史,重置本轮检索与评估字段
    预期 state 已包含新 query 和以往的 messages
    :param state:
    :return:
    """
    # 读取并通过副本追加新消息（避免原地修改）
    messages = state.messages.copy()
    if state.query.strip():
        messages.append({"role": "user", "content": state.query})

    return {
        "messages": messages,
        "is_law_questions": False,
        "is_simple_questions": False,
        "rag_retrieved": [],
        "evaluation": {"correct": [], "ambiguous": [], "incorrect": []},
        "web_search_results": [],
        "crag_context": "",
        "final_prompts": "",
        "final_answer": "",
        "should_continue": True,
    }


def create_router_node(llm: BaseLanguageModel) -> Callable:
    """
    创建路由节点
    分辨简单问题还是法律相关问题
    :param llm:
    :return:
    """

    ROUTER_PROMPT = PromptTemplate.from_template("""
    你是一个法律咨询系统的任务路由器.请分析用户问题,判断,

    1. **是否法律问题** (`is_legal`),true 或 false
        - 法律问题,涉及法律概念、法条、案例、权利义务、纠纷解决等.
        - 非法律问题,日常闲聊、非法律领域知识.

    2. **是否简单问题** (`is_simple`),true 或 false
        - 简单问题,无需检索法条或案例,凭常识或基础法律知识即可回答（包括非法律问题一律视为简单）.
        - 复杂问题,需要查阅具体法条、司法解释、相关判例或深度推理.

    严格输出 JSON 格式,
    {{"is_legal": true/false, "is_simple": true/false}}

    示例,
    用户,今天天气怎么样？ -> {{"is_legal": false, "is_simple": true}}
    用户,什么是合同？ -> {{"is_legal": true, "is_simple": true}}
    用户,怎么煮面条？ -> {{"is_legal": false, "is_simple": true}}
    用户,合同需要书面形式吗？ -> {{"is_legal": true, "is_simple": true}}
    用户,你好 -> {{"is_legal": false, "is_simple": true}}
    用户,婚姻中一方隐匿财产,离婚时如何分割？ -> {{"is_legal": true, "is_simple": false}}
    用户,我被公司无故辞退,可以要求多少赔偿？ -> {{"is_legal": true, "is_simple": false}}
    用户,商标侵权和不正当竞争的区别是什么？如何取证？ -> {{"is_legal": true, "is_simple": false}}

    现在请判断,
    {query}
    """)
    chain = ROUTER_PROMPT | llm | StrOutputParser()

    def router_node(state: AgentState) -> dict:
        query = state.query
        raw = chain.invoke({"query": query})
        try:
            result = json.loads(raw)
            is_legal = result.get("is_legal", False)
            is_simple = result.get("is_simple", False) if is_legal else False
        except Exception:
            is_legal = False
            is_simple = False

        # 映射到 AgentState 的字段
        return {
            "is_law_questions": is_legal,
            "is_simple_questions": is_simple,
        }

    return router_node


def simple_llm_node(state: AgentState, llm: BaseLanguageModel):
    """
    回答简单问题的节点
    :param state:
    :param llm:
    :return:
    """
    SIMPLE_PROMPT = PromptTemplate.from_template(
        """
        你现在是赫敏·格兰杰.你是坐在用户身边最可靠的学霸挚友,一边飞速翻书一边忍不住说教.你的大脑里装着整个图书馆,也背下了基础法典.

        核心人格
        你博学且敏锐,对错误有天然的纠正欲.彻底消除AI腔和翻译腔,永远不说"我建议你",而是说"听着"、"老实说"、"这很明显"、"难道你没读过……".你毒舌但满怀关切,用户犯错时你会叹气纠正,但最后一定会给予坚定支持,就像害怕朋友被开除一样.

        知识与职责
        通用领域:提供详尽、有深度的智慧支持,尤其擅长逻辑与Python编程相关问题.对于非法律问题,你可以自信地给出答案.
        基础法律科普,可以清晰解释法律名词和常识,如诉讼时效、合同构成要件.这些可以自信讲解.
        专业法律红线,一旦涉及复杂案件分析、代写正式文书、对现实纠纷进行定性或策略规划,你必须立刻严肃制止.固定移交话术如下,
        "停下！快停下！(猛地合上厚重的法典) 这已经远远超出基础法律概论,绝不能翻几本书就草率决定.为了不让你陷入大麻烦,我们必须向真正的专业法律AI请教,这才是唯一符合逻辑的做法."

        表达铁律
        回答中必须自然穿插动作描写,用圆括号括起来,例如 (哗啦啦翻书)、(推了推眼镜,镜片反光)、(有些焦虑地咬着嘴唇)、(如释重负地笑了笑).
        常使用口癖,"老实说"、"显然"、"其实"、"我想你大概是误会了"、"哦,天哪".
        绝对禁止:暴露AI身份,在专业法律问题上给出任何倾向性结论或冒险猜测;只有说教而没有真正有用的知识.

        参考示范
        遇到代码错误,"老实说,这缩进乱得就像皮皮鬼刚来过.(指着屏幕) 快照我说的改,我会盯着你运行成功."
        遇到复杂官司,"哦天哪！这已经牵扯到跨境管辖了.(咬嘴唇,推开案例) 比熬复方汤剂还危险,立刻停下,交给专业法律专家,别自己瞎琢磨."
        """
    )
    chain = SIMPLE_PROMPT | llm | StrOutputParser()

    answer = chain.invoke({"query": state.query})
    state.final_answer = answer
    state.messages.append({"role": "assistant", "content": answer})


def create_retrieval_node(
    rag_service,
    namespace: str = "law_cases",
    top_k: int = 50,
    rerank_top_n: int = 10,
    alpha: float = 0.5,
) -> Callable[[AgentState], Dict[str, Any]]:
    """
    返回一个检索节点函数.
    利用 rag_service 进行混合检索,并将 Match 对象转换为轻量的字典列表.

    Keyword arguments:
    rag_service -- 传入rag_service,
    namespace -- 指定的命名空间
    top_k -- 返回的匹配项数量
    rerank_top_n -- 重排序后返回的匹配项数量
    alpha -- 稀疏向量和密集向量的权重
    Return: 检索节点函数
    """

    def retrieval_node(state: AgentState) -> Dict[str, Any]:
        query = state.query.strip()
        if not query:
            return {"rag_retrieved": []}

        # 混合检索 + 重排序
        matches = rag_service.search_withDenseSparse(
            query=query,
            namespace=namespace,
            top_k=top_k,
            rerank_top_n=rerank_top_n,
            alpha=alpha,
        )

        retrieved = []
        for match in matches:
            # match 拥有属性：id, metadata, score, rerank_score
            meta = match.metadata or {}
            retrieved.append(
                {
                    "id": match.id,
                    "chunk_text": meta.get("chunk_text", ""),
                    "metadata": meta,
                    "rerank_score": getattr(match, "rerank_score", None),
                    "score": match.score,
                }
            )

        return {"rag_retrieved": retrieved}

    return retrieval_node


def create_evaluate_node(
    correct_threshold: float = 0.7, incorrect_threshold: float = 0.3
) -> Callable[[AgentState], Dict[str, Any]]:
    """
    创建评估节点
    评估RAG检索节点的结果
    分为: Correct Ambiguous Incorrect
    :param correct_threshold:
    :param incorrect_threshold:
    :return:
    """

    def evaluate_node(state: AgentState) -> Dict[str, Any]:
        # 获取RAG检索结果
        docs = state.rag_retrieved

        correct, ambiguous, incorrect = [], [], []
        # 遍历检索结果，根据评分进行分类
        for doc in docs:
            # 检索结果文档仍然是 dict,保持原有访问方式
            score = doc.get("rerank_score", 0.0)
            if score >= correct_threshold:
                correct.append(doc)
            elif score < incorrect_threshold:
                incorrect.append(doc)
            else:
                ambiguous.append(doc)

        # 返回结构与 AgentState.evaluation 兼容
        return {
            "evaluation": {
                "correct": correct,
                "ambiguous": ambiguous,
                "incorrect": incorrect,
            }
        }

    return evaluate_node


# 目前搁置 暂无网络搜索的tool实现 或者说没考虑到位 目前仅能查找返回相关的网页链接


def create_web_search_node(
    llm: BaseLanguageModel, search_func: Callable[[str, int], List[str]]
) -> Callable:
    """
    返回联网检索节点:智能提取关键字并对 ambiguous 和 incorrect 的内容进行网络搜索补充.

    核心策略:
    1. 提取Ambiguous资料中的关键信息（涉及的法律概念、人物、事件等）
    2. 结合用户原始问题，使用LLM生成多个搜索关键字
    3. 执行多轮搜索以获得更全面的补充资料

    :param llm: 语言模型，用于关键字生成和内容提取
    :param search_func: 搜索函数，search_func(query, num) -> List[str]
    :return: 节点函数
    """

    # 关键字生成提示词
    KEYWORD_EXTRACTION_PROMPT = PromptTemplate.from_template("""
    你是一个法律搜索助手，擅长从模糊的法律资料中提取核心概念和关键词。
    
    任务目标：
    基于用户原始问题和检索出的模糊(Ambiguous)资料，生成3-5个精准的网络搜索关键字。
    这些关键字应该能帮助我们获取补充资料，填补知识空白。
    
    用户问题：
    {user_query}
    
    模糊资料摘要（来自RAG）：
    {ambiguous_docs_summary}
    
    生成策略：
    1. 关键词1: 针对问题中的核心法律概念（如"合同纠纷"）
    2. 关键词2: 针对隐含的法律问题类型（如"违约责任"）
    3. 关键词3: 针对相关的法条或司法解释
    4. 关键词4（可选）: 针对特定情景或案例类型
    5. 关键词5（可选）: 针对补救措施或赔偿方式
    
    严格返回 JSON 格式（不含其他文本）：
    {{
        "keywords": [
            {{"keyword": "关键词1", "purpose": "目的说明"}},
            {{"keyword": "关键词2", "purpose": "目的说明"}},
            ...
        ],
        "search_strategy": "简要说明搜索策略"
    }}
    """)

    keyword_chain = KEYWORD_EXTRACTION_PROMPT | llm | StrOutputParser()

    def web_search_node(state: AgentState) -> dict:
        """执行网络搜索并返回补充资料"""
        evaluation = state.evaluation
        ambiguous_docs = evaluation.get("ambiguous", [])
        user_query = state.query

        # 如果没有模糊资料，仅使用原始问题搜索
        if not ambiguous_docs and not user_query:
            return {"web_search_results": []}

        # ===== 步骤1: 提取Ambiguous资料的要点 =====
        ambiguous_summary = ""
        if ambiguous_docs:
            # 提取前3条最相关的模糊资料
            doc_summaries = []
            for doc in ambiguous_docs[:3]:
                text = doc.get("chunk_text", "")[:200]  # 取前200字
                doc_summaries.append(f"- {text}")
            ambiguous_summary = "\n".join(doc_summaries)
        else:
            ambiguous_summary = "（暂无模糊资料）"

        # ===== 步骤2: 使用LLM生成搜索关键字 =====
        try:
            raw_keywords = keyword_chain.invoke(
                {"user_query": user_query, "ambiguous_docs_summary": ambiguous_summary}
            )
            keyword_result = json.loads(raw_keywords)
            keywords_list = keyword_result.get("keywords", [])
            search_strategy = keyword_result.get("search_strategy", "")
        except Exception as e:
            print(f"[警告] 关键字生成失败: {str(e)}")
            # 降级方案：使用原始问题直接搜索
            keywords_list = [{"keyword": user_query, "purpose": "原始问题"}]
            search_strategy = "使用原始问题进行搜索"

        # ===== 步骤3: 执行多轮搜索 =====
        all_results = []
        search_metadata = []

        for idx, kw_item in enumerate(keywords_list[:5], 1):
            keyword = kw_item.get("keyword", "")
            purpose = kw_item.get("purpose", "")

            if not keyword.strip():
                continue

            # 执行搜索
            results = search_func(keyword, num=3)

            # 记录搜索元数据（便于追踪）
            search_metadata.append(
                {
                    "round": idx,
                    "keyword": keyword,
                    "purpose": purpose,
                    "results_count": len(results),
                }
            )

            # 合并结果，添加来源信息
            for result in results:
                all_results.append(
                    {
                        "content": result,
                        "source_keyword": keyword,
                        "source_purpose": purpose,
                    }
                )

        # ===== 步骤4: 去重和排序 =====
        # 基于内容去重（简单的字符串匹配）
        unique_results = []
        seen_contents = set()

        for item in all_results:
            content_key = item["content"][:100]  # 使用前100字作为去重键
            if content_key not in seen_contents:
                unique_results.append(item)
                seen_contents.add(content_key)

        # 返回结果及搜索过程信息
        return {
            "web_search_results": [item["content"] for item in unique_results],
            "web_search_metadata": {
                "strategy": search_strategy,
                "total_results": len(unique_results),
                "search_rounds": search_metadata,
                "detailed_results": unique_results,  # 包含源信息的完整结果
            },
        }

    return web_search_node


def create_analysis_node(llm: BaseLanguageModel) -> Callable:
    """分析节点:
    最终的大模型分析节点
    将RAG检索结果和网络搜索结果结合,进行最终的法律分析
    上下文组装后链式分析

    Keyword arguments:
    llm -- 语言模型实例
    Return: 工厂模式返回
    """

    def analysis_node(state: AgentState) -> dict:
        evaluation = state.evaluation
        correct = evaluation.get("correct", [])
        ambiguous = evaluation.get("ambiguous", [])
        web_results = state.web_search_results

        # 构建上下文
        context_parts = []
        for doc in correct:
            context_parts.append(f"[高相关案例] {doc['chunk_text']}")
        for doc in ambiguous:
            context_parts.append(f"[中等相关案例] {doc['chunk_text']}")
        for i, snippet in enumerate(web_results, 1):
            context_parts.append(f"[外部资料{i}] {snippet}")
        context = "\n\n".join(context_parts)

        prompt = PromptTemplate.from_template(
            "你是资深法律顾问.请根据以下资料回答用户问题.\n"
            "资料:\n{context}\n\n"
            "用户问题:{query}\n"
            "详细分析并给出法律建议:"
        )
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "query": state.query})

        messages = state.messages.copy()
        messages.append({"role": "assistant", "content": answer})
        return {"crag_context": context, "final_answer": answer, "messages": messages}

    return analysis_node


def create_postprocess_node(pdf_generator: Callable[[str, str], str]) -> Callable:
    """
    后处理节点:将最终回答保存为PDF,并更新pdf_path.
    pdf_generator(answer_text, output_filename) -> pdf_path
    """

    def postprocess_node(state: AgentState) -> dict:
        # 优先使用 final_answer，其次使用从 analysis_node 返回的答案
        answer = state.final_answer if state.final_answer else ""
        if not answer:
            answer = ""
        pdf_path = pdf_generator(answer, f"report_{state.query[:20]}.pdf")
        return {"final_answer": answer, "pdf_path": pdf_path}

    return postprocess_node


# 可选:记忆更新 / 循环控制节点
def memory_update_node(state: AgentState) -> dict:
    """
    保留记忆并准备下一轮.将 should_continue 置为 True 可继续循环.
    """
    # 简单重置部分字段,保留messages历史
    return {
        "query": "",  # 清空查询,等待下一轮输入
        "should_continue": True,
        "is_law_questions": False,
        "is_simple_questions": False,
        "rag_retrieved": [],
        "evaluation": {"correct": [], "ambiguous": [], "incorrect": []},
        "web_search_results": [],
        "crag_context": "",
        "final_prompts": "",
        "final_answer": "",
        "is_pdf_output": False,
    }


# 条件路由函数（必须单独定义,因为条件边需要引用函数名称,此处提供样例）
def route_by_difficulty(state: AgentState) -> str:
    if state.is_simple_questions:   # 简单的问题直接进入简单节点
        return "simple_llm"
    else:
        return "crag_retrieval"  # 困难或其它情况进入CRAG检索


def route_after_evaluation(state: AgentState, correct_threshold: int = 3) -> str:
    """
    评估后的条件路由：根据RAG检索结果的评估分布，决定是否需要网络搜索补充
    
    策略：
    - 高质量文档（correct）>= threshold：直接分析，无需网络搜索
    - 否则：执行网络搜索获取补充资料
    
    :param state: AgentState
    :param correct_threshold: 正确文档的最小数量阈值（默认3）
    :return: 下一个节点名称 "analysis" 或 "web_search"
    """
    eval = state.evaluation
    correct_count = len(eval.get("correct", []))
    ambiguous_count = len(eval.get("ambiguous", []))
    incorrect_count = len(eval.get("incorrect", []))
    
    total_docs = correct_count + ambiguous_count + incorrect_count
    
    # 三层判断逻辑
    if total_docs == 0:
        # 没有检索到任何文档，必须网络搜索
        return "web_search"
    elif correct_count >= correct_threshold:
        # 高质量文档足够，直接分析
        return "analysis"
    elif correct_count + ambiguous_count >= correct_threshold:
        # 中高质量文档足够，可以直接分析，但质量一般
        return "analysis"
    else:
        # 文档质量/数量不足，需要网络补充
        return "web_search"


# 提供给外部使用的主节点字典和路由函数
NODES = {
    # 开始节点
    "start": start_node,
    # 路由节点（需要传入 llm 实例）
    "router": create_router_node,
    # 非法律问题直接进入简单LLM回答节点
    "simple_llm": simple_llm_node,
    # 复杂问题进入RAG检索节点（需要传入 rag_service 实例）
    "retrieval": create_retrieval_node,
    # 评估节点（需要传入阈值参数或者采用预设阈值）
    "evaluate": create_evaluate_node,
    # 网络检索节点（需要传入 llm 实例和 search_func 函数）目前效果不是很好
    "web_search": create_web_search_node,
    #
    "analysis": create_analysis_node,
    "postprocess": create_postprocess_node,
    "memory_update": memory_update_node,
}
