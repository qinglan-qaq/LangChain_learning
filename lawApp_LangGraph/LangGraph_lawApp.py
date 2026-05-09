"""
Plan & Execute Agent — 法律咨询智能体

双 LLM 架构:
    llm_planner  (DeepSeek Pro)   → 制定计划、重规划、给出思考过程
    llm_executor (DeepSeek Flash)  → 按计划执行、调用工具、无需深度思考

Graph 流程:
    START → planner → executor (loop) → replan_check → finalize → END
                ↑                        ↓
                └── replanner ←──────────┘ (质量不足 / 用户要求重规划)

核心组件:
    1. The Planner    — Pro LLM 分析问题 → JSON 计划 + 思考链
    2. The Executor   — Flash LLM 逐步调用工具,自动映射参数
    3. The Replanner  — Pro LLM 检查执行结果,不满则重新生成计划
    4. Conditional Edges — 质量门控 / 循环控制
"""
import json
import os
from typing import Any, Dict
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from lawApp_LangGraph.state import AgentState, EvaluationResult, PlanStep, RetrievedDocument, ToolCallRecord
from lawApp_LangGraph.tools import ALL_TOOLS

load_dotenv()


# 双 LLM 架构

_llm_kwargs = dict(
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

llm_planner = ChatOpenAI(
    model=os.getenv("DEEPSEEK_PRO_MODEL", "deepseek-chat"),
    temperature=0.4,
    max_tokens=4096,
    **_llm_kwargs,
)

llm_executor = ChatOpenAI(
    model=os.getenv("DEEPSEEK_FLASH_MODEL", "deepseek-chat"),
    temperature=0.25,
    max_tokens=2048,
    **_llm_kwargs,
)


# Executor 绑定全部工具,支持 Function Calling
llm_executor_with_tools = llm_executor.bind_tools(ALL_TOOLS)

# 工具名 → 工具对象
TOOL_BY_NAME: Dict[str, Any] = {t.name: t for t in ALL_TOOLS}


# 工具输出 → AgentState 字段映射


def merge_tool_output(state: AgentState, tool_name: str, output: Any) -> Dict[str, Any]:
    """将工具返回的 dict 合并到 AgentState"""
    updates: Dict[str, Any] = {}

    if tool_name == "retrieve_legal_knowledge":
        results = output.get("results", []) if isinstance(output, dict) else []
        docs = [RetrievedDocument(**r) if isinstance(r, dict) else r for r in results]
        updates["rag_documents"] = docs

    elif tool_name == "evaluate_case_relevance":
        if isinstance(output, dict):
            def _doc(d):
                return RetrievedDocument(**d) if isinstance(d, dict) else d

            updates["evaluation"] = EvaluationResult(
                total=output.get("total", 0),
                correct_count=output.get("correct_count", 0),
                ambiguous_count=output.get("ambiguous_count", 0),
                incorrect_count=output.get("incorrect_count", 0),
                quality_verdict=output.get("quality_verdict", ""),
                correct=[_doc(d) for d in output.get("correct", [])],
                ambiguous=[_doc(d) for d in output.get("ambiguous", [])],
                incorrect=[_doc(d) for d in output.get("incorrect", [])],
            )

    elif tool_name == "get_google_search":
        results = output.get("results", []) if isinstance(output, dict) else []
        snippets = [
            f"[{r.get('title','')}] {r.get('snippet','')} ({r.get('link','')})"
            for r in results
        ]
        updates["web_search_results"] = list(state.web_search_results) + snippets

    elif tool_name == "fetch_webpage_text":
        if isinstance(output, dict) and output.get("text"):
            text = f"[网页正文 | {output.get('url','')}]\n{output['text']}"
            updates["web_search_results"] = list(state.web_search_results) + [text]

    elif tool_name == "analyze_legal_issue":
        if isinstance(output, dict):
            updates["final_answer"] = output.get("final_answer", "")
            updates["crag_context"] = output.get("crag_context", "")

    elif tool_name == "markdown_to_pdf":
        if isinstance(output, str) and "PDF" in output:
            import re
            m = re.search(r"文件路径[::]\s*(.+)", output)
            if m:
                updates["pdf_path"] = m.group(1).strip()
                updates["is_pdf_output"] = True

    return updates



# 工具降级:不依赖 Flash LLM,直接参数映射

def _invoke_tool_direct(tool_name: str, state: AgentState) -> dict:
    """直接参数映射 + 调用工具(Flash LLM 调用失败时的降级路径)"""
    tool = TOOL_BY_NAME[tool_name]
    q = state.query

    mapping = {
        "retrieve_legal_knowledge": {"query": q, "top_k": 50, "rerank_top_n": 10},
        "evaluate_case_relevance": {
            "documents": [d.dict() for d in state.rag_documents] if state.rag_documents else []
        },
        "get_google_search": {"query": q},
        "analyze_legal_issue": {
            "query": q,
            "correct_cases": [d.dict() for d in state.evaluation.correct],
            "ambiguous_cases": [d.dict() for d in state.evaluation.ambiguous],
            "web_results": list(state.web_search_results),
        },
        "markdown_to_pdf": {
            "markdown_text": state.final_answer or "暂无内容",
            "filename": f"legal_report_{q[:20]}.pdf",
        },
    }

    raw = tool.invoke(mapping.get(tool_name, {"query": q}))
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"result": raw}
    return raw



# Node 1: The Planner — Pro LLM 制定计划 + 思考链

PLANNER_SYSTEM = """
    你是法律AI系统的任务规划师.分析用户问题,制定可执行的步骤计划.

    ## 可用工具
    {available_tools}

    ## 输出 JSON 格式
    {{
        "reasoning": ["思考过程1", "思考过程2", ...],
        "plan": [
            {{"step_id": 1, "description": "步骤描述", "tool_name": "工具名"}},
            ...
        ]
    }}

    ## 计划原则
    - 法律问题: retrieve_legal_knowledge → evaluate_case_relevance → analyze_legal_issue
    - 如评估结果为"不足": 插入 get_google_search / fetch_webpage_text 再分析
    - 简单闲聊: plan 为空数组 []
    - 用户要求 PDF 输出时才用 markdown_to_pdf
    - tool_name 必须是上述列表中的名称,不需要工具则填写 null

    ## 用户问题
    {query}
"""


def planner_node(state: AgentState) -> dict:
    """Pro LLM: 分析问题 → JSON 计划 + 思考链"""
    query = state.query.strip()
    if not query:
        return {"plan": [], "reasoning": ["无输入"], "final_answer": "请提供问题."}
    # 工具描述列表
    tools_desc = "\n".join(f"- {t.name}: {t.description[:120]}" for t in ALL_TOOLS)
    
    chain = PromptTemplate.from_template(PLANNER_SYSTEM) | llm_planner | StrOutputParser()
    
    raw = chain.invoke({"query": query, "available_tools": tools_desc})

    # 解析 llm返回结果,提取计划和思考链;解析失败则返回默认计划
    try:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]
        result = json.loads(raw)
        
    except json.JSONDecodeError:
        return {
            "reasoning": ["Planner 输出解析失败,使用默认法律检索计划"],
            "plan": [
                PlanStep(step_id=1, description="检索相关法律案例", tool_name="retrieve_legal_knowledge"),
                PlanStep(step_id=2, description="评估检索质量", tool_name="evaluate_case_relevance"),
                PlanStep(step_id=3, description="综合信息生成法律分析", tool_name="analyze_legal_issue"),
            ],
        }
    # 提取计划和思考链
    reasoning = result.get("reasoning", [])
    plan_dicts = result.get("plan", [])

    plan = []
    for p in plan_dicts:
        tn = p.get("tool_name", "")
        if tn and tn not in TOOL_BY_NAME:
            tn = ""
        plan.append(PlanStep(
            step_id=p.get("step_id", len(plan) + 1),
            description=p.get("description", ""),
            tool_name=tn,
        ))

    return {
        "plan": plan,
        "reasoning": reasoning,
        "current_step_index": 0,
        "replan_needed": False,
        "replan_reason": None,
    }



# Node 2: The Executor — Flash LLM 执行单步骤,调用工具,自动映射参数;失败则降级直接调用工具

EXECUTOR_PROMPT = """\
你是执行器,只做一件事:调用指定的工具.

当前步骤: {step_description}
指定工具: {tool_name}
用户问题: {user_query}

上下文数据:
- 已检索案例: {rag_summary}
- 案例评估: {eval_summary}
- 网络搜索: {web_summary}

规则:
1. 只调用 {tool_name},不要调用其他工具
2. 从上下文和用户问题中提取参数
3. 不要做推理,只需正确调用工具
"""


def executor_node(state: AgentState) -> dict:
    """Flash LLM: 调用指定工具,更新状态;失败则自动降级为直接参数映射"""
    idx = state.current_step_index
    plan = state.plan

    if idx >= len(plan):
        return {}

    step = plan[idx]
    step.status = "doing"

    # 无工具步骤 → 跳过
    if not step.tool_name:
        step.status = "done"
        return {"plan": plan, "current_step_index": idx + 1}

    tool_output = None
    error_msg = None

    # ── 路径 A: Flash LLM 辅助调用 ──
    try:
        # 状态摘要
        rag_summary = "暂无"
        if state.rag_documents:
            rag_summary = " | ".join(
                f"[{d.rerank_score:.2f}] {d.chunk_text[:100]}..."
                for d in state.rag_documents[:3]
            )

        eval_summary = "未评估"
        if state.evaluation and state.evaluation.total > 0:
            ev = state.evaluation
            eval_summary = (
                f"共{ev.total}条,高质量{ev.correct_count},中等{ev.ambiguous_count},"
                f"低质量{ev.incorrect_count},结论: {ev.quality_verdict}"
            )

        web_summary = "暂无"
        if state.web_search_results:
            web_summary = state.web_search_results[-3] if state.web_search_results else "暂无"

        prompt = EXECUTOR_PROMPT.format(
            step_description=step.description,
            tool_name=step.tool_name,
            user_query=state.query,
            rag_summary=rag_summary,
            eval_summary=eval_summary,
            web_summary=web_summary,
        )

        response = llm_executor_with_tools.invoke([SystemMessage(content=prompt)])

        if isinstance(response, AIMessage) and response.tool_calls:
            for tc in response.tool_calls:
                called = TOOL_BY_NAME.get(tc["name"])
                if called:
                    raw = called.invoke(tc["args"])
                    tool_output = json.loads(raw) if isinstance(raw, str) else raw
                    break
        else:
            raise RuntimeError("Flash LLM 未发起工具调用")
    except Exception as e:
        # ── 路径 B: 降级直接参数映射 ──
        try:
            tool_output = _invoke_tool_direct(step.tool_name, state)
        except Exception as e2:
            error_msg = f"{e} | 降级: {e2}"

    # 记录调用痕迹
    state.tool_calls.append(ToolCallRecord(
        step_id=step.step_id,
        tool_name=step.tool_name,
        input={"query": state.query},
        output=tool_output,
    ))

    # 合并结果到状态
    state_updates: Dict[str, Any] = {
        "plan": plan,
        "tool_calls": state.tool_calls,
    }

    if tool_output is not None:
        state_updates.update(merge_tool_output(state, step.tool_name, tool_output))

    step.status = "failed" if error_msg else "done"
    state_updates["current_step_index"] = idx + 1
    if error_msg:
        state_updates["error"] = error_msg

    return state_updates



# Node 3: Replan Check — 质量门控


def replan_check_node(state: AgentState) -> dict:
    """
    检查执行结果:
    - 用户要求重规划 (replan_needed)
    - 评估结果为"不足"但无联网搜索步骤
    - 执行出错
    """
    needs = False
    reasons = []

    if state.replan_needed:
        needs = True
        reasons.append(state.replan_reason or "用户触发重规划")

    if state.evaluation and state.evaluation.quality_verdict == "不足,建议进行网络搜索补充":
        executed = {tc.tool_name for tc in state.tool_calls}
        if "get_google_search" not in executed:
            needs = True
            reasons.append("检索质量不足,需补联网搜索")

    if state.error:
        needs = True
        reasons.append(f"执行异常: {state.error}")

    return {
        "replan_needed": needs,
        "replan_reason": "; ".join(reasons) if reasons else None,
    }



# Node 4: The Replanner — Pro LLM 重新规划


REPLANNER_SYSTEM = """\
你是任务规划师.基于已执行的步骤和当前结果,生成**补充步骤**.

## 已执行步骤
{executed_steps}

## 当前状态
- 案例数量: {doc_count}
- 评估结论: {quality}
- 网络搜索: {web_count} 条
- 错误: {error}

## 重规划原因
{replan_reason}

## 可用工具
{available_tools}

## 用户问题
{user_query}

## 输出 JSON
{{
    "reasoning": ["修正思路1", "修正思路2"],
    "additional_steps": [
        {{"step_id": {next_id}, "description": "...", "tool_name": "..."}}
    ]
}}
只输出需要**新增**的步骤,不要重复已完成的步骤.
"""


def replanner_node(state: AgentState) -> dict:
    """Pro LLM: 检查当前结果 → 生成补充计划 → 返回 Executor"""
    executed = "\n".join(
        f"[{'done' if s.status == 'done' else 'failed'}] "
        f"步骤{s.step_id}: {s.description} → {s.tool_name}"
        for s in state.plan
    )

    tools_desc = "\n".join(f"- {t.name}: {t.description[:120]}" for t in ALL_TOOLS)

    prompt = REPLANNER_SYSTEM.format(
        executed_steps=executed,
        doc_count=len(state.rag_documents),
        quality=state.evaluation.quality_verdict if state.evaluation else "未评估",
        web_count=len(state.web_search_results),
        error=state.error or "无",
        replan_reason=state.replan_reason or "质量不足",
        available_tools=tools_desc,
        user_query=state.query,
        next_id=len(state.plan) + 1,
    )

    chain = PromptTemplate.from_template(prompt) | llm_planner | StrOutputParser()
    raw = chain.invoke({})

    try:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]
        result = json.loads(raw)
    except json.JSONDecodeError:
        default = [
            PlanStep(step_id=len(state.plan) + 1, description="联网搜索补充", tool_name="get_google_search"),
            PlanStep(step_id=len(state.plan) + 2, description="综合信息生成分析", tool_name="analyze_legal_issue"),
        ]
        return {
            "plan": state.plan + default,
            "reasoning": state.reasoning + ["[Replan 降级] 插入默认补充步骤"],
            "replan_needed": False,
            "replan_reason": None,
            "error": None,
        }

    additional = result.get("additional_steps", [])
    new_reasoning = result.get("reasoning", [])

    base = len(state.plan)
    new_steps = []
    for p in additional:
        tn = p.get("tool_name", "")
        if tn and tn not in TOOL_BY_NAME:
            tn = ""
        new_steps.append(PlanStep(
            step_id=base + len(new_steps) + 1,
            description=p.get("description", ""),
            tool_name=tn,
        ))

    return {
        "plan": state.plan + new_steps,
        "reasoning": state.reasoning + [f"[Replan] {state.replan_reason}"] + new_reasoning,
        "replan_needed": False,
        "replan_reason": None,
        "error": None,
    }



# Node 5: Finalize — 组装最终回答

def finalize_node(state: AgentState) -> dict:
    """如果已有 final_answer 则直接使用；否则用已检索案例生成简要回答"""
    if state.final_answer:
        return {}

    if state.rag_documents:
        docs = "\n".join(f"- {d.chunk_text[:300]}" for d in state.rag_documents[:3])
        prompt = PromptTemplate.from_template(
            "基于以下案例,简要回答用户问题.\n案例:\n{docs}\n\n问题: {query}\n\n法律建议:"
        )
        chain = prompt | llm_executor | StrOutputParser()
        answer = chain.invoke({"docs": docs, "query": state.query})
        return {"final_answer": answer}

    return {"final_answer": "您好！请问有什么法律问题需要咨询？"}



# 条件路由函数 (Conditional Edges)


def route_after_planner(state: AgentState) -> str:
    """有步骤 → executor | 无步骤 → finalize"""
    return "executor" if state.plan else "finalize"


def route_after_executor(state: AgentState) -> str:
    """还有步骤 → 继续 executor | 全部完成 → replan_check"""
    return "executor" if state.current_step_index < len(state.plan) else "replan_check"


def route_after_replan_check(state: AgentState) -> str:
    """需重规划 → replanner | 质量通过 → finalize"""
    return "replanner" if state.replan_needed else "finalize"


# 构建 Graph

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("replan_check", replan_check_node)
    builder.add_node("replanner", replanner_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "planner")

    builder.add_conditional_edges(
        "planner", route_after_planner,
        {"executor": "executor", "finalize": "finalize"},
    )

    builder.add_conditional_edges(
        "executor", route_after_executor,
        {"executor": "executor", "replan_check": "replan_check"},
    )

    builder.add_conditional_edges(
        "replan_check", route_after_replan_check,
        {"replanner": "replanner", "finalize": "finalize"},
    )

    builder.add_edge("replanner", "executor")
    builder.add_edge("finalize", END)

    return builder.compile()


graph = build_graph()


