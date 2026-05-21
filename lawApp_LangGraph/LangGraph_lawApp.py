"""
Plan & Execute Agent — 法律咨询智能体

双 LLM 架构:
    llm_planner  (DeepSeek Pro)   → 制定计划、重规划、给出思考过程
    llm_executor (DeepSeek Flash)  → 按计划执行、调用工具、无需深度思考

Graph 流程:
    START → planner → executor (loop) → replan_check → finalize → END
                ↑                        ↓
                └ replanner ←┘ (质量不足 / 用户要求重规划)

核心组件:
    1. The Planner    — Pro LLM 分析问题 → JSON 计划 + 思考链
    2. The Executor   — Flash LLM 逐步调用工具,自动映射参数
    3. The Replanner  — Pro LLM 检查执行结果,不满则重新生成计划
    4. Conditional Edges — 质量门控 / 循环控制
"""

import json
import os
import time
from typing import Any, Dict
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from lawApp_LangGraph.state import AgentState, PlanStep, ToolCallRecord
from lawApp_LangGraph.tools import ALL_TOOLS
from lawApp_LangGraph.FastAPI.logging import debug, flow, tool as tool_log
from langsmith import traceable

load_dotenv()


# 双 LLM 架构

# LLM 配置 —— 使用环境变量配置 DeepSeek API Key 和模型名称
_llm_kwargs = dict(
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

# Planner规划节点的llm — 生成计划和思考链,需要更强的推理能力
llm_planner = ChatOpenAI(
    model=os.getenv("DEEPSEEK_PRO_MODEL"),
    temperature=0.4,
    max_tokens=4096,
    **_llm_kwargs,
)

# Executor执行节点的llm — 只需正确调用工具,使用更轻量的模型以降低成本和延迟
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


#  工具 → AgentState 字段合并
# 工具返回 dict 中与 AgentState 同名的 key 将被自动合并
# web_search_results 特殊处理:追加而非覆盖
_STATE_KEYS = {
    "rag_documents",
    "evaluation",
    "final_answer",
    "crag_context",
    "web_search_results",
    "pdf_path",
    "is_pdf_output",
    "memory_results",
    "memory_update",
}

#  Flash LLM 降级:直接参数映射
_TOOL_FALLBACK_ARGS = {
    "retrieve_legal_knowledge": lambda s: {
        "query": s.query,
        "top_k": 50,
        "rerank_top_n": 10,
    },
    "evaluate_case_relevance": lambda s: {
        "documents": [d.model_dump() for d in s.rag_documents]
        if s.rag_documents
        else []
    },
    "get_google_search": lambda s: {"query": s.query},
    "analyze_legal_issue": lambda s: {
        "query": s.query,
        "correct_cases": [
            d.model_dump() for d in (s.evaluation.correct if s.evaluation else [])
        ],
        "ambiguous_cases": [
            d.model_dump() for d in (s.evaluation.ambiguous if s.evaluation else [])
        ],
        "web_results": list(s.web_search_results),
    },
    "markdown_to_pdf": lambda s: {
        "markdown_text": s.final_answer or "暂无内容",
        "filename": f"legal_report_{s.query[:20]}.pdf",
    },
}


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
    - 如用户提及之前讨论过的话题: 先用 search_memory 搜索历史记忆获取上下文
    - 如用户表达了个人偏好/情况: 在生成最终回答后用 save_to_memory 保存 (memory_type='user_fact')
    - 简单闲聊: plan 为空数组 []
    - 用户要求 PDF 输出时才用 markdown_to_pdf
    - tool_name 必须是上述列表中的名称,不需要工具则填写 null

    ## 用户问题
    {query}
"""


@traceable(run_type="chain", name="Planner_计划节点")
def planner_node(state: AgentState) -> dict:
    """Pro LLM: 分析问题 → JSON 计划 + 思考链"""
    t0 = time.time()
    query = state.query.strip()
    debug.debug("→ 进入 Planner 节点", detail=f"query={query[:80]}")

    if not query:
        debug.info("← Planner 退出", detail="空输入", result="返回默认提示")
        return {"plan": [], "reasoning": ["无输入"], "final_answer": "抱一丝,你能再说一遍吗?"}

    tools_desc = "\n".join(f"- {t.name}: {t.description[:120]}" for t in ALL_TOOLS)

    # 链式执行生成计划与思考链
    chain = (
        PromptTemplate.from_template(PLANNER_SYSTEM) | llm_planner | StrOutputParser()
    )
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
        debug.warning("Planner JSON 解析失败", detail="使用默认法律检索计划")
        return {
            "reasoning": ["Planner 输出解析失败,使用默认法律检索计划"],
            "plan": [
                PlanStep(
                    step_id=1,
                    description="检索相关法律案例",
                    tool_name="retrieve_legal_knowledge",
                ),
                PlanStep(
                    step_id=2,
                    description="评估检索质量",
                    tool_name="evaluate_case_relevance",
                ),
                PlanStep(
                    step_id=3,
                    description="综合信息生成法律分析",
                    tool_name="analyze_legal_issue",
                ),
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
        plan.append(
            PlanStep(
                step_id=p.get("step_id", len(plan) + 1),
                description=p.get("description", ""),
                tool_name=tn,
            )
        )

    elapsed = time.time() - t0
    step_names = [f"{s.step_id}.{s.tool_name or '闲聊'}" for s in plan]
    debug.info(
        "← Planner 完成",
        detail=f"reasoning={len(reasoning)}条, plan={len(plan)}步",
        result=f"elapsed={elapsed:.2f}s | 步骤: {' → '.join(step_names) if step_names else '无(直接回答)'}",
    )
    return {
        "plan": plan,
        "reasoning": reasoning,
        "current_step_index": 0,
        "replan_needed": False,
        "replan_reason": None,
    }


# Node 2: The Executor — Flash LLM 执行单步骤,调用工具,自动映射参数;失败则降级直接调用工具

EXECUTOR_PROMPT = """
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


@traceable(run_type="chain", name="Executor_执行节点")
def executor_node(state: AgentState) -> dict:
    """Flash LLM: 调用指定工具,更新状态;失败则自动降级为直接参数映射"""
    t0 = time.time()
    # 当前步骤数
    idx = state.current_step_index
    # 规划节点列出的计划
    plan = state.plan

    if idx >= len(plan):
        debug.debug("→ Executor 跳过", detail=f"步骤索引{idx}超出计划长度{len(plan)}")
        return {}

    step = plan[idx]
    total_steps = len(plan)
    step.status = "doing"

    debug.debug(
        f"→ 进入 Executor 节点 [{idx + 1}/{total_steps}]",
        detail=f"tool_name={step.tool_name or '无'} | desc={step.description[:60]}",
    )

    # 无工具步骤 → 跳过
    if not step.tool_name:
        step.status = "done"
        debug.info(
            f"← Executor 完成 [{idx + 1}/{total_steps}]",
            detail="无工具步骤,跳过",
            result=f"elapsed={time.time() - t0:.2f}s",
        )
        return {"plan": plan, "current_step_index": idx + 1}

    tool_output = None
    error_msg = None

    #  路径 A: Flash LLM 辅助调用
    try:
        rag_summary = "暂无"
        if state.rag_documents:
            rag_summary = " | ".join(
                f"[{d.rerank_score:.2f}] {d.chunk_text[:100]}..."
                for d in state.rag_documents[:3]
            )
        # 评估结果摘要
        eval_summary = "未评估"
        if state.evaluation and state.evaluation.total > 0:
            ev = state.evaluation
            eval_summary = (
                f"共{ev.total}条,高质量{ev.correct_count},中等{ev.ambiguous_count},"
                f"低质量{ev.incorrect_count},结论: {ev.quality_verdict}"
            )
        # 网络搜索摘要
        web_summary = "暂无"
        if state.web_search_results:
            web_summary = (
                state.web_search_results[-3] if state.web_search_results else "暂无"
            )

        prompt = EXECUTOR_PROMPT.format(
            step_description=step.description,
            tool_name=step.tool_name,
            user_query=state.query,
            rag_summary=rag_summary,
            eval_summary=eval_summary,
            web_summary=web_summary,
        )

        debug.debug("Executor 调用 Flash LLM", detail=f"tool={step.tool_name}")
        
        """
        这里是当前Executor节点的核心操作逻辑:
        在此之前,Planner节点已经生成了一个包含步骤描述和工具名称的计划
        Executor节点根据当前步骤的工具名称构造提示词.
        调用 llm_executor_with_tools.invoke() 
        让 Flash LLM 根据提示词分析当前步骤和上下文,自动提取参数并调用指定工具
        """
        response = llm_executor_with_tools.invoke([SystemMessage(content=prompt)])

        if isinstance(response, AIMessage) and response.tool_calls:
            for tc in response.tool_calls:
                called = TOOL_BY_NAME.get(tc["name"])
                if called:
                    debug.debug(
                        "LLM 发起工具调用",
                        detail=f"tool={tc['name']} | args={str(tc.get('args', {}))[:200]}",
                    )
                    tool_output = called.invoke(tc["args"])
                    break
        else:
            raise RuntimeError("Flash LLM 未发起工具调用")
    except Exception as e:
        #  路径 B: 降级直接参数映射
        debug.warning(
            "Flash LLM 调用失败,降级为参数映射",
            detail=f"tool={step.tool_name} | error={str(e)[:100]}",
        )
        try:
            args_fn = _TOOL_FALLBACK_ARGS.get(
                step.tool_name, lambda s: {"query": s.query}
            )
            tool_output = TOOL_BY_NAME[step.tool_name].invoke(args_fn(state))
            debug.info("降级参数映射成功", detail=f"tool={step.tool_name}")
        except Exception as e2:
            error_msg = f"{e} | 降级: {e2}"

    # 记录调用痕迹
    state.tool_calls.append(
        ToolCallRecord(
            step_id=step.step_id,
            tool_name=step.tool_name,
            input={"query": state.query},
            output=tool_output,
        )
    )

    # 合并工具输出到 AgentState
    state_updates: Dict[str, Any] = {
        "plan": plan,
        "tool_calls": state.tool_calls,
    }

    tool_result_summary = ""
    if isinstance(tool_output, dict):
        for k, v in tool_output.items():
            if k in _STATE_KEYS:
                if k == "web_search_results":
                    state_updates[k] = list(state.web_search_results) + v
                    tool_result_summary = f"web_results={len(v)}条"
                elif k == "rag_documents":
                    state_updates[k] = v
                    tool_result_summary = f"rag_docs={len(v)}条"
                elif k == "final_answer":
                    state_updates[k] = v
                    tool_result_summary = f"answer_len={len(v)}"
                elif k == "evaluation":
                    state_updates[k] = v
                    verdict = v.get("quality_verdict", "") if isinstance(v, dict) else getattr(v, "quality_verdict", "")
                    tool_result_summary = f"verdict={verdict}"
                else:
                    state_updates[k] = v

    step.status = "failed" if error_msg else "done"
    state_updates["current_step_index"] = idx + 1
    if error_msg:
        state_updates["error"] = error_msg

    elapsed = time.time() - t0
    status = "失败" if error_msg else "完成"
    debug.info(
        f"← Executor {status} [{idx + 1}/{total_steps}]",
        detail=f"tool={step.tool_name}",
        result=f"{tool_result_summary} | elapsed={elapsed:.2f}s" if tool_result_summary else f"elapsed={elapsed:.2f}s",
    )
    return state_updates


# Node 3: Replan Check — Flash LLM 质量门控

REPLAN_CHECK_PROMPT = """
你是法律AI系统的质量审核员。检查已执行步骤的结果，判断当前信息是否足以生成高质量的法律回答。

## 用户原始问题
{user_query}

## 已执行步骤及结果
{executed_summary}

## 当前数据状态
- 检索到的案例数量: {doc_count}
- 案例质量评估: {quality_verdict}
- 网络搜索补充: {web_count} 条
- 执行错误: {error_info}

## 判断标准
1. 如果已检索到相关案例且质量评估为"充足" → 不需要重规划
2. 如果检索结果为空或质量评估为"不足"，且尚未进行网络搜索 → 需要重规划（补充 get_google_search）
3. 如果执行中出现了无法恢复的错误 → 需要重规划
4. 如果已有 final_answer 或 analyze_legal_issue 已成功执行 → 不需要重规划
5. 如果已有足够案例且进行了法律分析 → 不需要重规划

## 输出 JSON
{{
    "needs_replan": false,
    "reason": "简短说明判断依据,不超过50字"
}}
"""


@traceable(run_type="chain", name="ReplanCheck_重规划检查节点")
def replan_check_node(state: AgentState) -> dict:
    """Flash LLM: 分析已执行步骤的结果,语义级判断是否需要重规划"""
    t0 = time.time()
    debug.debug("→ 进入 Replan Check 节点 (Flash LLM)", detail="LLM 语义判断执行质量...")

    # 拼装已执行步骤摘要executed_summary
    steps_desc: list[str] = []
    for s in state.plan:
        status_label = "✓" if s.status == "done" else "✗" if s.status == "failed" else "⋯"
        steps_desc.append(
            f"[{status_label}] 步骤{s.step_id}: {s.description} → 工具: {s.tool_name or '无'}"
        )
    executed_summary = "\n".join(steps_desc) if steps_desc else "无已执行步骤"

    # 评估结论
    quality_verdict = "未评估"
    if state.evaluation and state.evaluation.total > 0:
        ev = state.evaluation
        quality_verdict = (
            f"{ev.quality_verdict} (共{ev.total}条, "
            f"高质量{ev.correct_count}, 中等{ev.ambiguous_count}, 低质量{ev.incorrect_count})"
        )

    prompt = REPLAN_CHECK_PROMPT.format(
        user_query=state.query,
        executed_summary=executed_summary,
        doc_count=len(state.rag_documents),
        quality_verdict=quality_verdict,
        web_count=len(state.web_search_results),
        error_info=state.error or "无",
    )

    try:
        chain = PromptTemplate.from_template(prompt) | llm_executor | StrOutputParser()
        raw = chain.invoke({})
        raw = raw.strip()
        
        # 提取JSON部分,兼容 LLM 输出中夹带文本的情况
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]
        result = json.loads(raw)
        needs = result.get("needs_replan", False)
        reason = result.get("reason", "")
    except Exception as e:
        # 降级: LLM 判断失败时回退到规则判断
        debug.warning(
            "Replan Check LLM 解析失败,降级为规则判断",
            detail=str(e),
        )
        needs, reason = _fallback_replan_check(state)

    elapsed = time.time() - t0
    if needs:
        debug.info(
            "← Replan Check: 需要重规划",
            detail=reason,
            result=f"elapsed={elapsed:.2f}s | → Replanner",
        )
    else:
        debug.info(
            "← Replan Check: 质量通过",
            detail=reason,
            result=f"elapsed={elapsed:.2f}s | → Finalize",
        )
    return {
        "replan_needed": needs,
        "replan_reason": reason or None,
    }


def _fallback_replan_check(state: AgentState) -> tuple[bool, str]:
    """规则兜底判断 —— LLM 解析失败时使用"""
    if state.error:
        return True, f"执行异常: {state.error}"
    if (
        state.evaluation
        and state.evaluation.quality_verdict == "不足,建议进行网络搜索补充"
    ):
        executed = {tc.tool_name for tc in state.tool_calls}
        if "get_google_search" not in executed:
            return True, "检索质量不足,需补联网搜索"
    return False, "规则兜底: 无明显问题"


# Node 4: The Replanner — Pro LLM 重新规划


REPLANNER_SYSTEM = """
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


@traceable(run_type="chain", name="Replanner_重规划节点")
def replanner_node(state: AgentState) -> dict:
    """Pro LLM: 检查当前结果 → 生成补充计划 → 返回 Executor"""
    t0 = time.time()
    reason = state.replan_reason or "质量不足"
    debug.debug(
        "→ 进入 Replanner 节点",
        detail=f"原因: {reason} | 已完成{len(state.plan)}步",
    )

    executed = "\n".join(
        f"[{'done' if s.status == 'done' else 'failed'}] "
        f"步骤{s.step_id}: {s.description} → {s.tool_name}"
        for s in state.plan
    )

    tools_desc = "\n".join(f"- {t.name}: {t.description[:120]}" for t in ALL_TOOLS)

    # 生成补充步骤
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

    chain = llm_planner | StrOutputParser()
    raw = chain.invoke(prompt)

    try:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]
        result = json.loads(raw)
    except json.JSONDecodeError:
        debug.warning("Replanner JSON 解析失败", detail="使用默认补充步骤")
        default = [
            PlanStep(
                step_id=len(state.plan) + 1,
                description="联网搜索补充",
                tool_name="get_google_search",
            ),
            PlanStep(
                step_id=len(state.plan) + 2,
                description="综合信息生成分析",
                tool_name="analyze_legal_issue",
            ),
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
        new_steps.append(
            PlanStep(
                step_id=base + len(new_steps) + 1,
                description=p.get("description", ""),
                tool_name=tn,
            )
        )

    elapsed = time.time() - t0
    new_step_names = [f"{s.step_id}.{s.tool_name}" for s in new_steps]
    debug.info(
        "← Replanner 完成",
        detail=f"新增{len(new_steps)}步: {' , '.join(new_step_names)}",
        result=f"elapsed={elapsed:.2f}s | → 路由回 Executor",
    )
    return {
        "plan": state.plan + new_steps,
        "reasoning": state.reasoning
        + [f"[Replan] {state.replan_reason}"]
        + new_reasoning,
        "replan_needed": False,
        "replan_reason": None,
        "error": None,
    }


# Node 5: Finalize — 组装最终回答


@traceable(run_type="chain", name="Finalize_最终组装节点")
def finalize_node(state: AgentState) -> dict:
    """如果已有 final_answer 则直接使用；否则用已检索案例生成简要回答"""
    t0 = time.time()
    debug.debug("→ 进入 Finalize 节点", detail="组装最终回答...")

    if state.final_answer:
        debug.info(
            "← Finalize 完成 (已有答案)",
            detail=f"answer_len={len(state.final_answer)}",
            result=f"elapsed={time.time() - t0:.2f}s",
        )
        return {}

    if state.rag_documents:
        debug.debug("Finalize 兜底生成", detail=f"使用{len(state.rag_documents[:3])}条案例")
        docs = "\n".join(f"- {d.chunk_text[:300]}" for d in state.rag_documents[:3])
        prompt = PromptTemplate.from_template(
            "基于以下案例,简要回答用户问题.\n案例:\n{docs}\n\n问题: {query}\n\n法律建议:"
        )
        chain = prompt | llm_executor | StrOutputParser()
        answer = chain.invoke({"docs": docs, "query": state.query})
        debug.info(
            "← Finalize 完成 (兜底)",
            detail=f"answer_len={len(answer)}",
            result=f"elapsed={time.time() - t0:.2f}s",
        )
        return {"final_answer": answer}

    debug.info("← Finalize 完成", detail="无可用数据", result="返回默认欢迎语")
    return {"final_answer": "您好！请问有什么法律问题需要咨询？"}


# 条件路由函数 (Conditional Edges)


def route_after_planner(state: AgentState) -> str:
    """有步骤 → executor | 无步骤 → finalize"""
    target = "executor" if state.plan else "finalize"
    debug.debug(f"路由: Planner → {target}", detail=f"plan_steps={len(state.plan)}")
    return target


def route_after_executor(state: AgentState) -> str:
    """还有步骤 → 继续 executor | 全部完成 → replan_check"""
    target = "executor" if state.current_step_index < len(state.plan) else "replan_check"
    debug.debug(
        f"路由: Executor → {target}",
        detail=f"step={state.current_step_index}/{len(state.plan)}",
    )
    return target


def route_after_replan_check(state: AgentState) -> str:
    """需重规划 → replanner | 质量通过 → finalize"""
    target = "replanner" if state.replan_needed else "finalize"
    debug.debug(
        f"路由: ReplanCheck → {target}",
        detail=f"replan_needed={state.replan_needed}",
    )
    return target


# 构建 Graph


def build_graph(checkpointer=None):
    builder = StateGraph(AgentState)

    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("replan_check", replan_check_node)
    builder.add_node("replanner", replanner_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "planner")

    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {"executor": "executor", "finalize": "finalize"},
    )

    builder.add_conditional_edges(
        "executor",
        route_after_executor,
        {"executor": "executor", "replan_check": "replan_check"},
    )

    builder.add_conditional_edges(
        "replan_check",
        route_after_replan_check,
        {"replanner": "replanner", "finalize": "finalize"},
    )

    builder.add_edge("replanner", "executor")
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)


checkpointer = MemorySaver()
graph = build_graph(checkpointer=checkpointer)
