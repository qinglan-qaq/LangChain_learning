from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
import uvicorn

from lawApp_LangGraph.LangGraph_lawApp import graph
from lawApp_LangGraph.tools import ALL_TOOLS
from lawApp_LangGraph.FastAPI.model import (
    QueryRequest,
    QueryResponse,
    ToolInfo,
)
from lawApp_LangGraph.FastAPI.utils import (
    build_response,
    ensure_session,
    invoke_graph,
    sse_event,
)
from lawApp_LangGraph.FastAPI.logging import (
    setup_logging,
    set_session,
    flow,
    debug,
    system,
)
from dotenv import load_dotenv

load_dotenv(dotenv_path="lawApp_LangGraph/.env")

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(
        log_dir=os.getenv("LOG_DIR", "./logs"),
        console_level=os.getenv("LOG_CONSOLE_LEVEL", "DEBUG"),
        file_level=os.getenv("LOG_FILE_LEVEL", "INFO"),
    )
    system.info("系统启动", detail=f"日志系统已初始化", result="Legal Consultation API v2.0.0")
    yield
    system.info("系统关闭", detail="服务正在关闭")

app = FastAPI(
    title="Legal Consultation API",
    description="基于 LangGraph 的法律咨询后端接口",
    version="2.0.0",
    lifespan=lifespan,
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# HTTP 请求/响应日志中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    elapsed = time.time() - t0
    system.info(
        f"{request.method} {request.url.path}",
        detail=f"status={response.status_code}",
        result=f"elapsed={elapsed:.3f}s",
    )
    return response


# 普通询问接口
@app.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):
    """同步问答: 等待完整结果后返回 JSON."""
    sid = ensure_session(request.session_id)
    set_session(sid)
    query_preview = request.query[:80].replace("\n", " ")
    
    flow.info("流程开始", summary="用户提问", detail=f"query={query_preview}")

    t0 = time.time()
    try:
        state = await invoke_graph(request.query, sid)
    except Exception as e:
        flow.error("流程异常", summary="Graph 执行失败", detail=str(e))
        raise HTTPException(status_code=500, detail=f"Graph 执行失败: {e}")

    elapsed = time.time() - t0
    response = build_response(state, sid)
    tool_names = response.tool_calls or []
    flow.info(
        "流程结束",
        summary="回答生成完毕",
        detail=f"answer_len={len(response.final_answer)}, tool_calls={len(tool_names)}",
        result=f"总耗时={elapsed:.2f}s | 工具: {', '.join(tool_names) if tool_names else '无'}",
    )
    return response


# SSE 流式询问接口
@app.get("/ask/stream")
async def ask_stream(query: str = "", session_id: str | None = None):
    """SSE 流式问答: 实时推送规划、工具调用进度和最终回答."""
    sid = ensure_session(session_id)
    set_session(sid)
    query_preview = query[:80].replace("\n", " ")
    flow.info("流式流程开始", summary="用户提问", detail=f"query={query_preview}")

    config = {"configurable": {"thread_id": sid}}
    debug.debug("流式请求配置就绪", detail=f"thread_id={sid}")

    async def event_stream():
        try:
            prev_step_idx = -1
            async for chunk in graph.astream(
                {"query": query}, config=config, stream_mode="values"
            ):
                plan = chunk.get("plan", []) or []
                step_idx = chunk.get("current_step_index", 0)
                if step_idx != prev_step_idx and step_idx < len(plan):
                    prev_step_idx = step_idx
                    step = plan[step_idx]
                    desc = (
                        step.get("description", "")
                        if isinstance(step, dict)
                        else getattr(step, "description", "")
                    )
                    tn = (
                        step.get("tool_name", "")
                        if isinstance(step, dict)
                        else getattr(step, "tool_name", "")
                    )
                    yield sse_event("progress", desc)
                    if tn:
                        yield sse_event("tool_call", tn)

            answer = chunk.get("final_answer", "")
            yield sse_event("answer", answer)
            yield sse_event("session_id", sid)
            yield sse_event("done")
            flow.info(
                "流式流程结束",
                summary="流式回答完成",
                result=f"answer_len={len(answer)}",
            )
        except Exception as e:
            flow.error("流式流程异常", detail=str(e))
            yield sse_event("error", str(e))

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# 生成 PDF 报告接口
@app.post("/ask/pdf")
async def ask_pdf(request: QueryRequest):
    """生成 PDF 报告并返回文件下载."""
    from lawApp_LangGraph.tools.tools import markdown_to_pdf

    sid = ensure_session(request.session_id)
    set_session(sid)
    query_preview = request.query[:80].replace("\n", " ")
    flow.info("PDF流程开始", summary="用户提问", detail=f"query={query_preview}")

    t0 = time.time()
    state = await invoke_graph(request.query, sid)

    answer = state.get("final_answer", "")
    if not answer:
        flow.error("PDF流程失败", detail="未能生成回答内容")
        raise HTTPException(status_code=500, detail="未能生成回答内容")

    safe_name = request.query[:30].strip().replace(" ", "_").replace("/", "_")
    filename = f"legal_answer_{safe_name}.pdf"
    pdf_path = markdown_to_pdf(answer, filename)

    if not pdf_path or not os.path.exists(pdf_path):
        flow.error("PDF流程失败", detail="PDF 生成失败")
        raise HTTPException(status_code=500, detail="PDF 生成失败")

    elapsed = time.time() - t0
    flow.info(
        "PDF流程结束",
        summary="PDF 报告已生成",
        detail=f"file={filename}",
        result=f"总耗时={elapsed:.2f}s",
    )
    return FileResponse(
        path=pdf_path,
        filename=os.path.basename(pdf_path),
        media_type="application/pdf",
    )


# 获取工具列表接口
@app.get("/tools", response_model=list[ToolInfo])
async def list_tools():
    """返回 Agent 可用的全部工具列表及描述."""
    tools = [ToolInfo(name=t.name, description=t.description or "") for t in ALL_TOOLS]
    system.info("工具列表查询", result=f"共 {len(tools)} 个工具可用")
    return tools


@app.get("/home")
async def home():
    return {
        "service": "Legal Consultation API",
        "version": "2.0.0",
        "endpoints": {
            "ask": "POST /ask",
            "ask_stream": "GET /ask/stream?query=xxx&session_id=xxx",
            "ask_pdf": "POST /ask/pdf",
            "tools": "GET /tools",
            "home": "GET /home",
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
