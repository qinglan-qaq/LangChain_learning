from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Legal Consultation API",
    description="基于 LangGraph 的法律咨询后端接口",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 端点 ──────────────────────────────────────────


@app.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):
    """同步问答: 等待完整结果后返回 JSON。"""
    sid = ensure_session(request.session_id)
    state = await invoke_graph(request.query, sid)
    return build_response(state, sid)


@app.post("/ask/stream")
async def ask_stream(request: QueryRequest):
    """SSE 流式问答: 实时推送规划、工具调用进度和最终回答。

    event 类型:
      - progress:   当前步骤描述
      - tool_call:  正在调用的工具名
      - answer:     最终回答全文
      - session_id: 会话 ID (客户端保存用于多轮)
      - done:       流结束
      - error:      异常信息
    """
    sid = ensure_session(request.session_id)
    config = {"configurable": {"thread_id": sid}}

    async def event_stream():
        try:
            prev_step_idx = -1

            async for chunk in graph.astream(
                {"query": request.query}, config=config, stream_mode="values"
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

        except Exception as e:
            yield sse_event("error", str(e))

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/ask/pdf")
async def ask_pdf(request: QueryRequest):
    """生成 PDF 报告并返回文件下载。"""
    from lawApp_LangGraph.tools.tools import markdown_to_pdf

    sid = ensure_session(request.session_id)
    state = await invoke_graph(request.query, sid)

    answer = state.get("final_answer", "")
    if not answer:
        raise HTTPException(status_code=500, detail="未能生成回答内容")

    safe_name = request.query[:30].strip().replace(" ", "_").replace("/", "_")
    filename = f"legal_answer_{safe_name}.pdf"
    pdf_path = markdown_to_pdf(answer, filename)

    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(status_code=500, detail="PDF 生成失败")

    return FileResponse(
        path=pdf_path,
        filename=os.path.basename(pdf_path),
        media_type="application/pdf",
    )


@app.get("/tools", response_model=list[ToolInfo])
async def list_tools():
    """返回 Agent 可用的全部工具列表及描述。"""
    return [ToolInfo(name=t.name, description=t.description or "") for t in ALL_TOOLS]


@app.get("/home")
async def home():
    return {
        "service": "Legal Consultation API",
        "version": "2.0.0",
        "endpoints": {
            "ask": "POST /ask",
            "ask_stream": "POST /ask/stream",
            "ask_pdf": "POST /ask/pdf",
            "tools": "GET /tools",
            "home": "GET /home",
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
