from __future__ import annotations

import os
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import uvicorn

from lawApp_LangGraph.LangGraph_lawApp import graph
from lawApp_LangGraph.tools.tools import markdown_to_pdf
from lawApp_LangGraph.FastAPI.model import (
    QueryRequest,
    QueryResponse,
    EvaluationResult,
    RetrievedDoc,
    Message,
)

app = FastAPI(
    title="Legal Consultation API",
    description="基于 LangGraph 的法律咨询后端接口",
    version="1.0.0",
)


def _extract_pdf_path(markdown_output: str) -> Optional[str]:
    """从 markdown_to_pdf 返回结果中提取实际 PDF 文件路径。"""
    if not markdown_output:
        return None

    if "文件路径：" in markdown_output:
        return markdown_output.split("文件路径：", 1)[-1].strip()

    if os.path.exists(markdown_output):
        return markdown_output

    return None


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """使用问答请求体处理用户问题，返回结构化结果。

    session_id 作为 LangGraph thread_id 实现短期记忆:
    - 客户端传入相同 session_id 可维持多轮对话上下文
    - 不传则自动生成新会话
    """
    try:
        session_id = request.session_id or uuid.uuid4().hex
        config = {"configurable": {"thread_id": session_id}}

        final_state = None
        for chunk in graph.stream({"query": request.query, "messages": []}, config=config):
            final_state = chunk

        if not final_state:
            raise HTTPException(status_code=500, detail="未能生成最终状态")

        answer = final_state.get("final_answer", "")
        if not answer:
            raise HTTPException(status_code=500, detail="未能生成答案")

        evaluation = final_state.get("evaluation", {}) or {}
        return QueryResponse(
            query=request.query,
            final_answer=answer,
            session_id=session_id,
            messages=[Message(**message) for message in final_state.get("messages", [])],
            crag_context=final_state.get("crag_context"),
            pdf_path=final_state.get("pdf_path"),
            is_law_questions=final_state.get("is_law_questions", False),
            is_simple_questions=final_state.get("is_simple_questions", False),
            evaluation=EvaluationResult(
                correct=[RetrievedDoc(**item) for item in evaluation.get("correct", [])],
                ambiguous=[RetrievedDoc(**item) for item in evaluation.get("ambiguous", [])],
                incorrect=[RetrievedDoc(**item) for item in evaluation.get("incorrect", [])],
            ),
            web_search_results=final_state.get("web_search_results", []),
            output_format="text",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理请求时出错: {str(e)}")


@app.post("/ask/pdf")
async def ask_question_pdf(request: QueryRequest):
    """生成 PDF 并返回下载文件。"""
    try:
        session_id = request.session_id or uuid.uuid4().hex
        config = {"configurable": {"thread_id": session_id}}

        final_state = None
        for chunk in graph.stream({"query": request.query, "messages": []}, config=config):
            final_state = chunk

        if not final_state:
            raise HTTPException(status_code=500, detail="未能生成最终状态")

        answer = final_state.get("final_answer", "")
        if not answer:
            raise HTTPException(status_code=500, detail="未能生成答案")

        pdf_filename = f"legal_answer_{request.query[:20].strip().replace(' ', '_')}.pdf"
        markdown_output = markdown_to_pdf(answer, pdf_filename)
        pdf_path = _extract_pdf_path(markdown_output)

        if not pdf_path or not os.path.exists(pdf_path):
            raise HTTPException(status_code=500, detail="PDF 生成失败")

        return FileResponse(path=pdf_path, filename=os.path.basename(pdf_path), media_type="application/pdf")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理请求时出错: {str(e)}")


@app.get("/home")
async def root():
    return {"message": "法律咨询 API 服务运行中", "version": "1.0.0"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)     