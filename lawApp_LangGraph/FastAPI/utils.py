from __future__ import annotations

import json
import uuid
from typing import Optional

from lawApp_LangGraph.LangGraph_lawApp import graph
from lawApp_LangGraph.FastAPI.model import QueryResponse, SourceInfo

# 获取确保会话 ID
def ensure_session(session_id: Optional[str]) -> str:
    return session_id or uuid.uuid4().hex

# 调用 LangGraph
async def invoke_graph(query: str, session_id: str) -> dict:
    config = {"configurable": {"thread_id": session_id}}
    return await graph.ainvoke({"query": query}, config=config)


def build_sources(state: dict) -> list[SourceInfo]:
    sources: list[SourceInfo] = []
    seen: set[str] = set()

    for doc in state.get("rag_documents", []) or []:
        if isinstance(doc, dict):
            cn, yr, txt = (
                doc.get("case_number", ""),
                doc.get("year", ""),
                doc.get("chunk_text", ""),
            )
        else:
            cn, yr, txt = (
                getattr(doc, "case_number", ""),
                getattr(doc, "year", ""),
                getattr(doc, "chunk_text", ""),
            )
        key = f"{cn}-{yr}"
        if key not in seen and cn:
            seen.add(key)
            sources.append(SourceInfo(case_number=cn, year=yr, snippet=txt[:200]))

    for item in state.get("web_search_snippets", []) or []:
        if isinstance(item, dict):
            title, link, snippet = item.get("title", ""), item.get("link", ""), item.get("snippet", "")
        else:
            title = getattr(item, "title", "")
            link = getattr(item, "link", "")
            snippet = getattr(item, "snippet", "")
        sources.append(
            SourceInfo(
                title=title,
                link=link,
                snippet=snippet,
            )
        )

    return sources


def build_tool_calls(state: dict) -> list[str]:
    calls = state.get("tool_calls", []) or []
    result: list[str] = []
    for tc in calls:
        if isinstance(tc, dict):
            result.append(tc.get("tool_name", ""))
        else:
            result.append(getattr(tc, "tool_name", ""))
    return [r for r in result if r]


def build_response(state: dict, session_id: str) -> QueryResponse:
    return QueryResponse(
        query=state.get("query", ""),
        session_id=session_id,
        final_answer=state.get("final_answer", ""),
        sources=build_sources(state),
        tool_calls=build_tool_calls(state),
        reasoning=state.get("reasoning", []) or [],
    )


def sse_event(event: str, data: str = "") -> str:
    return f"data: {json.dumps({'event': event, 'data': data}, ensure_ascii=False)}\n\n"
