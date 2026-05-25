from __future__ import annotations

import asyncio
import contextvars
import json
import uuid
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from lawApp_LangGraph.LangGraph_lawApp import graph

from lawApp_LangGraph.FastAPI.model import QueryResponse, SourceInfo

# ── Stream Queue (context var 穿透 LangGraph 节点 & 工具) ──

_stream_queue: contextvars.ContextVar[asyncio.Queue | None] = contextvars.ContextVar(
    "stream_queue", default=None
)


def set_stream_queue(q: asyncio.Queue | None) -> None:
    _stream_queue.set(q)


def get_stream_queue() -> asyncio.Queue | None:
    return _stream_queue.get(None)


# ── 工具函数 ──


def ensure_session(session_id: Optional[str]) -> str:
    if not session_id or not session_id.strip():
        return uuid.uuid4().hex
    return session_id


async def invoke_graph(query: str, session_id: str) -> dict:
    from lawApp_LangGraph.LangGraph_lawApp import graph

    config = {"configurable": {"thread_id": session_id}}
    return await graph.ainvoke({"query": query}, config=config)


def _field(item, key: str, default: str = ""):
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


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

    for item in state.get("web_search_results", []) or []:
        if isinstance(item, dict):
            title, link, snippet = (
                item.get("title", ""),
                item.get("link", ""),
                item.get("snippet", ""),
            )
        else:
            title = getattr(item, "title", "")
            link = getattr(item, "link", "")
            snippet = getattr(item, "snippet", "")
        sources.append(SourceInfo(title=title, link=link, snippet=snippet))
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
        final_prompt=state.get("final_prompts", ""),
        sources=build_sources(state),
        tool_calls=build_tool_calls(state),
        reasoning=state.get("reasoning", []) or [],
    )


def sse_event(event: str, data: str = "") -> str:
    return (
        f"data: {json.dumps({'event': event, 'data': data}, ensure_ascii=False)}\n\n"
    )
