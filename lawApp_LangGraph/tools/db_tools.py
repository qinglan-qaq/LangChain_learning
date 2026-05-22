"""
长期记忆工具集 — 基于 PostgreSQL + pgvector

    search_memory  — 语义搜索过往记忆,召回相关上下文
    save_to_memory — 保存事实/偏好/对话摘要到长期记忆

Agent 可在关键节点调用这些工具,实现跨会话的知识积累。
"""

import os
import time
from typing import Optional
from dotenv import load_dotenv
import psycopg2
from langchain_core.tools import tool
from sentence_transformers import SentenceTransformer

from lawApp_LangGraph.FastAPI.logging import tool as tool_log, system as sys_log
from langsmith import traceable

load_dotenv()
_embedder = None
_conn = None


# 私有方法新建一个单例 SentenceTransformer 实例 嵌入模型
@traceable(run_type="llm", name="Embedder_嵌入模型初始化")
def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        sys_log.info(
            "初始化 Memory Embedder (冷启动)",
            detail=f"model={os.getenv('MEMORY_EMBED_MODEL', 'BAAI/bge-large-zh-v1.5')}",
        )
        _embedder = SentenceTransformer(
            os.getenv("MEMORY_EMBED_MODEL", "BAAI/bge-large-zh-v1.5")
        )
    return _embedder


# 私有方法新建一个单例 PostgreSQL 连接实例
@traceable(run_type="chain", name="DB_数据库连接")
def _get_conn():
    global _conn
    if _conn is None:
        _conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
        )
    return _conn


# 建表函数,确保 agent_memory 表存在,在 search_memory 和 save_to_memory 中调用
@traceable(run_type="chain", name="DB_记忆表初始化")
def ensure_memory_table():
    """在 law_app 数据库中创建 agent_memory 表 (如不存在)"""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS agent_memory (
            id          SERIAL PRIMARY KEY,
            thread_id   TEXT NOT NULL DEFAULT 'default',
            memory_type TEXT NOT NULL DEFAULT 'general',
            content     TEXT NOT NULL,
            embedding   VECTOR(1024),
            metadata    JSONB DEFAULT '{}'::jsonb,
            created_at  TIMESTAMP DEFAULT NOW()
        );
    """)
    conn.commit()
    cur.close()


# Tool A: 搜索记忆

@tool
@traceable(run_type="tool", name="tool_记忆搜索")
def search_memory(query: str, top_k: int = 3) -> dict:
    """搜索长期记忆库,召回与当前问题相关的历史信息。

    适用场景:
    - 用户提及之前讨论过的话题
    - 需要参考过往的法律偏好或决策
    - 跨会话的上下文补充

    参数:
    query: 搜索查询,描述需要回忆的内容
    top_k: 返回条数,默认 5

    返回:
    dict, 含 memories 列表,每项为 {memory_type, content, created_at, similarity}
    """
    t0 = time.time()
    tool_log.info(
        "→ 调用工具: search_memory",
        detail=f"query={query[:60]} | top_k={top_k}",
    )

    ensure_memory_table()

    embedder = _get_embedder()
    query_vec = embedder.encode(query, normalize_embeddings=True).tolist()

    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT memory_type, content, created_at,
        1 - (embedding <=> %s::vector) AS similarity
        FROM agent_memory
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (query_vec, query_vec, top_k),
    )
    rows = cur.fetchall()
    cur.close()

    memories = [
        {
            "memory_type": r[0],
            "content": r[1],
            "created_at": r[2].isoformat() if r[2] else "",
            "similarity": round(float(r[3]), 4),
        }
        for r in rows
    ]

    top_sim = memories[0]["similarity"] if memories else 0
    tool_log.info(
        "← 工具返回: search_memory",
        detail=f"命中{len(memories)}条记忆",
        result=f"top_similarity={top_sim:.3f} | elapsed={time.time() - t0:.2f}s",
    )
    return {
        "memory_results": memories,
        "status": "success" if memories else "empty",
        "count": len(memories),
    }


# Tool C: 法律条文检索

@tool
@traceable(run_type="tool", name="tool_法律条文检索")
def fetch_laws(query: str, top_k: int = 5) -> dict:
    """从法律条文数据库中语义检索相关法条.使用 PGVector 向量相似度搜索,
    召回与查询问题最相关的法律法规条文,为法律分析提供权威依据.

    适用场景:
    - 需要引用具体法律条文支撑法律意见
    - 查找特定领域的法律法规,不仅限于某部法律
    - 确认某法律问题的适用法条
    - Planner 判断回答需要法律条文依据时优先调用

    参数:
    query: 法律问题或关键词,用于语义匹配相关法条,中文
    top_k: 返回条数,默认 5

    返回:
    dict, 含 law_results 列表,每项为 {law_title, chapter, article_number, content, similarity}
    """
    t0 = time.time()
    tool_log.info(
        "→ 调用工具: fetch_laws",
        detail=f"query={query[:80]} | top_k={top_k}",
    )

    embedder = _get_embedder()
    query_vec = embedder.encode(query, normalize_embeddings=True).tolist()

    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT law_title, chapter, article_number, content,
        1 - (embedding <=> %s::vector) AS similarity
        FROM law_vector
        WHERE embedding IS NOT NULL
        AND status = '现行有效'
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (query_vec, query_vec, top_k),
    )
    rows = cur.fetchall()
    cur.close()

    laws = [
        {
            "law_title": r[0],
            "chapter": r[1] or "",
            "article_number": r[2],
            "content": r[3][:600],
            "similarity": round(float(r[4]), 4),
        }
        for r in rows
    ]

    top_sim = laws[0]["similarity"] if laws else 0
    tool_log.info(
        "← 工具返回: fetch_laws",
        detail=f"命中{len(laws)}条法条",
        result=f"top_similarity={top_sim:.3f} | elapsed={time.time() - t0:.2f}s",
    )
    return {
        "law_results": laws,
        "status": "success" if laws else "empty",
        "count": len(laws),
    }


# Tool D: 保存记忆

MAX_EMBED_LEN = 512  # 嵌入文本上限, 超出自动截断


@tool
@traceable(run_type="tool", name="tool_记忆保存")
def save_to_memory(
    content: str,
    memory_type: str = "general",
    thread_id: str = "default",
    summary: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> dict:
    """将重要信息保存到长期记忆库,供未来会话使用。

    关键: content 是完整原文(存 metadata), summary 是简短摘要(用于向量检索)。
    请自行提供一句话 summary,避免长文本被嵌入后拉高 token 成本。
    如果不传 summary 且 content 较短(<512字),则直接用 content 做嵌入。

    适用场景:
    - 用户明确表达了偏好或需求
    - 对话中得到了重要的结论或建议
    - 用户告知了个人情况(职业、所在地等)
    - 法律咨询的关键结论

    参数:
    content: 要保存的完整记忆文本
    summary: 简短摘要(1-2句),用于语义搜索匹配。不传则用 content 截断
    memory_type: 记忆类型,如 'user_fact' / 'legal_preference' / 'conclusion' / 'general'
    thread_id: 会话线程标识,默认 'default'
    metadata: 附加元数据,如 {'law_title': '民法典', 'article': '第一千零四十二条'}

    返回:
    dict, 含 status / id / memory_type / is_truncated
    """
    t0 = time.time()
    tool_log.info(
        "→ 调用工具: save_to_memory",
        detail=f"type={memory_type} | thread={thread_id} | content_len={len(content)}",
    )

    ensure_memory_table()

    embed_text = (summary or content).strip()

    is_truncated = False
    if len(embed_text) > MAX_EMBED_LEN:
        embed_text = embed_text[:MAX_EMBED_LEN]
        is_truncated = True

    embedder = _get_embedder()
    embedding = embedder.encode(embed_text, normalize_embeddings=True).tolist()

    import json

    meta = metadata or {}
    if summary:
        meta["full_content"] = content

    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO agent_memory (thread_id, memory_type, content, embedding, metadata)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id
        """,
        (thread_id, memory_type, embed_text, embedding, json.dumps(meta, ensure_ascii=False)),
    )
    new_id = cur.fetchone()[0]
    conn.commit()
    cur.close()

    msg = f"记忆已保存 (id={new_id}, type={memory_type}"
    if is_truncated:
        msg += f", 嵌入文本已截断至 {MAX_EMBED_LEN} 字"
    msg += ")"

    tool_log.info(
        "← 工具返回: save_to_memory",
        detail=f"id={new_id} | type={memory_type}" + (" | truncated" if is_truncated else ""),
        result=f"elapsed={time.time() - t0:.2f}s",
    )
    return {
        "memory_update": {
            "id": new_id,
            "memory_type": memory_type,
            "is_truncated": is_truncated,
            "summary": embed_text,
        },
        "status": "success",
        "message": msg,
    }


