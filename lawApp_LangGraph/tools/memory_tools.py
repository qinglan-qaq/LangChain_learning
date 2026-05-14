"""
长期记忆工具集 — 基于 PostgreSQL + pgvector

    search_memory  — 语义搜索过往记忆,召回相关上下文
    save_to_memory — 保存事实/偏好/对话摘要到长期记忆

Agent 可在关键节点调用这些工具,实现跨会话的知识积累。
"""

import os
from typing import Optional

import psycopg2
from langchain_core.tools import tool
from sentence_transformers import SentenceTransformer

# ---- 懒加载单例 ----
_embedder = None
_conn = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(
            os.getenv("MEMORY_EMBED_MODEL", "BAAI/bge-large-zh-v1.5")
        )
    return _embedder


def _get_conn():
    global _conn
    if _conn is None:
        _conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME", "Law_app"),
            user=os.getenv("DB_USER", "my_pgsql"),
            password=os.getenv("DB_PASSWORD", "123123"),
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5433")),
        )
    return _conn


# ---- 建表 (幂等) ----
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


# =
# Tool A: 搜索记忆
# =


@tool
def search_memory(query: str, top_k: int = 5) -> dict:
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

    return {
        "status": "success" if memories else "empty",
        "count": len(memories),
        "memories": memories,
    }


# =
# Tool B: 保存记忆
# =


@tool
def save_to_memory(
    content: str,
    memory_type: str = "general",
    thread_id: str = "default",
    metadata: Optional[dict] = None,
) -> dict:
    """将重要信息保存到长期记忆库,供未来会话使用。

    适用场景:
    - 用户明确表达了偏好或需求
    - 对话中得到了重要的结论或建议
    - 用户告知了个人情况(职业、所在地等)
    - 法律咨询的关键结论

    参数:
    content: 要保存的记忆文本
    memory_type: 记忆类型,如 'user_fact' / 'legal_preference' / 'conclusion' / 'general'
    thread_id: 会话线程标识,默认 'default'
    metadata: 附加元数据,如 {'law_title': '民法典', 'article': '第一千零四十二条'}

    返回:
    dict, 含 status / id / memory_type
    """
    ensure_memory_table()

    embedder = _get_embedder()
    embedding = embedder.encode(content, normalize_embeddings=True).tolist()

    import json

    meta_json = json.dumps(metadata or {}, ensure_ascii=False)

    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO agent_memory (thread_id, memory_type, content, embedding, metadata)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id
        """,
        (thread_id, memory_type, content, embedding, meta_json),
    )
    new_id = cur.fetchone()[0]
    conn.commit()
    cur.close()

    return {
        "status": "success",
        "id": new_id,
        "memory_type": memory_type,
        "message": f"记忆已保存 (id={new_id}, type={memory_type})",
    }
