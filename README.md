# lawApp_LangGraph — 法律智能咨询 Agent

<div align="center">

基于 **LangGraph Plan & Execute** 架构的婚姻家庭法 AI 咨询系统，集成混合检索增强生成 (CRAG)、长期记忆与流式服务。

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0-1C3C3C?style=flat&logo=langchain)](https://langchain-ai.github.io/langgraph/)
[![Pinecone](https://img.shields.io/badge/Pinecone-Serverless-1C17FF?style=flat)](https://www.pinecone.io/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-4169E1?style=flat&logo=postgresql)](https://www.postgresql.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)](LICENSE)

</div>

---

## 目录

- [架构概览](#架构概览)
- [核心亮点](#核心亮点)
- [Agent 工作流](#agent-工作流)
- [检索系统](#检索系统)
- [工具链](#工具链)
- [API 接口](#api-接口)
- [数据模型](#数据模型)
- [日志系统](#日志系统)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [技术栈](#技术栈)

---

## 架构概览

系统采用 **Plan & Execute** 范式：由强模型制定执行计划，轻量模型逐步执行，在保证回答质量的同时大幅降低推理成本。

```text
用户请求 → FastAPI → LangGraph StateGraph
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
    planner         executor        replan_check
  (DeepSeek Pro)  (DeepSeek Flash)  (DeepSeek Flash)
        │               │               │
        │   ┌───────────┘         ┌─────┴─────┐
        │   ▼                     ▼           ▼
        │  工具调用            finalize    replanner
        │  ├─ Pinecone 混合检索     │        (DeepSeek Pro)
        │  ├─ SerpAPI 联网搜索      │           │
        │  ├─ PostgreSQL 记忆       │           ▼
        │  └─ PDF 报告生成          │       executor
        │                           │
        └───────────────────────────┘
```

**双 LLM 成本优化**：规划/重规划使用 DeepSeek Pro（强推理），执行/质量门控使用 DeepSeek Flash（低成本低延迟），单次查询可节省 **40-60% token 消耗**。

---

## 核心亮点

### 架构设计

| 亮点 | 说明 |
|------|------|
| **Plan & Execute 柔性管线** | Agent 根据问题复杂度自主决策——闲聊跳过执行直接回答，法律问题走完整 CRAG 管线 |
| **三级优雅降级** | JSON 解析失败 → 默认计划兜底；LLM 调用失败 → 参数映射兜底；语义判断失败 → 规则兜底 |
| **质量闭环** | 检索 → 评估(三档) → 不足则联网补充 → 重规划 → 再执行，形成自纠正反馈环 |
| **全链路类型安全** | Pydantic v2 覆盖工具返回值 → 图状态 → API 响应的完整数据流 |

### 检索增强

| 亮点 | 说明 |
|------|------|
| **混合检索** | 密集语义匹配 (BGE) + 稀疏关键词匹配 (BM25)，可调 alpha 权重兼顾"意思相近"与"关键词命中" |
| **CrossEncoder 重排序** | BGE-reranker-large 全注意力交互打分，非简单向量距离，Top-N 精度大幅提升 |
| **CRAG 三档评估** | 将文档分为 correct / ambiguous / incorrect 三档，Agent 根据评估结论自主决定是否联网补充 |

### 工程实践

| 亮点 | 说明 |
|------|------|
| **懒加载单例** | RAG 服务、嵌入模型、LLM、数据库连接全部延迟初始化，避免冷启动峰值 |
| **5 类结构化日志** | 通过 contextvars 实现 session_id 全链路透传，无需修改任何函数签名 |
| **SSE 流式推送** | 实时展示推理 token、工具调用、最终答案，消除长等待焦虑 |
| **并发安全** | 基于 LangGraph MemorySaver 检查点实现线程级会话隔离 |

---

## Agent 工作流

`LangGraph StateGraph` 包含 **5 个节点** 与 **4 个条件边**：

### 节点

| 节点 | 模型 | 职责 |
|------|------|------|
| `planner` | DeepSeek Pro (t=0.4, 4096 tokens) | 分析用户问题，输出 JSON 推理链 + 分步计划 |
| `executor` | DeepSeek Flash (t=0.25, 2048 tokens) | 按步骤调用工具，支持 LLM Function Calling 与直接参数映射降级 |
| `replan_check` | DeepSeek Flash | 评估执行质量（案例数、评估结果、错误），判定是否需要重规划 |
| `replanner` | DeepSeek Pro | 基于已执行步骤生成补充计划（不重复，最多 10 轮） |
| `finalize` | — | 组装最终答案，已有答案直接用，否则 LLM 兜底生成 |

### 路由逻辑

```text
START → planner ──[plan 为空]──→ finalize → END
          │
          └──[有步骤]──→ executor ──[有剩余步骤]──→ executor (循环)
                              │
                              └──[无剩余]──→ replan_check ──[通过]──→ finalize
                                                   │
                                                   └──[未通过]──→ replanner → executor
```

### 多轮对话示例

```text
用户: "离婚后彩礼能要回来吗?"
  │
  ├─ planner: 识别为婚姻财产纠纷 → 制定 CRAG 三步计划
  │
  ├─ executor #1: 调用 retrieve_legal_knowledge → Pinecone 混合检索 → 返回 5 条判例
  ├─ executor #2: 调用 evaluate_case_relevance → 2条 correct, 1条 ambiguous → 判定不足
  │
  ├─ replan_check: 质量不充分 → 触发重规划
  ├─ replanner: 补充联网搜索 + 重新分析
  │
  ├─ executor #3: 调用 get_google_search → 获取最新司法解释
  ├─ executor #4: 调用 analyze_legal_issue → Saul Goodman 风格法律分析
  │
  ├─ replan_check: 已有 final_answer → 通过
  └─ finalize: 返回完整答案 + 援引案例 + 法条依据
```

---

## 检索系统

### 混合检索管线

```text
用户 Query
  ├─→ BGE-large-zh-v1.5 (密集向量, 1024 维)
  ├─→ BM25Encoder (稀疏向量, 预计算参数)
  └─→ convex_scale(alpha=0.7) 融合
       ↓
  Pinecone 混合查询 (dotproduct)
       ↓
  BGE-reranker-large CrossEncoder 重排序
       ↓
  返回 Top-N 结果
```

### 关键参数

| 组件 | 选型 | 说明 |
|------|------|------|
| 向量数据库 | Pinecone Serverless (us-east-1) | 密集 + 稀疏双向量存储 |
| 嵌入模型 | BAAI/bge-large-zh-v1.5 | 1024 维，中文语义优化，归一化输出 |
| 重排序 | BAAI/bge-reranker-large | CrossEncoder 全注意力架构 |
| 稀疏编码 | pinecone-text BM25Encoder | 预计算参数 ~387KB，启动即用 |
| 文本分割 | MarkdownHeaderTextSplitter + RecursiveCharTextSplitter | chunk_size=512, overlap=50 |

### 数据规模

- **11 个** 年度法院案例文件（2014-2024，婚姻家庭与继承纠纷）
- **约 5,030 条** 向量记录
- **7 部** 相关法律文本（民法典、反家暴法、妇女权益保障法等）

---

## 工具链

系统提供 **7 个工具**，通过 `langchain_core.tools` 注册，执行器按需调用：

| 工具 | 分类 | 功能 |
|------|------|------|
| `retrieve_legal_knowledge` | CRAG | Pinecone 混合检索法律案例，支持 top_k / alpha / namespace 参数调优 |
| `evaluate_case_relevance` | CRAG | 三档质量评估（correct ≥0.7 / ambiguous 0.3~0.7 / incorrect <0.3） |
| `analyze_legal_issue` | CRAG | 角色化法律分析生成，整合案例、法条与网络资料 |
| `get_google_search` | 外部搜索 | SerpAPI 联网搜索，最多 8 条结构化结果 |
| `search_memory` | 长期记忆 | PostgreSQL + pgvector 语义记忆搜索 (cosine 相似度) |
| `save_to_memory` | 长期记忆 | 保存用户事实/偏好，支持 memory_type 分类 |
| `markdown_to_pdf` | 输出 | Markdown → HTML → PDF (A4, 中文字体) |

### 长期记忆

- **存储引擎**：PostgreSQL 15 + pgvector 扩展
- **嵌入维度**：1024（与 Pinecone 共用 BGE 模型）
- **表结构**：`agent_memory(id, thread_id, memory_type, content, embedding, metadata, created_at)`
- **搜索**：`cosine_similarity = 1 - (embedding <=> query_vec)`
- **设计要点**：原文存 metadata，摘要/截断文本用于向量嵌入；建表幂等；懒加载单例

---

## API 接口

| 端点 | 方法 | 功能 |
|------|------|------|
| `/ask` | POST | 同步问答，返回完整 JSON（含推理链、工具调用、援引来源） |
| `/ask/stream` | POST | SSE 流式问答，实时推送推理 token / 工具调用 / 最终答案 |
| `/ask/pdf` | POST | 生成 PDF 法律报告并返回文件下载 |
| `/tools` | GET | 列出所有可用工具及参数描述 |
| `/home` | GET | 健康检查 + 服务信息 |

### 请求 / 响应

```python
# 请求
QueryRequest(
    query: str,           # 1-5000 字符
    session_id: str | None # 可选，支持多轮对话
)

# 响应
QueryResponse(
    query: str,
    session_id: str,
    final_answer: str,
    sources: list[str],
    tool_calls: list[ToolCallRecord],
    reasoning: str
)
```

---

## 数据模型

三层 Pydantic v2 模型体系，覆盖全数据流：

```python
# A. 工具返回层
RetrievedDocument  # rank, rerank_score, hybrid_score, year, case_number, chunk_text
EvaluationResult   # correct/ambiguous/incorrect 计数与列表, quality_verdict
WebSearchResult    # title, link, snippet

# B. 计划执行层
PlanStep           # step_id, description, tool_name, status, retry_count
ToolCallRecord     # step_id, tool_name, tool_input, output, timestamp

# C. 顶层 AgentState
AgentState         # 会话标识 · 请求上下文 · 计划执行 · 输出 · 管线数据 · 记忆 · 流控
```

---

## 日志系统

五类结构化日志，通过 `contextvars` 实现 session_id 全链路透传：

| Logger | 输出 | 级别 | 用途 |
|--------|------|------|------|
| `agent_flow` | 文件 (10MB 轮转) + 控制台 | INFO+ | 每次请求流程摘要 |
| `agent_debug` | 控制台 | DEBUG | 节点级执行链路 |
| `tool` | 控制台 | DEBUG | 工具调用参数 / 返回值 |
| `rag` | 控制台 | DEBUG | RAG 检索各环节耗时 |
| `system` | 文件 (轮转) + 控制台 | INFO+ | 启动 / 关闭 / 异常 |

控制台格式（带 ANSI 颜色）：

```text
15:37:22 | INFO  | sess_1234 | agent_flow | 执行完毕 | 调用了3个工具 | 成功
```

---

## 快速开始

### 环境要求

- Python 3.11+
- PostgreSQL 15（需 pgvector 扩展）
- wkhtmltopdf（PDF 生成依赖）

### 安装

```bash
# 1. 克隆仓库
git clone <repo-url> && cd LangChain

# 2. 安装依赖
pip install -r lawApp_LangGraph/requirements.txt

# 3. 配置环境变量
cp lawApp_LangGraph/.env.example lawApp_LangGraph/.env
# 编辑 .env 填入 API Key 与数据库信息
```

### 环境变量

```ini
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
PINECONE_API_KEY=xxx
PINECONE_INDEX_NAME=pinecone-law-agent
SERPAPI_API_KEY=xxx
DB_NAME=Law_app
DB_USER=postgres
DB_PASSWORD=xxx
DB_HOST=localhost
DB_PORT=5433
```

### 启动

```bash
cd lawApp_LangGraph/FastAPI
python api.py

# 或使用 uvicorn
uvicorn lawApp_LangGraph.FastAPI.api:app --host 0.0.0.0 --port 8000 --reload
```

服务启动后访问 `http://localhost:8000/home` 验证健康状态。

---

## 项目结构

```text
lawApp_LangGraph/
├── LangGraph_lawApp.py          # 主入口：StateGraph 定义 (5节点 + 4条件边)
├── state.py                     # Pydantic 数据模型 (三层体系)
├── requirements.txt             # Python 依赖
├── .env                         # 环境变量
│
├── FastAPI/                     # Web 服务层
│   ├── api.py                   # FastAPI 应用 (v2.0.0)，5端点 + CORS + 生命周期
│   ├── model.py                 # 请求 / 响应模型
│   ├── utils.py                 # 会话管理、图调用、响应构建、SSE 事件队列
│   └── logging.py               # 5类日志 + contextvars 会话透传
│
├── tools/                       # Agent 工具集
│   ├── __init__.py              # 工具注册表 (ALL_TOOLS)
│   ├── tools.py                 # SerpAPI 搜索 + PDF 生成
│   ├── rag_tools.py             # CRAG 管线 (检索 / 评估 / 分析 + 角色提示词)
│   └── db_tools.py              # 长期记忆 + 法条检索 (pgvector)
│
├── RAG_service/                 # RAG 检索服务
│   ├── RAG_program.py           # 混合检索 + BM25 + CrossEncoder 重排序
│   └── bm25_law_params.json     # 预计算 BM25 参数
│
├── node/                        # 备用节点实现 (legacy)
│   └── langgraph_nodes.py       # 节点工厂函数
│
├── Documents/                   # 法律文档
│   ├── LawDocument/             # 7 部法律 TXT
│   └── MarkDownFiles/           # 11 个案例 MD (2014-2024)
│
└── logs/                        # 运行时日志
```

---

## 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| Agent 框架 | LangGraph | StateGraph 编排 + MemorySaver 检查点 |
| LLM | DeepSeek (langchain-openai) | Pro 规划 / Flash 执行，双模型架构 |
| Web 服务 | FastAPI + Uvicorn | REST API + SSE 流式 |
| 数据模型 | Pydantic v2 | 全链路类型约束 |
| 向量检索 | Pinecone Serverless | 密集 + 稀疏混合查询 |
| 嵌入 & 重排 | BGE-large-zh-v1.5 / BGE-reranker-large | HuggingFace + sentence-transformers |
| 稀疏编码 | pinecone-text BM25Encoder | 关键词匹配 |
| 长期记忆 | PostgreSQL 15 + pgvector | 语义记忆存取 |
| 联网搜索 | SerpAPI | Google 搜索结果结构化 |
| PDF 生成 | markdown + pdfkit (wkhtmltopdf) | Markdown → HTML → PDF |
| 文本分割 | langchain-text-splitters | MarkdownHeader + RecursiveChar |
| 日志 | Python logging + contextvars | 5 类结构化日志 + 会话透传 |
| 追踪 | LangSmith | LLM 调用链追踪（可选） |

---

<div align="center">
  <sub>Built with LangGraph · DeepSeek · Pinecone · FastAPI</sub>
</div>
