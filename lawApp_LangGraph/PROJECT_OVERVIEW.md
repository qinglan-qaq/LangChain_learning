# lawApp_LangGraph — 基于 LangGraph 的 Plan & Execute 法律智能咨询系统

## 一、项目定位

一个**生产级法律咨询 AI Agent 系统**，面向婚姻家庭法领域，基于 LangGraph 框架实现 Plan & Execute（先规划后执行）架构。支持多轮对话、混合检索增强生成(CRAG)、长期记忆、联网搜索补充、PDF 报告导出，通过 FastAPI 提供 RESTful + SSE 流式 API 服务。

---

## 二、技术架构总览

### 2.1 整体拓扑

```
用户 → FastAPI (/ask, /ask/stream, /ask/pdf)
         ↓
    LangGraph StateGraph (Plan & Execute)
         ↓
    ┌─────────────────────────────────────────┐
    │  planner (DeepSeek Pro)                 │  ← 制定 JSON 计划 + 思考链
    │     ↓                                   │
    │  executor (DeepSeek Flash) ⇄ 循环执行    │  ← 调用工具 / 降级兜底
    │     ↓                                   │
    │  replan_check (DeepSeek Flash)          │  ← 质量门控(LLM语义判断)
    │     ↓ (不充分)                           │
    │  replanner (DeepSeek Pro) → executor     │  ← 补充计划
    │     ↓ (充分)                             │
    │  finalize                               │  ← 组装最终回答
    └─────────────────────────────────────────┘
         ↓                ↓                ↓
   [Pinecone]      [SerpAPI]       [PostgreSQL+pgvector]
   混合检索+重排    联网搜索补充      长期记忆存取
```

### 2.2 核心设计模式

- **Plan & Execute Agent**：先由强模型制定完整执行计划，再由轻量模型逐步执行，大幅降低单次推理成本
- **CRAG (Corrective RAG)**：检索 → 评估(三档) → 质量判定 → 不足则联网补充 → 生成分析
- **Dual LLM 架构**：Pro 模型负责规划/重规划(需要强推理)，Flash 模型负责执行/质量检查(降低成本与延迟)
- **优雅降级**：每个 LLM 调用点都有 fallback 机制（JSON解析失败→默认计划，LLM调用失败→直接参数映射，语义判断失败→规则兜底）

---

## 三、Graph 节点详解 (LangGraph_lawApp.py)

### 3.1 节点 1：planner_node（规划节点）
- **模型**：DeepSeek Pro（temperature=0.4, max_tokens=4096）
- **输入**：用户 query + 可用工具列表
- **输出**：JSON 格式的 `reasoning`（思考链） + `plan`（步骤列表，每步含 step_id/description/tool_name）
- **兜底策略**：JSON 解析失败时，自动生成默认的法律检索三步计划（检索→评估→分析）
- **设计原则**：法律问题走 CRAG 管线，简单闲聊 plan 为空直接跳 finalize

### 3.2 节点 2：executor_node（执行节点）
- **模型**：DeepSeek Flash（temperature=0.25, max_tokens=2048）+ bind_tools(ALL_TOOLS)
- **双路径设计**：
  - 路径 A（优先）：Flash LLM 根据步骤描述和上下文自动选择工具参数并发起 Function Calling
  - 路径 B（降级）：LLM 调用失败时，通过预定义的 `_TOOL_FALLBACK_ARGS` 字典直接参数映射
- **状态合并**：工具返回的 dict 中与 AgentState 同名的 key 自动合并；`web_search_results` 特殊处理为追加而非覆盖
- **步骤追踪**：每步记录 `ToolCallRecord`（step_id, tool_name, input, output, timestamp）

### 3.3 节点 3：replan_check_node（质量门控）
- **模型**：DeepSeek Flash
- **功能**：分析已执行步骤的结果摘要 + 当前数据状态（案例数/质量评估/网络搜索/错误），LLM 语义判断是否需要重规划
- **降级**：LLM 判断失败时回退到 `_fallback_replan_check` 规则判断（检查错误 + 评估质量 + 是否已联网搜索）

### 3.4 节点 4：replanner_node（重规划节点）
- **模型**：DeepSeek Pro
- **功能**：基于已执行步骤和当前状态，生成**补充步骤**（不重复已完成步骤）
- **兜底**：JSON 解析失败时默认插入"联网搜索 + 法律分析"两步

### 3.5 节点 5：finalize_node（组装节点）
- **优先级**：已有 final_answer → 直接使用；有 RAG 文档 → LLM 兜底生成；都没有 → 默认欢迎语

### 3.6 条件路由

| 路由函数 | 判断逻辑 | 分支 |
|---------|---------|------|
| `route_after_planner` | plan 是否为空 | executor / finalize |
| `route_after_executor` | 是否还有未执行步骤 | executor(循环) / replan_check |
| `route_after_replan_check` | quality 是否通过 | finalize / replanner → executor |

---

## 四、RAG 检索增强系统 (RAG_service/RAG_program.py)

### 4.1 混合检索策略

```
用户Query
   ├─→ BGE-large-zh-v1.5 (密集向量, 1024维)
   ├─→ BM25Encoder (稀疏向量, 预训练参数)
   └─→ hybrid_convex_scale(alpha=0.7) 融合
        ↓
   Pinecone 混合查询 (dotproduct 度量, dense vector type)
        ↓
   BGE-reranker-large CrossEncoder 重排序
        ↓
   返回 Top-N 重排序结果
```

### 4.2 关键技术选型
- **向量数据库**：Pinecone Serverless（aws us-east-1），dotproduct 度量
- **密集嵌入**：BAAI/bge-large-zh-v1.5（1024 维，中文优化，归一化嵌入）
- **稀疏编码**：pinecone-text BM25Encoder，预计算参数文件 ~387KB
- **重排序**：BAAI/bge-reranker-large CrossEncoder，全注意力机制让查询与文档充分交互
- **文本分割**：MarkdownHeaderTextSplitter（按一级标题分割案例）+ RecursiveCharacterTextSplitter（chunk_size=512, overlap=50）

### 4.3 数据规模
- 11 个 Markdown 法院案例文件（2014-2024 年度婚姻家庭与继承纠纷）
- 约 5030 条向量记录
- 每个案例包含：案号、案由、基本案情、案件焦点、裁判要旨、法官后语

---

## 五、工具系统 (tools/)

### 5.1 工具清单（7个工具）

| 工具名 | 分类 | 功能 | 关键细节 |
|-------|------|------|---------|
| `retrieve_legal_knowledge` | CRAG管线 | 混合检索法律案例 | 懒加载 RAG_service 单例，支持 top_k/rerank_top_n/alpha/namespace 参数调优 |
| `evaluate_case_relevance` | CRAG管线 | 三档质量评估 | correct(≥0.7) / ambiguous(0.3~0.7) / incorrect(<0.3)；≥3条高质量或可用→"充足" |
| `analyze_legal_issue` | CRAG管线 | LLM 法律分析生成 | Saul Goodman 角色设定，整合高/中相关案例+网络资料，返回 final_answer+sources |
| `get_google_search` | 外部搜索 | SerpAPI 谷歌搜索 | 最多8条结构化结果，用于检索不足时联网补充 |
| `search_memory` | 长期记忆 | 语义搜索历史记忆 | PostgreSQL+pgvector，BGE 嵌入，cosine 相似度排序 |
| `save_to_memory` | 长期记忆 | 保存事实/偏好 | 支持 summary 字段（嵌入截断512字），memory_type 分类 |
| `markdown_to_pdf` | 输出 | Markdown→PDF报告 | markdown库转HTML + pdfkit(wkhtmltopdf)，A4页面，中文字体 |

### 5.2 长期记忆系统 (tools/memory_tools.py)

- **存储**：PostgreSQL 15 + pgvector 扩展，`agent_memory` 表
- **嵌入模型**：BAAI/bge-large-zh-v1.5（1024维，与 Pinecone 共用嵌入维度）
- **表结构**：id, thread_id, memory_type, content, embedding(VECTOR 1024), metadata(JSONB), created_at
- **搜索**：`cosine similarity = 1 - (embedding <=> query_vec)`，pgvector 运算符 `<=>` 表示欧氏距离
- **设计亮点**：
  - content 存完整原文（metadata 中），summary/截断文本用于向量嵌入
  - 建表幂等（CREATE EXTENSION IF NOT EXISTS / CREATE TABLE IF NOT EXISTS）
  - 懒加载单例模式避免重复初始化嵌入模型和数据库连接

---

## 六、数据模型体系 (state.py)

三层 Pydantic v2 模型，统一全项目数据格式：

```
A. 工具返回层
   ├── RetrievedDocument: rank, id, rerank_score, hybrid_score, year, case_number, case_cause, chunk_text
   └── EvaluationResult: total, correct_count, ambiguous_count, incorrect_count, quality_verdict, 三档分类列表

B. 计划执行层
   ├── PlanStep: step_id, description, tool_name, status(pending/doing/done/failed), retry_count
   └── ToolCallRecord: step_id, tool_name, tool_input, output, timestamp

C. 顶层 AgentState
   ├── 会话标识: session_id, user_id
   ├── 请求上下文: query, is_pdf_output, messages
   ├── 计划执行: plan, current_step_index, replan_needed, replan_reason
   ├── 输出: final_answer, reasoning(思考链)
   ├── 管线数据: rag_documents, evaluation, web_search_results, crag_context
   ├── 长期记忆: memory_results, memory_update
   └── 流程控制: should_continue, error
```

---

## 七、Web 服务层 (FastAPI/)

### 7.1 API 端点

| 端点 | 方法 | 功能 |
|------|------|------|
| `/ask` | POST | 同步问答，返回完整 JSON 响应 |
| `/ask/stream` | POST | SSE 流式问答，实时推送规划进度/工具调用/最终回答 |
| `/ask/pdf` | POST | 生成 PDF 法律报告并返回文件下载 |
| `/tools` | GET | 列出所有可用工具及描述 |
| `/home` | GET | 健康检查 + 服务信息 |

### 7.2 请求/响应模型 (model.py)
- `QueryRequest`：query (1-5000字符) + session_id (可选，支持多轮对话)
- `QueryResponse`：query + session_id + final_answer + sources + tool_calls + reasoning

### 7.3 会话管理 (utils.py)
- 自动生成/复用 session_id
- 使用 LangGraph `MemorySaver` 检查点机制实现线程级会话隔离
- 通过 `graph.ainvoke()` 异步调用，支持并发请求

### 7.4 结构化日志系统 (logging.py)

**五类日志**，使用 contextvars 实现 session_id 全链路透传：

| Logger | 输出目标 | 级别 | 用途 |
|--------|---------|------|------|
| agent_flow | 文件(轮转) + 控制台 | INFO+ | 每次请求的流程摘要 |
| agent_debug | 控制台 | DEBUG | 节点级详细执行链路 |
| tool | 控制台 | DEBUG | 工具调用参数/返回值摘要 |
| rag | 控制台 | DEBUG | RAG 检索各环节耗时 |
| system | 文件(轮转) + 控制台 | INFO+ | 启动/关闭/异常 |

**日志格式**：
- 控制台：`15:37:22 | INFO  | sess_1234 | agent_flow | Message | summary | detail | result`（带 ANSI 颜色）
- 文件：`2026-05-18 10:37:22 | INFO  | sess_1234 | agent_flow | Message`

**便捷包装器**：`flow.info()`, `debug.debug()`, `tool.info()`, `rag.debug()`, `system.warning()` 等，支持 `summary`/`detail`/`result` 额外参数。

---

## 八、角色设定与提示词工程

### 8.1 法律分析角色
- **当前**：Saul Goodman（《风骚律师》）风格 — 市侩幽默、江湖气、攻击性辩护策略
- **备选**（注释中）：Kim Wexler 风格 — 专业沉稳、极度务实、给当事人安全感
- **简单问答备选**（legacy node 中）：Hermione Granger 风格 — 博学学霸

### 8.2 Prompt 模板设计
- `PLANNER_SYSTEM`：任务规划师角色，输出 JSON plan + reasoning，含计划原则（法律问题 CRAG 管线、闲聊跳过、PDF 按需触发）
- `EXECUTOR_PROMPT`：纯执行器角色，只调用指定工具，不推理，从上下文提取参数
- `REPLAN_CHECK_PROMPT`：质量审核员角色，5条判断标准，输出 needs_replan + reason
- `REPLANNER_SYSTEM`：补充步骤生成，只输出新增步骤，不重复已完成步骤
- `LEGAL_ANALYSIS_PROMPT`：Saul Goodman 完整人设（Profile / Tone / Domain Expertise / Workflow）

---

## 九、技术栈清单

| 层级 | 技术 | 版本 | 作用 |
|------|------|------|------|
| **Agent 框架** | LangGraph | ≥0.2.0 | StateGraph 状态图编排、MemorySaver 会话检查点 |
| **LLM 基座** | langchain-openai (DeepSeek) | ≥0.2.0 | ChatOpenAI 兼容接口调用 DeepSeek API |
| **Web 框架** | FastAPI + Uvicorn | ≥0.115 / ≥0.32 | REST API + SSE 流式 + CORS 中间件 |
| **数据模型** | Pydantic | ≥2.7 | 全项目统一数据模型、请求/响应校验 |
| **向量数据库** | Pinecone | ≥5.0 | Serverless 混合检索（密集+稀疏向量） |
| **嵌入模型** | BAAI/bge-large-zh-v1.5 | - | 1024维中文语义向量 |
| **重排序** | BAAI/bge-reranker-large | - | CrossEncoder 全注意力重排序 |
| **稀疏编码** | pinecone-text (BM25) | ≥0.4 | BM25 关键词匹配 |
| **长期记忆** | PostgreSQL 15 + pgvector | - | 语义记忆存储与搜索 |
| **联网搜索** | SerpAPI (google-search-results) | ≥2.4 | 法律信息联网补充 |
| **PDF 生成** | markdown + pdfkit (wkhtmltopdf) | ≥3.7 / ≥1.0 | Markdown→HTML→PDF |
| **文本分割** | langchain-text-splitters | ≥0.3 | Markdown 标题分割 + 递归字符分割 |
| **日志** | Python logging + contextvars | 标准库 | 5类结构化日志 + 会话透传 |
| **配置管理** | python-dotenv | ≥1.0 | .env 环境变量管理 |
| **调试追踪** | LangSmith | - | LLM 调用链追踪（可选） |
| **嵌入工具** | sentence-transformers | ≥3.0 | 本地运行 BGE 嵌入与重排序模型 |

---

## 十、项目亮点

### 10.1 架构亮点
1. **双 LLM 成本优化**：规划用 Pro（慢但强），执行用 Flash（快但便宜），单次查询可节省 40-60% token 成本
2. **Plan & Execute 柔性管线**：不是硬编码的线性流程，Agent 可根据问题复杂度自主决定执行路径（法律问题走完整 CRAG、闲聊直接回答）
3. **三级降级策略**：JSON 解析降级 → LLM 调用降级 → 规则判断降级，确保系统在任何异常下都能给出合理响应
4. **质量闭环**：检索→评估→补充→重规划，形成自我纠错的闭环

### 10.2 检索亮点
1. **混合检索**：密集语义匹配 + 稀疏关键词匹配，alpha 可调权重，兼顾"意思相近"和"关键词命中"
2. **CrossEncoder 重排序**：不是简单的向量距离，而是让查询和每个文档经过全注意力交互后打分，大幅提升 Top-N 精度
3. **CRAG 三档评估**：阈值可配，Agent 可根据评估结论自主决策是否需要联网补充

### 10.3 工程亮点
1. **Pydantic v2 全链路类型安全**：从工具返回到图状态到 API 响应，整个数据流都有类型约束
2. **懒加载单例模式**：RAG_service / LLM / Embedder / DB 连接全部懒加载，避免冷启动时一次性加载所有重模型
3. **5类结构化日志 + contextvars 会话透传**：无需修改任何函数签名，session_id 自动注入所有日志
4. **SSE 流式推送**：用户可实时看到规划进度和工具调用，避免长时间等待的焦虑
5. **优雅降级无处不在**：每个可能失败的点都有 fallback，生产环境友好

### 10.4 数据亮点
1. **约 5030 条向量化法律案例**：覆盖 2014-2024 年婚姻家庭与继承纠纷
2. **长期记忆系统**：跨会话保留用户偏好/事实/决策，支持个性化服务
3. **BM25 参数预计算**：避免每次启动重新拟合，启动即用

---

## 十一、数据流全景

```
用户: "离婚后彩礼能要回来吗?"
  │
  ├─ [FastAPI] 接收请求, 生成/复用 session_id
  │
  ├─ [planner] Pro LLM 分析
  │   → reasoning: ["涉及婚姻财产纠纷", "需检索彩礼返还相关判例", ...]
  │   → plan: [
  │       {step_id:1, tool: retrieve_legal_knowledge},
  │       {step_id:2, tool: evaluate_case_relevance},
  │       {step_id:3, tool: analyze_legal_issue}
  │     ]
  │
  ├─ [executor #1] Flash LLM → 调用 retrieve_legal_knowledge("离婚彩礼返还")
  │   → Pinecone 混合检索 → BGE-reranker 重排序
  │   → 返回 5 条案例 (rag_documents)
  │
  ├─ [executor #2] Flash LLM → 调用 evaluate_case_relevance(rag_documents)
  │   → 2条 correct, 1条 ambiguous, 2条 incorrect
  │   → quality_verdict: "不足,建议进行网络搜索补充" (correct<3)
  │
  ├─ [replan_check] Flash LLM 判断 → needs_replan=true
  │
  ├─ [replanner] Pro LLM → 新增步骤:
  │   {step_id:4, tool: get_google_search}
  │   {step_id:5, tool: analyze_legal_issue}
  │
  ├─ [executor #3] → 调用 get_google_search("离婚彩礼返还 最新司法解释")
  │   → SerpAPI 返回 8 条网络资料 (web_search_results)
  │
  ├─ [executor #4] Flash LLM → 调用 analyze_legal_issue(
  │     query, correct_cases, ambiguous_cases, web_results
  │   )
  │   → Saul Goodman 风格法律分析 (final_answer, sources)
  │
  ├─ [replan_check] → needs_replan=false (已有 final_answer)
  │
  └─ [finalize] → 返回 final_answer → FastAPI → JSON/SSE 响应
```

---

## 十二、已知问题与改进方向

1. **递归限制**：当前 LangGraph recursive limit = 25，极复杂查询可能触发 `GRAPH_RECURSION_LIMIT` 错误
2. **BM25 路径硬编码**：`bm25_law_params.json` 路径为绝对路径，需改为相对路径或配置化
3. **角色切换**：Saul Goodman 风格可能不适合所有用户场景，可考虑根据用户偏好动态切换 Kim/Saul 角色
4. **并发性能**：工具中的懒加载单例在多线程/多进程下可能存在竞争条件，生产环境建议使用连接池
5. **法律条文检索**：当前仅通过 Pinecone 检索案例，法律 TXT 文件（民法典等）已入库 PostgreSQL 但未在 Agent 流程中作为工具暴露
6. **评估阈值**：CORRECT_THRESHOLD(0.7)、INCORRECT_THRESHOLD(0.3)、MIN_QUALITY_DOCS(3) 等常量可能需要根据检索质量持续调优

---

## 十三、项目目录结构

```
lawApp_LangGraph/
├── LangGraph_lawApp.py          # 主入口：LangGraph Plan & Execute Agent (5节点+4条件路由)
├── state.py                     # 统一 Pydantic 数据模型 (三层模型体系)
├── requirements.txt             # Python 依赖
├── .env                         # 环境变量配置
├── sample_law.txt               # 法律文本样例
│
├── FastAPI/                     # Web 服务层
│   ├── api.py                   # FastAPI 应用 (v2.0.0), 5个端点 + CORS + 日志中间件
│   ├── model.py                 # 请求/响应 Pydantic 模型
│   ├── utils.py                 # 会话管理、图调用、响应构建、SSE工具
│   └── logging.py               # 5类结构化日志系统 + contextvars 会话透传
│
├── tools/                       # Agent 工具集
│   ├── __init__.py              # 工具注册表 (ALL_TOOLS = 7个工具)
│   ├── tools.py                 # 网络搜索(SerpAPI) + PDF生成(markdown+pdfkit)
│   ├── rag_tools.py             # CRAG 管线工具 (检索/评估/分析) + Saul Goodman 提示词
│   ├── memory_tools.py          # 长期记忆工具 (PostgreSQL+pgvector+BGE)
│   └── _prompt_temp.txt         # 法律分析提示词模板(备选角色设定)
│
├── RAG_service/                 # RAG 向量检索服务
│   ├── RAG_program.py           # RAG_service 类: 混合检索 + BM25 + CrossEncoder 重排序
│   ├── bm25_law_params.json     # 预计算 BM25 参数 (~387KB)
│   └── RAG_Service_Test.ipynb   # RAG 测试笔记本
│
├── node/                        # 备用节点工厂(legacy 实现)
│   ├── langgraph_nodes.py       # 节点工厂函数 (替代架构)
│   └── nodes_test.ipynb         # 节点测试笔记本
│
├── Documents/                   # 法律文档数据
│   ├── LawDocument/             # 7个相关法律 TXT 文件 (民法典、反家暴法等)
│   └── MarkDownFiles/           # 11个案例 MD 文件 (2014-2024年度)
│
├── logs/                        # 运行时日志
│   ├── agent_flow.log           # 流程摘要日志 (轮转)
│   └── system.log               # 系统日志 (轮转)
│
├── pdf_outputs/                 # 生成的 PDF 报告
└── PGtest.ipynb                 # PostgreSQL 数据入库笔记本
```

---

## 十四、部署与运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量 (.env)
DEEPSEEK_API_KEY=xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
PINECONE_API_KEY=xxx
PINECONE_INDEX_NAME=pinecone-law-agent
SERPAPI_API_KEY=xxx
DB_NAME=Law_app DB_USER=xxx DB_PASSWORD=xxx DB_HOST=localhost DB_PORT=5433

# 3. 启动 FastAPI 服务
cd lawApp_LangGraph/FastAPI
python api.py
# 或
uvicorn lawApp_LangGraph.FastAPI.api:app --host 0.0.0.0 --port 8000 --reload
```

---

*文档生成时间：2026-05-20 | 项目版本：v2.0.0*
