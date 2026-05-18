"""
Agent 分层结构化日志系统


四类 Logger:
    agent_flow  → 文件 (agent_flow.log)         : 每次请求的流程摘要
    agent_debug → 控制台 (DEBUG 开启时)         : 节点级详细执行链路
    tool       → 控制台                        : 工具调用及返回值摘要
    rag        → 控制台                        : RAG 检索管线各环节耗时与结果

三级日志:
    INFO     — 关键节点 (进入/完成/结果摘要)
    DEBUG    — 节点内部细节 (参数、中间状态、检索管线步骤)
    WARNING  — 降级/重试/异常但可恢复
    ERROR    — 致命错误

日志格式:
    时间 | LEVEL    | 会话ID | 模块.节点 | 概要 | 详细 | → 结果

用法:
    from lawApp_LangGraph.FastAPI.logging import (
        agent_flow, agent_debug, tool_log, rag_log, sys_log, set_session
    )

    set_session(session_id)

    agent_flow.info("流程开始", summary="用户提问", detail=f"query={query}...")
    agent_debug.debug("进入规划节点", detail=f"query={query[:80]}")
    tool_log.info("调用检索工具", result=f"返回{doc_count}条, top_score={score}")
    rag_log.debug("密集向量编码完成", detail=f"text_len={n}", result=f"elapsed={t}s")
"""

import logging
import logging.handlers
import sys
import os
import contextvars
from pathlib import Path


#  Session 上下文 (contextvars 实现全链路透传,无需改函数签名)


_session_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "session_id", default="-"
)


def set_session(session_id: str) -> None:
    """设置当前请求的 session_id,后续所有日志自动携带"""
    _session_id.set(session_id)


def get_session() -> str:
    return _session_id.get()



#  自定义结构化 Formatter


# 控制台颜色 (ANSI)
_COLORS = {
    "DEBUG": "\033[36m",  # cyan
    "INFO": "\033[32m",  # green
    "WARNING": "\033[33m",  # yellow
    "ERROR": "\033[31m",  # red
    "RESET": "\033[0m",
}

# 简洁版——给 agent_flow 文件日志用,一行一个事件
_FILE_FMT = "%(asctime)s | %(levelname)-5s | %(session)-8s | %(name)-20s | %(msg)s"

class _AgentFormatter(logging.Formatter):
    """控制台格式化器: 注入 session_id、颜色、结构化字段"""

    def __init__(self, use_color: bool = True):
        super().__init__(datefmt="%H:%M:%S")
        self.use_color = use_color and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        record.session = get_session()[:8]
        record.levelname_color = ""
        record.reset = ""
        if self.use_color:
            c = _COLORS.get(record.levelname, "")
            if c:
                record.levelname_color = c
                record.reset = _COLORS["RESET"]

        # 拼接 summary / detail / result 到 message 尾部
        extras: list[str] = []
        for attr in ("summary", "detail", "result"):
            val = getattr(record, attr, None) or ""
            if val:
                extras.append(val)
        if extras:
            record.msg = record.msg + " | " + " | ".join(extras)

        return super().format(record)


class _FileFormatter(logging.Formatter):
    """文件格式化器: 无颜色,一行一个事件"""

    def __init__(self):
        super().__init__(fmt=_FILE_FMT, datefmt="%Y-%m-%d %H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        record.session = get_session()[:8]
        # 同样拼上扩展字段
        extras: list[str] = []
        for attr in ("summary", "detail", "result"):
            val = getattr(record, attr, None) or ""
            if val:
                extras.append(val)
        if extras:
            record.msg = record.msg + " | " + " | ".join(extras)
        return super().format(record)



#  Logger 工厂

_loggers: dict[str, logging.Logger] = {}
_initialized = False


def _make_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """创建一个纯净的 logger,由 setup_logging() 统一装配 handler"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    _loggers[name] = logger
    return logger


def setup_logging(
    log_dir: str = "./logs",
    console_level: int = logging.INFO,
    file_level: int = logging.INFO,
) -> None:
    """初始化所有日志 handler —— 在 FastAPI lifespan 中调用一次

    Args:
        log_dir: 日志文件目录
        console_level: 控制台最低输出级别 (DEBUG / INFO / WARNING / ERROR)
        file_level:   文件最低输出级别
    """
    global _initialized
    if _initialized:
        return

    os.makedirs(log_dir, exist_ok=True)

    # ---- 控制台 handler (agent_debug / tool / rag / sys 共享) ----
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(_AgentFormatter(use_color=True))

    # ---- 文件 handler (agent_flow 专用) ----
    file_handler = logging.handlers.RotatingFileHandler(
        Path(log_dir) / "agent_flow.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(_FileFormatter())

    # 为每个 logger 挂载对应 handler
    for name, logger in _loggers.items():
        logger.handlers.clear()
        if name  "agent_flow":
            logger.addHandler(file_handler)
            # agent_flow 也输出到控制台 (INFO 及以上)
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(max(console_level, logging.INFO))
            ch.setFormatter(_AgentFormatter(use_color=True))
            logger.addHandler(ch)
        else:
            logger.addHandler(console_handler)

    _initialized = True


# ---- 实例化四类 Logger ----

# 1. Agent 流程日志 —— 文件 (每次请求的完整链路摘要)
agent_flow = _make_logger("agent_flow", logging.DEBUG)

# 2. Agent 调试日志 —— 控制台 (节点级详细执行步骤, 仅 DEBUG 级别)
agent_debug = _make_logger("agent_debug", logging.DEBUG)

# 3. 工具调用日志 —— 控制台 (工具名 / 参数概要 / 返回值概要)
tool_log = _make_logger("tool", logging.DEBUG)

# 4. RAG 检索管线日志 —— 控制台 (embedding / Pinecone / rerank 各环节)
rag_log = _make_logger("rag", logging.DEBUG)

# 5. 系统日志 —— 控制台 + 文件 (启动/关闭/异常)
sys_log = _make_logger("system", logging.DEBUG)


# 
#  便捷日志函数 —— 封装 extra 字段,使调用方只需一行代码
# 

class AgentFlowLogger:
    """Agent 流程日志 —— 写入 agent_flow.log, 同时控制台输出 INFO 以上"""

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def info(self, msg: str, summary: str = "", detail: str = "", result: str = ""):
        self._logger.info(
            msg, extra={"summary": summary, "detail": detail, "result": result}
        )

    def warning(
        self, msg: str, summary: str = "", detail: str = "", result: str = ""
    ):
        self._logger.warning(
            msg, extra={"summary": summary, "detail": detail, "result": result}
        )

    def error(self, msg: str, summary: str = "", detail: str = "", result: str = ""):
        self._logger.error(
            msg, extra={"summary": summary, "detail": detail, "result": result}
        )


class AgentDebugLogger:
    """Agent 调试日志 —— 控制台输出, DEBUG 级别显示完整执行链路"""

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def debug(self, msg: str, detail: str = "", result: str = ""):
        self._logger.debug(msg, extra={"detail": detail, "result": result})

    def info(self, msg: str, detail: str = "", result: str = ""):
        self._logger.info(msg, extra={"detail": detail, "result": result})

    def warning(self, msg: str, detail: str = "", result: str = ""):
        self._logger.warning(msg, extra={"detail": detail, "result": result})

    def error(self, msg: str, detail: str = "", result: str = ""):
        self._logger.error(msg, extra={"detail": detail, "result": result})


class ToolLogger:
    """工具调用日志 —— 控制台输出, 记录工具名 / 参数概要 / 返回值"""

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def debug(self, msg: str, detail: str = "", result: str = ""):
        self._logger.debug(msg, extra={"detail": detail, "result": result})

    def info(self, msg: str, detail: str = "", result: str = ""):
        self._logger.info(msg, extra={"detail": detail, "result": result})

    def warning(self, msg: str, detail: str = "", result: str = ""):
        self._logger.warning(msg, extra={"detail": detail, "result": result})

    def error(self, msg: str, detail: str = "", result: str = ""):
        self._logger.error(msg, extra={"detail": detail, "result": result})


class RAGLogger:
    """RAG 检索管线日志 —— 控制台输出, DEBUG 级别显示各环节耗时"""

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def debug(self, msg: str, detail: str = "", result: str = ""):
        self._logger.debug(msg, extra={"detail": detail, "result": result})

    def info(self, msg: str, detail: str = "", result: str = ""):
        self._logger.info(msg, extra={"detail": detail, "result": result})

    def warning(self, msg: str, detail: str = "", result: str = ""):
        self._logger.warning(msg, extra={"detail": detail, "result": result})

    def error(self, msg: str, detail: str = "", result: str = ""):
        self._logger.error(msg, extra={"detail": detail, "result": result})


class SystemLogger:
    """系统日志 —— 启动/关闭/异常, 控制台 + 文件"""

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def info(self, msg: str, detail: str = "", result: str = ""):
        self._logger.info(msg, extra={"detail": detail, "result": result})

    def debug(self, msg: str, detail: str = "", result: str = ""):
        self._logger.debug(msg, extra={"detail": detail, "result": result})

    def warning(self, msg: str, detail: str = "", result: str = ""):
        self._logger.warning(msg, extra={"detail": detail, "result": result})

    def error(self, msg: str, detail: str = "", result: str = ""):
        self._logger.error(msg, extra={"detail": detail, "result": result})


# 包装后的便捷实例
flow = AgentFlowLogger(agent_flow)
debug = AgentDebugLogger(agent_debug)
tool = ToolLogger(tool_log)
rag = RAGLogger(rag_log)
system = SystemLogger(sys_log)
