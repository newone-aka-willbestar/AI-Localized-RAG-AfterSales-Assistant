"""
审计日志模块。

职责：
- 每次 /ask 请求完成后，记录一条结构化 JSON 日志
- 字段：时间戳、session_id、问题、意图、耗时、来源数、回答长度、是否出错
- 同时写入：rotating 日志文件（按大小滚动）+ 标准 logger（供控制台/Docker 日志聚合）

为什么要审计日志？
- 方便离线分析：哪类问题最多、哪个意图耗时最长、哪个 session 最活跃
- 生产排障：用 session_id + timestamp 快速定位某次出错的请求
- 性能基线：持续收集 latency，发现 P99 劣化时及时告警

为什么用 RotatingFileHandler？
- 防止日志文件无限增长撑爆磁盘
- 默认保留最近 10 个文件 × 5MB = 50MB 上限，够几个月的运营日志

线程安全：logging 模块的 Handler 内部已有 Lock，直接复用，不额外加锁。
"""
import json
import logging
import logging.handlers
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 审计日志文件路径（可通过环境变量覆盖）
_AUDIT_LOG_PATH = os.environ.get("AUDIT_LOG_PATH", "./rag_cache/audit.log")
_MAX_BYTES = 5 * 1024 * 1024   # 5 MB per file
_BACKUP_COUNT = 10              # 保留最近 10 个滚动文件

# 懒加载：只在第一次记录时初始化文件 handler，避免启动时创建空文件
_audit_logger: Optional[logging.Logger] = None


def _get_audit_logger() -> logging.Logger:
    """
    获取（或初始化）审计专用 Logger。

    审计 Logger 与应用 Logger 隔离：
    - 独立名称 "audit"，不随根 logger 级别变化
    - FileHandler 只写 JSON，不带时间戳前缀（时间在 JSON 字段里）
    - 同时保留一个 StreamHandler 供 Docker / CloudWatch 日志聚合
    """
    global _audit_logger
    if _audit_logger is not None:
        return _audit_logger

    audit_logger = logging.getLogger("audit")
    audit_logger.setLevel(logging.INFO)
    audit_logger.propagate = False  # 不向根 logger 传播，避免重复输出

    # 文件 handler（滚动）
    log_path = Path(_AUDIT_LOG_PATH)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        str(log_path),
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    # 只输出消息本身（JSON 字符串），不加 levelname / asctime 前缀
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    audit_logger.addHandler(file_handler)

    # 控制台 handler（给 Docker logs / 终端查看）
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("[AUDIT] %(message)s"))
    audit_logger.addHandler(stream_handler)

    _audit_logger = audit_logger
    logger.info(f"审计日志初始化完成，写入路径: {log_path.resolve()}")
    return _audit_logger


def record(
    *,
    question: str,
    intent: str,
    latency_ms: float,
    answer_length: int,
    source_count: int,
    session_id: Optional[str] = None,
    error: Optional[str] = None,
    provider: str = "",
    has_history: bool = False,
) -> None:
    """
    记录一条问答审计日志。

    Args:
        question:      用户原始问题（截断到 200 字，避免日志过大）
        intent:        意图分类结果（knowledge_query / chitchat / ...）
        latency_ms:    端到端耗时（毫秒），从收到请求到返回答案
        answer_length: 回答字符数，间接反映答案质量
        source_count:  检索到的原文来源数（chitchat 为 0）
        session_id:    会话 ID，可关联同一对话的多条记录
        error:         出错时的异常信息，正常请求传 None
        provider:      LLM 提供商（ollama / deepseek）
        has_history:   本轮是否注入了历史上下文
    """
    entry = {
        "ts":            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "session_id":    session_id or "",
        "question":      question[:200],
        "intent":        intent,
        "latency_ms":    round(latency_ms, 1),
        "answer_length": answer_length,
        "source_count":  source_count,
        "has_history":   has_history,
        "provider":      provider,
        "ok":            error is None,
        "error":         error or "",
    }
    try:
        _get_audit_logger().info(json.dumps(entry, ensure_ascii=False))
    except Exception as e:
        # 审计日志失败绝不能影响主链路，静默记录到应用日志
        logger.warning(f"审计日志写入失败（不影响功能）: {e}")


class Timer:
    """
    上下文管理器，用于测量代码块耗时（毫秒）。

    用法：
        with Timer() as t:
            result = ask_graph.invoke(state)
        audit_log.record(..., latency_ms=t.elapsed_ms)
    """
    def __init__(self):
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
