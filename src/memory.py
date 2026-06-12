"""
会话记忆管理模块。

职责：
- 跨轮次保留对话历史，让 RAG 能感知"上一句说了什么"
- 每个 session_id 独立存储，互不干扰
- TTL 自动过期（默认 30 分钟无活动即清理），防止内存泄漏

三层记忆架构
─────────────
工作记忆（Working）：当前轮次的问题和检索上下文 → 由 rag.ask() 在本轮管理
会话记忆（Session） ：本次对话的 Q&A 历史     → 本模块负责
用户画像（Profile） ：跨会话持久偏好（暂未实现）→ 后续可接 SQLite/Redis

为什么不用 LangChain Memory？
  LangChain 的 ConversationBufferMemory 是有状态对象，绑定单一 chain 实例，
  难以在多用户/多线程场景下按 session_id 隔离。
  自实现的 Dict + Lock 更透明、更容易测试，也不引入额外依赖。
"""
import logging
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# 默认保留的最大历史轮数（一问一答 = 1轮）
DEFAULT_MAX_TURNS = 5
# 默认 TTL：30 分钟不活动则自动过期（秒）
DEFAULT_TTL_SECONDS = 1800


@dataclass
class Turn:
    """一轮对话：用户问题 + 助手回答"""
    question: str
    answer: str
    intent: str = ""


@dataclass
class Session:
    """单个会话的状态"""
    turns: list[Turn] = field(default_factory=list)
    last_active: float = field(default_factory=time.time)

    def add_turn(self, question: str, answer: str, intent: str = "") -> None:
        self.turns.append(Turn(question=question, answer=answer, intent=intent))
        self.last_active = time.time()

    def trim(self, max_turns: int) -> None:
        """只保留最近 max_turns 轮"""
        if len(self.turns) > max_turns:
            self.turns = self.turns[-max_turns:]

    def is_expired(self, ttl: float) -> bool:
        return (time.time() - self.last_active) > ttl

    def to_history_text(self) -> str:
        """
        将历史轮次格式化为 prompt 可用的文本块。

        示例输出：
          [历史对话]
          用户: E05错误怎么处理?
          助手: E05 表示过载保护，请检查电机负载...
          用户: 那E06呢?
          助手: E06 表示通信超时...
        """
        if not self.turns:
            return ""
        lines = ["[历史对话]"]
        for t in self.turns:
            lines.append(f"用户: {t.question}")
            # 只取前 200 字，避免 prompt 过长
            short_answer = t.answer[:200] + "..." if len(t.answer) > 200 else t.answer
            lines.append(f"助手: {short_answer}")
        return "\n".join(lines)


class SessionMemory:
    """
    线程安全的多用户会话记忆管理器。

    典型使用（在 api.py 中）：
        memory = SessionMemory()               # 全局单例
        history = memory.get_history(sid)      # 取出历史文本，注入 prompt
        memory.add_turn(sid, q, a, intent)     # 回答完毕后记录本轮
    """

    def __init__(
        self,
        max_turns: int = DEFAULT_MAX_TURNS,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
    ):
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._max_turns = max_turns
        self._ttl = ttl_seconds

    # ── 主要 API ──────────────────────────────────────────

    def get_history(self, session_id: str) -> str:
        """
        获取 session_id 对应的历史对话文本，供注入 prompt。
        若会话不存在或已过期，返回空字符串。
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.is_expired(self._ttl):
                # 已过期则顺便清理
                self._sessions.pop(session_id, None)
                return ""
            return session.to_history_text()

    def add_turn(
        self,
        session_id: str,
        question: str,
        answer: str,
        intent: str = "",
    ) -> None:
        """记录本轮问答，更新活跃时间"""
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = Session()
            session = self._sessions[session_id]
            session.add_turn(question=question, answer=answer, intent=intent)
            session.trim(self._max_turns)

    def clear(self, session_id: str) -> None:
        """手动清除某个会话（用户主动"清空对话"时调用）"""
        with self._lock:
            self._sessions.pop(session_id, None)
        logger.debug(f"会话已清除: {session_id}")

    def evict_expired(self) -> int:
        """
        清除所有已过期会话，返回清除数量。
        可定期调用（如每小时），也可在每次 add_turn 时触发。
        生产环境建议用 APScheduler 或 BackgroundTasks 定期调用。
        """
        with self._lock:
            expired = [
                sid for sid, sess in self._sessions.items()
                if sess.is_expired(self._ttl)
            ]
            for sid in expired:
                del self._sessions[sid]

        if expired:
            logger.info(f"清理过期会话 {len(expired)} 个")
        return len(expired)

    # ── 调试 / 运维 ───────────────────────────────────────

    def session_count(self) -> int:
        """当前活跃会话数（含可能已过期但未清理的）"""
        with self._lock:
            return len(self._sessions)

    def turn_count(self, session_id: str) -> int:
        """指定会话的历史轮数，不存在返回 0"""
        with self._lock:
            session = self._sessions.get(session_id)
            return len(session.turns) if session else 0
