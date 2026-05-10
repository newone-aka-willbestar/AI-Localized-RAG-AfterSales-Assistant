"""
Badcase 记录模块。

职责：
- 用户点踩（👎）时，将问答对写入 SQLite，供后续分析和模型改进
- 提供查询接口，方便导出 Badcase 列表

为什么用 SQLite？
- 零配置，无需额外服务，文件即数据库
- 支持 SQL 查询，方便导出 CSV 或接入分析工具
- 对于 Badcase 这种低频写入场景完全够用

表结构：
  badcases (
    id          INTEGER PRIMARY KEY,
    question    TEXT,       -- 用户原始问题
    answer      TEXT,       -- 系统回答
    intent      TEXT,       -- 意图分类结果
    sources     TEXT,       -- 检索来源（JSON 字符串）
    feedback    TEXT,       -- "bad" 或 "good"
    note        TEXT,       -- 用户附加说明（可选）
    created_at  TEXT        -- ISO 8601 时间戳
  )
"""
import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class BadcaseStore:
    """
    线程安全的 Badcase SQLite 存储。

    每次写操作使用独立连接（避免跨线程共享 Connection），
    check_same_thread=False 配合外部 Lock 保证安全。
    """

    def __init__(self, db_path: str = "./rag_cache/badcases.db"):
        self._path = Path(db_path)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """建表（幂等，已存在则跳过）"""
        with self._lock:
            conn = self._connect()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS badcases (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    question   TEXT    NOT NULL,
                    answer     TEXT    NOT NULL,
                    intent     TEXT    DEFAULT '',
                    sources    TEXT    DEFAULT '[]',
                    feedback   TEXT    NOT NULL,
                    note       TEXT    DEFAULT '',
                    created_at TEXT    NOT NULL
                )
            """)
            conn.commit()
            conn.close()
        logger.info(f"Badcase 数据库就绪: {self._path}")

    def record(
        self,
        question: str,
        answer: str,
        feedback: str,           # "bad" 或 "good"
        intent: str = "",
        sources: Optional[list] = None,
        note: str = "",
    ) -> int:
        """
        写入一条反馈记录，返回记录 id。

        Args:
            question: 用户问题
            answer:   系统回答
            feedback: "bad"（点踩）或 "good"（点赞）
            intent:   意图分类标签
            sources:  检索来源列表
            note:     用户附加说明
        """
        created_at = datetime.now(timezone.utc).isoformat()
        sources_json = json.dumps(sources or [], ensure_ascii=False)

        with self._lock:
            conn = self._connect()
            cursor = conn.execute(
                """INSERT INTO badcases
                   (question, answer, intent, sources, feedback, note, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (question, answer, intent, sources_json, feedback, note, created_at),
            )
            record_id = cursor.lastrowid
            conn.commit()
            conn.close()

        logger.info(f"已记录反馈 id={record_id} feedback={feedback} intent={intent}")
        return record_id

    def list_bad(self, limit: int = 50) -> list[dict]:
        """返回最近 N 条 bad 反馈，按时间倒序"""
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM badcases WHERE feedback='bad' ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def stats(self) -> dict:
        """返回统计信息：总数、bad 数、good 数"""
        conn = self._connect()
        total = conn.execute("SELECT COUNT(*) FROM badcases").fetchone()[0]
        bad   = conn.execute("SELECT COUNT(*) FROM badcases WHERE feedback='bad'").fetchone()[0]
        good  = conn.execute("SELECT COUNT(*) FROM badcases WHERE feedback='good'").fetchone()[0]
        conn.close()
        return {"total": total, "bad": bad, "good": good}
