"""
审计日志模块测试。

策略：
- 用临时目录隔离每个测试的日志文件，互不干扰
- 验证 JSON 结构完整性和字段语义
- 验证 Timer 计时精度
- 验证异常不影响主链路（fire-and-forget 保证）
- 验证日志文件实际写入磁盘
"""
import json
import os
import time
import importlib
import pytest
from pathlib import Path
from unittest.mock import patch


# ==========================================
# 测试夹具
# ==========================================

@pytest.fixture(autouse=True)
def reset_audit_logger():
    """
    每个测试前重置 _audit_logger 全局变量，
    确保下一个测试重新初始化（防止路径污染）。
    """
    import src.audit_log as al
    original = al._audit_logger
    al._audit_logger = None
    yield
    # 关闭 handler，释放文件句柄，再还原
    if al._audit_logger is not None:
        for h in al._audit_logger.handlers[:]:
            h.close()
            al._audit_logger.removeHandler(h)
    al._audit_logger = original


@pytest.fixture
def log_path(tmp_path):
    """每个测试用独立的临时日志文件"""
    return str(tmp_path / "test_audit.log")


# ==========================================
# Timer
# ==========================================

class TestTimer:

    def test_elapsed_ms_positive(self):
        from src.audit_log import Timer
        with Timer() as t:
            time.sleep(0.05)
        assert t.elapsed_ms > 0

    def test_elapsed_ms_unit_is_milliseconds(self):
        """sleep 50ms，elapsed 应在 40~500ms 之间（CI 机器可能慢）"""
        from src.audit_log import Timer
        with Timer() as t:
            time.sleep(0.05)
        assert 40 <= t.elapsed_ms <= 500

    def test_elapsed_zero_before_exit(self):
        """未退出上下文时 elapsed_ms 为 0"""
        from src.audit_log import Timer
        t = Timer()
        assert t.elapsed_ms == 0.0

    def test_context_manager_returns_self(self):
        from src.audit_log import Timer
        t = Timer()
        result = t.__enter__()
        assert result is t
        t.__exit__(None, None, None)

    def test_short_block_measurable(self):
        """极短代码块也能正常计时（不返回负值）"""
        from src.audit_log import Timer
        with Timer() as t:
            _ = 1 + 1
        assert t.elapsed_ms >= 0


# ==========================================
# record() — JSON 结构
# ==========================================

class TestRecordStructure:

    def test_record_writes_valid_json(self, log_path):
        import src.audit_log as al
        with patch.dict(os.environ, {"AUDIT_LOG_PATH": log_path}):
            al._AUDIT_LOG_PATH = log_path
            al.record(
                question="E05错误怎么处理",
                intent="knowledge_query",
                latency_ms=123.4,
                answer_length=256,
                source_count=3,
            )

        with open(log_path, encoding="utf-8") as f:
            line = f.readline().strip()
        entry = json.loads(line)   # 不报错 = 合法 JSON
        assert isinstance(entry, dict)

    def test_record_contains_required_fields(self, log_path):
        import src.audit_log as al
        al._AUDIT_LOG_PATH = log_path
        al.record(
            question="保修期多久",
            intent="knowledge_query",
            latency_ms=200.0,
            answer_length=100,
            source_count=2,
        )

        with open(log_path, encoding="utf-8") as f:
            entry = json.loads(f.readline())

        required = {"ts", "session_id", "question", "intent",
                    "latency_ms", "answer_length", "source_count",
                    "has_history", "provider", "ok", "error"}
        assert required.issubset(entry.keys())

    def test_ok_is_true_when_no_error(self, log_path):
        import src.audit_log as al
        al._AUDIT_LOG_PATH = log_path
        al.record(question="q", intent="knowledge_query",
                  latency_ms=10, answer_length=50, source_count=1)

        with open(log_path, encoding="utf-8") as f:
            entry = json.loads(f.readline())
        assert entry["ok"] is True
        assert entry["error"] == ""

    def test_ok_is_false_when_error_provided(self, log_path):
        import src.audit_log as al
        al._AUDIT_LOG_PATH = log_path
        al.record(question="q", intent="knowledge_query",
                  latency_ms=10, answer_length=0, source_count=0,
                  error="TimeoutError: model took too long")

        with open(log_path, encoding="utf-8") as f:
            entry = json.loads(f.readline())
        assert entry["ok"] is False
        assert "TimeoutError" in entry["error"]

    def test_question_truncated_at_200_chars(self, log_path):
        import src.audit_log as al
        al._AUDIT_LOG_PATH = log_path
        long_q = "问" * 300
        al.record(question=long_q, intent="knowledge_query",
                  latency_ms=10, answer_length=50, source_count=1)

        with open(log_path, encoding="utf-8") as f:
            entry = json.loads(f.readline())
        assert len(entry["question"]) == 200

    def test_session_id_recorded(self, log_path):
        import src.audit_log as al
        al._AUDIT_LOG_PATH = log_path
        al.record(question="q", intent="chitchat",
                  latency_ms=5, answer_length=20, source_count=0,
                  session_id="test-session-123")

        with open(log_path, encoding="utf-8") as f:
            entry = json.loads(f.readline())
        assert entry["session_id"] == "test-session-123"

    def test_no_session_id_defaults_to_empty_string(self, log_path):
        import src.audit_log as al
        al._AUDIT_LOG_PATH = log_path
        al.record(question="q", intent="knowledge_query",
                  latency_ms=10, answer_length=50, source_count=1)

        with open(log_path, encoding="utf-8") as f:
            entry = json.loads(f.readline())
        assert entry["session_id"] == ""

    def test_has_history_field(self, log_path):
        import src.audit_log as al
        al._AUDIT_LOG_PATH = log_path
        al.record(question="q", intent="knowledge_query",
                  latency_ms=10, answer_length=50, source_count=1,
                  has_history=True)

        with open(log_path, encoding="utf-8") as f:
            entry = json.loads(f.readline())
        assert entry["has_history"] is True

    def test_ts_is_iso8601_format(self, log_path):
        import src.audit_log as al
        al._AUDIT_LOG_PATH = log_path
        al.record(question="q", intent="knowledge_query",
                  latency_ms=10, answer_length=50, source_count=1)

        with open(log_path, encoding="utf-8") as f:
            entry = json.loads(f.readline())
        # 格式：2026-05-10T12:00:00Z
        ts = entry["ts"]
        assert len(ts) == 20
        assert ts.endswith("Z")
        assert "T" in ts


# ==========================================
# 文件写入
# ==========================================

class TestFileWriting:

    def test_log_file_created(self, log_path):
        import src.audit_log as al
        al._AUDIT_LOG_PATH = log_path
        al.record(question="q", intent="k", latency_ms=1,
                  answer_length=1, source_count=0)
        assert Path(log_path).exists()

    def test_multiple_records_each_on_own_line(self, log_path):
        import src.audit_log as al
        al._AUDIT_LOG_PATH = log_path
        for i in range(3):
            al.record(question=f"问题{i}", intent="knowledge_query",
                      latency_ms=i * 10, answer_length=50, source_count=1)

        with open(log_path, encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        assert len(lines) == 3
        for line in lines:
            json.loads(line)  # 每行都是合法 JSON

    def test_parent_dir_auto_created(self, tmp_path):
        import src.audit_log as al
        nested_path = str(tmp_path / "deep" / "nested" / "audit.log")
        al._AUDIT_LOG_PATH = nested_path
        al.record(question="q", intent="k", latency_ms=1,
                  answer_length=1, source_count=0)
        assert Path(nested_path).exists()


# ==========================================
# 容错性（主链路不受影响）
# ==========================================

class TestFaultTolerance:

    def test_handler_error_does_not_raise(self, tmp_path):
        """日志 handler 内部出错时，record() 不向调用方抛异常"""
        import src.audit_log as al
        al._AUDIT_LOG_PATH = str(tmp_path / "audit.log")

        # 让 _get_audit_logger 返回会抛异常的 mock
        broken_logger = type("L", (), {"info": staticmethod(lambda *a: (_ for _ in ()).throw(OSError("disk full")))})()
        with patch("src.audit_log._get_audit_logger", return_value=broken_logger):
            # 不应该抛出异常
            al.record(question="q", intent="k", latency_ms=1,
                      answer_length=1, source_count=0)
