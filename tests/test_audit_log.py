"""
审计日志模块测试。

策略：
- 将日志输出重定向到临时文件，不污染真实日志目录
- 验证：JSON 格式、字段完整性、Timer 计时、异常容错
"""
import json
import os
import time
import pytest
import tempfile

import src.audit_log as audit_log_module
from src.audit_log import Timer


# ==========================================
# 测试夹具：重定向审计日志到临时文件
# ==========================================

@pytest.fixture(autouse=True)
def reset_audit_logger(tmp_path, monkeypatch):
    """每个测试前重置全局 audit logger，使用临时路径"""
    monkeypatch.setattr(audit_log_module, "_audit_logger", None)
    monkeypatch.setattr(audit_log_module, "_AUDIT_LOG_PATH",
                        str(tmp_path / "test_audit.log"))
    yield
    # 关闭 handler 防止文件句柄泄漏
    if audit_log_module._audit_logger:
        for h in audit_log_module._audit_logger.handlers[:]:
            h.close()
            audit_log_module._audit_logger.removeHandler(h)
        audit_log_module._audit_logger = None


def read_last_log_entry(tmp_path) -> dict:
    log_file = tmp_path / "test_audit.log"
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    return json.loads(lines[-1])


# ==========================================
# record() 字段验证
# ==========================================

class TestRecord:

    def test_log_file_created(self, tmp_path):
        audit_log_module.record(
            question="测试问题", intent="rag",
            latency_ms=120.5, answer_length=200,
            source_count=3,
        )
        assert (tmp_path / "test_audit.log").exists()

    def test_all_required_fields_present(self, tmp_path):
        audit_log_module.record(
            question="什么是 RAG？", intent="knowledge_query",
            latency_ms=250.0, answer_length=150, source_count=4,
            session_id="sess_001", provider="deepseek",
        )
        entry = read_last_log_entry(tmp_path)
        required = {"ts", "session_id", "question", "intent",
                    "latency_ms", "answer_length", "source_count",
                    "has_history", "provider", "ok", "error"}
        assert required.issubset(entry.keys())

    def test_ok_true_when_no_error(self, tmp_path):
        audit_log_module.record(
            question="q", intent="rag",
            latency_ms=100.0, answer_length=50, source_count=2,
        )
        entry = read_last_log_entry(tmp_path)
        assert entry["ok"] is True
        assert entry["error"] == ""

    def test_ok_false_when_error_provided(self, tmp_path):
        audit_log_module.record(
            question="q", intent="rag",
            latency_ms=100.0, answer_length=0, source_count=0,
            error="连接超时",
        )
        entry = read_last_log_entry(tmp_path)
        assert entry["ok"] is False
        assert "超时" in entry["error"]

    def test_question_truncated_to_200_chars(self, tmp_path):
        long_q = "问" * 300
        audit_log_module.record(
            question=long_q, intent="rag",
            latency_ms=100.0, answer_length=10, source_count=1,
        )
        entry = read_last_log_entry(tmp_path)
        assert len(entry["question"]) <= 200

    def test_session_id_recorded(self, tmp_path):
        audit_log_module.record(
            question="q", intent="chitchat",
            latency_ms=80.0, answer_length=30, source_count=0,
            session_id="user_abc",
        )
        entry = read_last_log_entry(tmp_path)
        assert entry["session_id"] == "user_abc"

    def test_has_history_flag_recorded(self, tmp_path):
        audit_log_module.record(
            question="q", intent="rag",
            latency_ms=100.0, answer_length=50, source_count=2,
            has_history=True,
        )
        entry = read_last_log_entry(tmp_path)
        assert entry["has_history"] is True

    def test_output_is_valid_json(self, tmp_path):
        audit_log_module.record(
            question="测试", intent="rag",
            latency_ms=99.9, answer_length=10, source_count=1,
        )
        log_file = tmp_path / "test_audit.log"
        for line in log_file.read_text(encoding="utf-8").strip().splitlines():
            # 每行都应该能被 json.loads 解析（跳过 [AUDIT] 前缀的控制台输出）
            line = line.replace("[AUDIT] ", "")
            json.loads(line)  # 不应抛出异常


# ==========================================
# Timer 计时器
# ==========================================

class TestTimer:

    def test_elapsed_ms_positive(self):
        with Timer() as t:
            time.sleep(0.05)
        assert t.elapsed_ms > 0

    def test_elapsed_ms_approximately_correct(self):
        with Timer() as t:
            time.sleep(0.1)
        # 100ms ± 50ms（留足余量应对 CI 环境抖动）
        assert 50 < t.elapsed_ms < 300

    def test_timer_reusable(self):
        t = Timer()
        with t:
            time.sleep(0.02)
        first = t.elapsed_ms
        with t:
            time.sleep(0.05)
        second = t.elapsed_ms
        assert second > first

    def test_elapsed_ms_default_zero(self):
        t = Timer()
        assert t.elapsed_ms == 0.0
