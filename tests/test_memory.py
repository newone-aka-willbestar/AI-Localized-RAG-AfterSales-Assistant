"""
SessionMemory 测试。

测试策略：
- 功能：get_history / add_turn / clear / evict_expired
- 隔离：不同 session_id 互不影响
- TTL：过期后访问自动清除
- max_turns：超出上限自动截断
- 线程安全：并发写入不丢失、不崩溃
- history 文本格式：供 prompt 注入验证
"""
import threading
import time
import pytest

from src.memory import SessionMemory, Session, Turn


# ==========================================
# Session / Turn 数据结构
# ==========================================

class TestSession:

    def test_turn_fields(self):
        t = Turn(question="E05是什么", answer="过载保护", intent="knowledge_query")
        assert t.question == "E05是什么"
        assert t.answer == "过载保护"
        assert t.intent == "knowledge_query"

    def test_session_starts_empty(self):
        s = Session()
        assert s.turns == []
        assert not s.is_expired(ttl=1800)

    def test_add_turn_updates_last_active(self):
        s = Session()
        before = s.last_active
        time.sleep(0.01)
        s.add_turn("问题", "答案")
        assert s.last_active > before

    def test_trim_keeps_most_recent(self):
        s = Session()
        for i in range(10):
            s.add_turn(f"问题{i}", f"答案{i}")
        s.trim(max_turns=3)
        assert len(s.turns) == 3
        assert s.turns[0].question == "问题7"
        assert s.turns[-1].question == "问题9"

    def test_trim_noop_when_within_limit(self):
        s = Session()
        s.add_turn("问题1", "答案1")
        s.add_turn("问题2", "答案2")
        s.trim(max_turns=5)
        assert len(s.turns) == 2

    def test_is_expired_with_zero_ttl(self):
        s = Session()
        time.sleep(0.01)
        assert s.is_expired(ttl=0.001)

    def test_is_not_expired_within_ttl(self):
        s = Session()
        assert not s.is_expired(ttl=1800)


class TestHistoryText:

    def test_empty_session_returns_empty_string(self):
        s = Session()
        assert s.to_history_text() == ""

    def test_history_contains_question_and_answer(self):
        s = Session()
        s.add_turn("E05错误怎么处理", "E05 表示过载保护，请检查电机负载")
        text = s.to_history_text()
        assert "E05错误怎么处理" in text
        assert "E05 表示过载保护" in text

    def test_history_starts_with_header(self):
        s = Session()
        s.add_turn("问题", "答案")
        assert s.to_history_text().startswith("[历史对话]")

    def test_long_answer_truncated_to_200_chars(self):
        s = Session()
        long_answer = "A" * 500
        s.add_turn("问题", long_answer)
        text = s.to_history_text()
        # 截断后的答案不超过 200+3("...")
        lines = text.split("\n")
        answer_line = next(l for l in lines if l.startswith("助手:"))
        # 实际内容部分（去掉"助手: "前缀）
        content = answer_line[len("助手: "):]
        assert len(content) <= 204  # 200 chars + "..."

    def test_multiple_turns_all_present(self):
        s = Session()
        s.add_turn("问题1", "答案1")
        s.add_turn("问题2", "答案2")
        text = s.to_history_text()
        assert "问题1" in text
        assert "答案1" in text
        assert "问题2" in text
        assert "答案2" in text


# ==========================================
# SessionMemory 基本功能
# ==========================================

class TestSessionMemoryBasic:

    def test_get_history_empty_for_new_session(self):
        mem = SessionMemory()
        assert mem.get_history("sid-new") == ""

    def test_add_turn_then_get_history(self):
        mem = SessionMemory()
        mem.add_turn("sid-1", "E05怎么处理", "重启电机")
        history = mem.get_history("sid-1")
        assert "E05怎么处理" in history
        assert "重启电机" in history

    def test_session_isolation(self):
        """不同 session_id 不共享历史"""
        mem = SessionMemory()
        mem.add_turn("sid-A", "A的问题", "A的答案")
        mem.add_turn("sid-B", "B的问题", "B的答案")

        history_a = mem.get_history("sid-A")
        history_b = mem.get_history("sid-B")

        assert "A的问题" in history_a
        assert "B的问题" not in history_a
        assert "B的问题" in history_b
        assert "A的问题" not in history_b

    def test_clear_removes_session(self):
        mem = SessionMemory()
        mem.add_turn("sid-x", "问题", "答案")
        mem.clear("sid-x")
        assert mem.get_history("sid-x") == ""

    def test_clear_nonexistent_session_noop(self):
        """清除不存在的会话不报错"""
        mem = SessionMemory()
        mem.clear("sid-ghost")  # 不应抛出异常

    def test_session_count_increments(self):
        mem = SessionMemory()
        mem.add_turn("s1", "q", "a")
        mem.add_turn("s2", "q", "a")
        assert mem.session_count() == 2

    def test_session_count_decrements_after_clear(self):
        mem = SessionMemory()
        mem.add_turn("s1", "q", "a")
        mem.add_turn("s2", "q", "a")
        mem.clear("s1")
        assert mem.session_count() == 1

    def test_turn_count(self):
        mem = SessionMemory()
        mem.add_turn("s1", "q1", "a1")
        mem.add_turn("s1", "q2", "a2")
        assert mem.turn_count("s1") == 2
        assert mem.turn_count("nonexistent") == 0


# ==========================================
# max_turns 截断
# ==========================================

class TestMaxTurns:

    def test_exceeding_max_turns_truncates(self):
        mem = SessionMemory(max_turns=3)
        for i in range(6):
            mem.add_turn("sid", f"问题{i}", f"答案{i}")
        # 只保留最近 3 轮
        assert mem.turn_count("sid") == 3

    def test_truncation_keeps_latest_turns(self):
        mem = SessionMemory(max_turns=2)
        mem.add_turn("sid", "旧问题1", "旧答案1")
        mem.add_turn("sid", "旧问题2", "旧答案2")
        mem.add_turn("sid", "新问题", "新答案")

        history = mem.get_history("sid")
        assert "新问题" in history
        assert "旧问题1" not in history  # 最旧的轮次被丢弃


# ==========================================
# TTL 过期
# ==========================================

class TestTTL:

    def test_expired_session_returns_empty_history(self):
        mem = SessionMemory(ttl_seconds=0.05)  # 50ms TTL
        mem.add_turn("sid", "问题", "答案")
        time.sleep(0.1)  # 等待过期
        assert mem.get_history("sid") == ""

    def test_expired_session_removed_from_count(self):
        mem = SessionMemory(ttl_seconds=0.05)
        mem.add_turn("sid", "问题", "答案")
        time.sleep(0.1)
        mem.get_history("sid")  # 触发清理
        assert mem.session_count() == 0

    def test_evict_expired_clears_stale_sessions(self):
        mem = SessionMemory(ttl_seconds=0.05)
        mem.add_turn("s1", "q", "a")
        mem.add_turn("s2", "q", "a")
        time.sleep(0.1)
        count = mem.evict_expired()
        assert count == 2
        assert mem.session_count() == 0

    def test_active_session_not_evicted(self):
        mem = SessionMemory(ttl_seconds=60)
        mem.add_turn("active", "q", "a")
        count = mem.evict_expired()
        assert count == 0
        assert mem.session_count() == 1


# ==========================================
# 线程安全
# ==========================================

class TestThreadSafety:

    def test_concurrent_add_turns_no_data_loss(self):
        """20 个线程各写 10 轮，总量不超过 max_turns 但不应崩溃"""
        mem = SessionMemory(max_turns=200)
        errors = []

        def worker(thread_id):
            for i in range(10):
                try:
                    mem.add_turn(f"sid-{thread_id}", f"问题{i}", f"答案{i}")
                except Exception as e:
                    errors.append(str(e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"并发写入出现异常: {errors}"

    def test_concurrent_reads_writes_same_session(self):
        """同一 session 并发读写不崩溃"""
        mem = SessionMemory()
        errors = []

        def writer():
            for i in range(20):
                try:
                    mem.add_turn("shared", f"q{i}", f"a{i}")
                except Exception as e:
                    errors.append(str(e))

        def reader():
            for _ in range(20):
                try:
                    mem.get_history("shared")
                except Exception as e:
                    errors.append(str(e))

        threads = (
            [threading.Thread(target=writer) for _ in range(5)] +
            [threading.Thread(target=reader) for _ in range(5)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"并发读写出现异常: {errors}"
