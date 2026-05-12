"""
SessionMemory 测试。

策略：
- 不依赖任何外部服务，纯内存操作
- 验证：历史格式、TTL 过期、max_turns 截断、线程安全
"""
import time
import threading
import pytest

from src.memory import SessionMemory


# ==========================================
# 基本读写
# ==========================================

class TestBasicReadWrite:

    def test_new_session_returns_empty_history(self):
        mem = SessionMemory()
        assert mem.get_history("user_001") == ""

    def test_add_turn_then_get_history_not_empty(self):
        mem = SessionMemory()
        mem.add_turn("s1", "问题", "答案")
        history = mem.get_history("s1")
        assert history != ""

    def test_history_contains_question_and_answer(self):
        mem = SessionMemory()
        mem.add_turn("s1", "RAG 是什么？", "RAG 是检索增强生成。")
        history = mem.get_history("s1")
        assert "RAG 是什么" in history
        assert "检索增强生成" in history

    def test_history_contains_header_tag(self):
        mem = SessionMemory()
        mem.add_turn("s1", "问题", "答案")
        assert "[历史对话]" in mem.get_history("s1")

    def test_multiple_turns_all_appear_in_history(self):
        mem = SessionMemory()
        mem.add_turn("s1", "第一个问题", "第一个答案")
        mem.add_turn("s1", "第二个问题", "第二个答案")
        history = mem.get_history("s1")
        assert "第一个问题" in history
        assert "第二个问题" in history

    def test_different_sessions_are_isolated(self):
        mem = SessionMemory()
        mem.add_turn("alice", "Alice 的问题", "Alice 的答案")
        mem.add_turn("bob", "Bob 的问题", "Bob 的答案")
        assert "Alice" in mem.get_history("alice")
        assert "Alice" not in mem.get_history("bob")
        assert "Bob" in mem.get_history("bob")


# ==========================================
# max_turns 截断
# ==========================================

class TestMaxTurns:

    def test_turns_trimmed_to_max(self):
        mem = SessionMemory(max_turns=3)
        for i in range(6):
            mem.add_turn("s1", f"问题{i}", f"答案{i}")
        assert mem.turn_count("s1") == 3

    def test_most_recent_turns_kept(self):
        mem = SessionMemory(max_turns=2)
        mem.add_turn("s1", "旧问题", "旧答案")
        mem.add_turn("s1", "新问题A", "新答案A")
        mem.add_turn("s1", "新问题B", "新答案B")
        history = mem.get_history("s1")
        assert "旧问题" not in history
        assert "新问题A" in history
        assert "新问题B" in history


# ==========================================
# TTL 过期
# ==========================================

class TestTTL:

    def test_expired_session_returns_empty(self):
        mem = SessionMemory(ttl_seconds=0.05)  # 50ms TTL
        mem.add_turn("s1", "问题", "答案")
        time.sleep(0.1)
        assert mem.get_history("s1") == ""

    def test_active_session_not_expired(self):
        mem = SessionMemory(ttl_seconds=60)
        mem.add_turn("s1", "问题", "答案")
        assert mem.get_history("s1") != ""

    def test_evict_expired_removes_stale_sessions(self):
        mem = SessionMemory(ttl_seconds=0.05)
        mem.add_turn("s1", "问题", "答案")
        mem.add_turn("s2", "问题", "答案")
        time.sleep(0.1)
        evicted = mem.evict_expired()
        assert evicted == 2
        assert mem.session_count() == 0


# ==========================================
# clear 和 session_count
# ==========================================

class TestClearAndCount:

    def test_clear_removes_session(self):
        mem = SessionMemory()
        mem.add_turn("s1", "问题", "答案")
        mem.clear("s1")
        assert mem.get_history("s1") == ""

    def test_clear_nonexistent_session_no_error(self):
        mem = SessionMemory()
        mem.clear("不存在的session")  # 不应抛出异常

    def test_session_count_increments(self):
        mem = SessionMemory()
        assert mem.session_count() == 0
        mem.add_turn("s1", "q", "a")
        assert mem.session_count() == 1
        mem.add_turn("s2", "q", "a")
        assert mem.session_count() == 2

    def test_turn_count_correct(self):
        mem = SessionMemory()
        mem.add_turn("s1", "q1", "a1")
        mem.add_turn("s1", "q2", "a2")
        assert mem.turn_count("s1") == 2

    def test_turn_count_nonexistent_returns_zero(self):
        mem = SessionMemory()
        assert mem.turn_count("不存在") == 0


# ==========================================
# 线程安全
# ==========================================

class TestThreadSafety:

    def test_concurrent_writes_no_exception(self):
        mem = SessionMemory()
        errors = []

        def worker(sid):
            try:
                for i in range(20):
                    mem.add_turn(sid, f"q{i}", f"a{i}")
                    mem.get_history(sid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(f"s{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"并发写入出现异常: {errors}"
