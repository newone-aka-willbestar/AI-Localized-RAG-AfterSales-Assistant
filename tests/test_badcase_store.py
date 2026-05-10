"""
BadcaseStore 测试。

测试策略：
- 每个测试用独立临时数据库（tmpdir），互不干扰
- 覆盖：写入、查询、统计、线程安全、边界情况
"""
import json
import threading
import pytest
from pathlib import Path

from src.badcase_store import BadcaseStore


# ==========================================
# 测试夹具
# ==========================================

@pytest.fixture
def store(tmp_path):
    """每个测试用独立 SQLite 文件，完全隔离"""
    db_path = str(tmp_path / "test_badcases.db")
    return BadcaseStore(db_path=db_path)


# ==========================================
# 初始化
# ==========================================

class TestInit:

    def test_db_file_created_on_init(self, tmp_path):
        """初始化时自动创建数据库文件"""
        db_path = str(tmp_path / "subdir" / "badcases.db")
        BadcaseStore(db_path=db_path)
        assert Path(db_path).exists()

    def test_db_directory_auto_created(self, tmp_path):
        """父目录不存在时自动创建"""
        db_path = str(tmp_path / "deep" / "nested" / "dir" / "test.db")
        BadcaseStore(db_path=db_path)
        assert Path(db_path).exists()

    def test_init_is_idempotent(self, tmp_path):
        """多次初始化同一数据库不报错（CREATE TABLE IF NOT EXISTS）"""
        db_path = str(tmp_path / "idempotent.db")
        BadcaseStore(db_path=db_path)
        BadcaseStore(db_path=db_path)  # 不应抛出异常


# ==========================================
# 写入 record()
# ==========================================

class TestRecord:

    def test_record_returns_positive_int(self, store):
        record_id = store.record(
            question="设备报警怎么处理",
            answer="请检查电源连接",
            feedback="bad",
        )
        assert isinstance(record_id, int)
        assert record_id > 0

    def test_record_ids_are_sequential(self, store):
        id1 = store.record("问题1", "答案1", "bad")
        id2 = store.record("问题2", "答案2", "good")
        assert id2 > id1

    def test_record_with_all_fields(self, store):
        record_id = store.record(
            question="如何校准传感器",
            answer="按手册第3章步骤操作",
            feedback="bad",
            intent="operation",
            sources=[{"source": "manual.pdf", "content_excerpt": "第3章..."}],
            note="回答不够详细",
        )
        items = store.list_bad(limit=1)
        assert len(items) == 1
        item = items[0]
        assert item["question"] == "如何校准传感器"
        assert item["answer"] == "按手册第3章步骤操作"
        assert item["feedback"] == "bad"
        assert item["intent"] == "operation"
        assert item["note"] == "回答不够详细"
        # sources 以 JSON 字符串存储
        sources = json.loads(item["sources"])
        assert sources[0]["source"] == "manual.pdf"

    def test_record_good_feedback(self, store):
        store.record("问题", "答案", "good")
        stats = store.stats()
        assert stats["good"] == 1
        assert stats["bad"] == 0

    def test_record_sources_defaults_to_empty_list(self, store):
        store.record("问题", "答案", "bad")
        items = store.list_bad()
        sources = json.loads(items[0]["sources"])
        assert sources == []

    def test_record_intent_defaults_to_empty_string(self, store):
        store.record("问题", "答案", "bad")
        items = store.list_bad()
        assert items[0]["intent"] == ""

    def test_created_at_is_iso8601(self, store):
        """created_at 应为 ISO 8601 格式（含时区信息）"""
        from datetime import datetime
        store.record("问题", "答案", "bad")
        items = store.list_bad()
        ts = items[0]["created_at"]
        # 能被 fromisoformat 解析 → 格式正确
        dt = datetime.fromisoformat(ts)
        assert dt.tzinfo is not None  # 有时区信息


# ==========================================
# 查询 list_bad()
# ==========================================

class TestListBad:

    def test_list_bad_returns_only_bad_records(self, store):
        store.record("好问题", "好答案", "good")
        store.record("坏问题1", "坏答案1", "bad")
        store.record("坏问题2", "坏答案2", "bad")

        items = store.list_bad()
        assert len(items) == 2
        for item in items:
            assert item["feedback"] == "bad"

    def test_list_bad_sorted_by_id_desc(self, store):
        """最新的记录排在前面"""
        store.record("问题1", "答案1", "bad")
        store.record("问题2", "答案2", "bad")
        store.record("问题3", "答案3", "bad")

        items = store.list_bad()
        ids = [item["id"] for item in items]
        assert ids == sorted(ids, reverse=True)

    def test_list_bad_limit_respected(self, store):
        for i in range(10):
            store.record(f"问题{i}", f"答案{i}", "bad")

        items = store.list_bad(limit=3)
        assert len(items) == 3

    def test_list_bad_empty_when_no_records(self, store):
        assert store.list_bad() == []

    def test_list_bad_empty_when_only_good(self, store):
        store.record("好问题", "好答案", "good")
        assert store.list_bad() == []

    def test_list_bad_returns_dicts(self, store):
        store.record("问题", "答案", "bad")
        items = store.list_bad()
        assert isinstance(items[0], dict)


# ==========================================
# 统计 stats()
# ==========================================

class TestStats:

    def test_stats_initial_state(self, store):
        stats = store.stats()
        assert stats == {"total": 0, "bad": 0, "good": 0}

    def test_stats_counts_correctly(self, store):
        store.record("问题1", "答案1", "bad")
        store.record("问题2", "答案2", "bad")
        store.record("问题3", "答案3", "good")

        stats = store.stats()
        assert stats["total"] == 3
        assert stats["bad"] == 2
        assert stats["good"] == 1

    def test_stats_total_equals_bad_plus_good(self, store):
        for i in range(5):
            store.record(f"问题{i}", f"答案{i}", "bad" if i % 2 == 0 else "good")

        stats = store.stats()
        assert stats["total"] == stats["bad"] + stats["good"]


# ==========================================
# 线程安全
# ==========================================

class TestThreadSafety:

    def test_concurrent_writes_no_data_loss(self, store):
        """并发写入时，所有记录都应成功入库"""
        n_threads = 20
        n_writes_per_thread = 5
        errors = []

        def worker(thread_id):
            for i in range(n_writes_per_thread):
                try:
                    store.record(
                        question=f"线程{thread_id}问题{i}",
                        answer=f"答案{thread_id}-{i}",
                        feedback="bad",
                    )
                except Exception as e:
                    errors.append(str(e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"并发写入出现异常: {errors}"
        stats = store.stats()
        assert stats["total"] == n_threads * n_writes_per_thread

    def test_concurrent_reads_writes(self, store):
        """读写并发不崩溃"""
        errors = []

        def writer():
            for i in range(10):
                try:
                    store.record(f"问题{i}", f"答案{i}", "bad")
                except Exception as e:
                    errors.append(str(e))

        def reader():
            for _ in range(10):
                try:
                    store.stats()
                    store.list_bad(limit=5)
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


# ==========================================
# 边界情况
# ==========================================

class TestEdgeCases:

    def test_record_with_empty_strings(self, store):
        """空字符串字段不崩溃"""
        record_id = store.record(question="", answer="", feedback="bad")
        assert record_id > 0

    def test_record_with_special_characters(self, store):
        """特殊字符（引号、换行等）不破坏 SQL"""
        record_id = store.record(
            question="设备报错：'E05' \n 如何处理？",
            answer="请联系售后，报价¥1,200",
            feedback="bad",
            note='用户说"非常不满意"',
        )
        items = store.list_bad()
        assert items[0]["question"] == "设备报错：'E05' \n 如何处理？"

    def test_record_with_unicode(self, store):
        """Unicode 字符正确存储和读取"""
        record_id = store.record(
            question="设备温度超过85°C时应如何处理",
            answer="启动冷却模式，参考§3.2",
            feedback="bad",
        )
        items = store.list_bad()
        assert "85°C" in items[0]["question"]
        assert "§3.2" in items[0]["answer"]

    def test_large_sources_json(self, store):
        """大 sources 列表正确序列化存储"""
        sources = [{"source": f"doc{i}.pdf", "content": "x" * 200} for i in range(20)]
        store.record("问题", "答案", "bad", sources=sources)
        items = store.list_bad()
        loaded = json.loads(items[0]["sources"])
        assert len(loaded) == 20
