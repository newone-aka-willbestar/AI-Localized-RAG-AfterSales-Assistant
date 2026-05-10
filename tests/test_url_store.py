"""
测试 URLStore 模块。

覆盖：
1. seen() / mark() 基本语义
2. URL 规范化（fragment、末尾斜杠、大小写）
3. remove() 删除记录
4. 持久化：写入后重新加载能读到数据
5. 线程安全（并发 mark 不丢数据）
"""
import json
import threading
import pytest
from pathlib import Path


@pytest.fixture
def tmp_store(tmp_path):
    """每个测试用独立的临时文件路径"""
    from src.url_store import URLStore
    return URLStore(store_path=str(tmp_path / "urls.json"))


class TestBasicOperations:
    def test_new_url_not_seen(self, tmp_store):
        """新建库中任何 URL 都未见过"""
        assert tmp_store.seen("https://example.com/page") is False

    def test_mark_then_seen(self, tmp_store):
        """mark 后 seen 应返回 True"""
        url = "https://example.com/page"
        tmp_store.mark(url)
        assert tmp_store.seen(url) is True

    def test_len_increases_after_mark(self, tmp_store):
        """mark 后 len 增加"""
        assert len(tmp_store) == 0
        tmp_store.mark("https://a.com")
        assert len(tmp_store) == 1
        tmp_store.mark("https://b.com")
        assert len(tmp_store) == 2

    def test_remove_existing(self, tmp_store):
        """remove 已存在的 URL，返回 True，之后 seen 为 False"""
        url = "https://example.com/page"
        tmp_store.mark(url)
        result = tmp_store.remove(url)
        assert result is True
        assert tmp_store.seen(url) is False

    def test_remove_nonexistent(self, tmp_store):
        """remove 不存在的 URL 返回 False，不崩溃"""
        result = tmp_store.remove("https://notexist.com")
        assert result is False

    def test_all_urls_returns_list(self, tmp_store):
        """all_urls 返回已标记的规范化 URL 列表"""
        tmp_store.mark("https://example.com/a")
        tmp_store.mark("https://example.com/b")
        urls = tmp_store.all_urls()
        assert len(urls) == 2
        assert all(isinstance(u, str) for u in urls)


class TestURLNormalization:
    """同一个页面的不同写法应该被识别为同一个 URL"""

    def test_fragment_ignored(self, tmp_store):
        """#锚点不同 → 视为同一 URL"""
        tmp_store.mark("https://example.com/page#section1")
        assert tmp_store.seen("https://example.com/page#section2") is True
        assert tmp_store.seen("https://example.com/page") is True

    def test_trailing_slash_normalized(self, tmp_store):
        """末尾斜杠不影响去重"""
        tmp_store.mark("https://example.com/page/")
        assert tmp_store.seen("https://example.com/page") is True

    def test_scheme_case_insensitive(self, tmp_store):
        """scheme 大小写不敏感（HTTPS == https）"""
        tmp_store.mark("HTTPS://example.com/page")
        assert tmp_store.seen("https://example.com/page") is True

    def test_different_paths_are_different(self, tmp_store):
        """不同路径是不同 URL"""
        tmp_store.mark("https://example.com/page-a")
        assert tmp_store.seen("https://example.com/page-b") is False

    def test_query_string_preserved(self, tmp_store):
        """查询参数不同 → 不同 URL"""
        tmp_store.mark("https://example.com/search?q=pump")
        assert tmp_store.seen("https://example.com/search?q=valve") is False
        assert tmp_store.seen("https://example.com/search?q=pump") is True


class TestPersistence:
    def test_data_survives_reload(self, tmp_path):
        """mark 后新建 URLStore 实例，数据仍然存在"""
        from src.url_store import URLStore
        store_path = str(tmp_path / "urls.json")
        url = "https://example.com/page"

        store1 = URLStore(store_path=store_path)
        store1.mark(url)

        store2 = URLStore(store_path=store_path)  # 重新加载
        assert store2.seen(url) is True
        assert len(store2) == 1

    def test_json_file_is_human_readable(self, tmp_path):
        """持久化文件应该是合法 JSON，包含 url 和 crawled_at 字段"""
        from src.url_store import URLStore
        store_path = str(tmp_path / "urls.json")
        store = URLStore(store_path=store_path)
        store.mark("https://example.com")

        with open(store_path, encoding="utf-8") as f:
            data = json.load(f)

        # 文件应有一条记录
        assert len(data) == 1
        record = list(data.values())[0]
        assert "url" in record
        assert "crawled_at" in record


class TestThreadSafety:
    def test_concurrent_marks_no_data_loss(self, tmp_path):
        """10 个线程并发 mark，最终记录数应该等于不重复 URL 的数量"""
        from src.url_store import URLStore
        store = URLStore(store_path=str(tmp_path / "urls.json"))

        urls = [f"https://example.com/page-{i}" for i in range(10)]
        threads = [threading.Thread(target=store.mark, args=(u,)) for u in urls]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(store) == 10
