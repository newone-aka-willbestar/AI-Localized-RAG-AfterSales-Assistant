"""
测试 WebScraper 模块。

原则：不发真实 HTTP 请求，所有网络调用全部 mock。
覆盖：
1. 正常抓取 → 返回切分后的 Document 列表
2. URL 去重：已见过的 URL 返回空列表
3. force=True 强制重抓
4. trafilatura 失败 → 降级到 requests 方案
5. 两种方案都失败 → 抛出 WebScraperError
6. URL 格式校验（非 http/https 协议）
7. 语言检测：中文不翻译，英文触发翻译（TRANSLATION_ENABLED=true 时）
"""
import pytest
from unittest.mock import MagicMock, patch


# ==========================================
# 辅助工具
# ==========================================

def _make_mock_store(seen_result=False):
    """构造一个假的 URLStore"""
    mock_store = MagicMock()
    mock_store.seen.return_value = seen_result
    mock_store.mark.return_value = None
    return mock_store


# ==========================================
# 正常路径
# ==========================================

class TestScrapeSuccess:
    def test_returns_documents(self, tmp_path):
        """正常抓取返回非空 Document 列表"""
        from src.web_scraper import WebScraper
        scraper = WebScraper()

        fake_text = "这是一段工业设备维修手册的正文内容。" * 10
        mock_store = _make_mock_store(seen_result=False)

        with patch("src.web_scraper.get_url_store", return_value=mock_store), \
             patch("src.web_scraper._extract_with_trafilatura", return_value=fake_text), \
             patch("src.web_scraper._detect_language", return_value="zh"):

            docs = scraper.scrape("https://example.com/manual")

        assert len(docs) > 0
        assert all(hasattr(d, "page_content") for d in docs)

    def test_metadata_injected(self):
        """每个 Document 都有 source、chunk_id、lang、translated 字段"""
        from src.web_scraper import WebScraper
        scraper = WebScraper()

        fake_text = "设备故障排查手册内容。" * 20
        mock_store = _make_mock_store(seen_result=False)

        with patch("src.web_scraper.get_url_store", return_value=mock_store), \
             patch("src.web_scraper._extract_with_trafilatura", return_value=fake_text), \
             patch("src.web_scraper._detect_language", return_value="zh"):

            docs = scraper.scrape("https://example.com/manual")

        for doc in docs:
            assert "source" in doc.metadata
            assert "chunk_id" in doc.metadata
            assert "lang" in doc.metadata
            assert "translated" in doc.metadata
            assert doc.metadata["source"] == "https://example.com/manual"

    def test_url_marked_after_scrape(self):
        """抓取成功后 URLStore.mark 被调用"""
        from src.web_scraper import WebScraper
        scraper = WebScraper()
        mock_store = _make_mock_store(seen_result=False)

        with patch("src.web_scraper.get_url_store", return_value=mock_store), \
             patch("src.web_scraper._extract_with_trafilatura", return_value="内容" * 50), \
             patch("src.web_scraper._detect_language", return_value="zh"):

            scraper.scrape("https://example.com/page")

        mock_store.mark.assert_called_once_with("https://example.com/page")


# ==========================================
# 去重逻辑
# ==========================================

class TestDeduplication:
    def test_seen_url_returns_empty(self):
        """已抓取过的 URL 返回空列表，不重新抓取"""
        from src.web_scraper import WebScraper
        scraper = WebScraper()
        mock_store = _make_mock_store(seen_result=True)  # 标记为已见过

        with patch("src.web_scraper.get_url_store", return_value=mock_store):
            docs = scraper.scrape("https://example.com/page")

        assert docs == []

    def test_force_ignores_dedup(self):
        """force=True 时即使已见过也正常抓取"""
        from src.web_scraper import WebScraper
        scraper = WebScraper()
        mock_store = _make_mock_store(seen_result=True)  # 已见过

        with patch("src.web_scraper.get_url_store", return_value=mock_store), \
             patch("src.web_scraper._extract_with_trafilatura", return_value="内容" * 50), \
             patch("src.web_scraper._detect_language", return_value="zh"):

            docs = scraper.scrape("https://example.com/page", force=True)

        assert len(docs) > 0  # 强制重抓，有内容


# ==========================================
# 降级路径
# ==========================================

class TestFallback:
    def test_trafilatura_failure_uses_requests_fallback(self):
        """trafilatura 返回 None → 降级到 requests 方案"""
        from src.web_scraper import WebScraper
        scraper = WebScraper()
        mock_store = _make_mock_store(seen_result=False)
        fallback_text = "降级抓取到的内容，关于液压泵维修步骤。" * 10

        with patch("src.web_scraper.get_url_store", return_value=mock_store), \
             patch("src.web_scraper._extract_with_trafilatura", return_value=None), \
             patch("src.web_scraper._extract_with_requests_fallback", return_value=fallback_text), \
             patch("src.web_scraper._detect_language", return_value="zh"):

            docs = scraper.scrape("https://example.com/page")

        assert len(docs) > 0

    def test_both_methods_fail_raises_error(self):
        """两种抓取方式都失败时抛出 WebScraperError"""
        from src.web_scraper import WebScraper, WebScraperError
        scraper = WebScraper()
        mock_store = _make_mock_store(seen_result=False)

        with patch("src.web_scraper.get_url_store", return_value=mock_store), \
             patch("src.web_scraper._extract_with_trafilatura", return_value=None), \
             patch("src.web_scraper._extract_with_requests_fallback", return_value=None):

            with pytest.raises(WebScraperError, match="无法提取有效内容"):
                scraper.scrape("https://example.com/page")


# ==========================================
# URL 校验
# ==========================================

class TestURLValidation:
    def test_rejects_non_http_protocol(self):
        """file:// 协议应被拒绝"""
        from src.web_scraper import WebScraper, WebScraperError
        scraper = WebScraper()
        mock_store = _make_mock_store(seen_result=False)

        with patch("src.web_scraper.get_url_store", return_value=mock_store):
            with pytest.raises(WebScraperError, match="不支持的协议"):
                scraper.scrape("file:///etc/passwd")

    def test_rejects_missing_domain(self):
        """缺少域名的 URL 被拒绝"""
        from src.web_scraper import WebScraper, WebScraperError
        scraper = WebScraper()
        mock_store = _make_mock_store(seen_result=False)

        with patch("src.web_scraper.get_url_store", return_value=mock_store):
            with pytest.raises(WebScraperError, match="无效 URL"):
                scraper.scrape("https://")

    def test_accepts_http_and_https(self):
        """http 和 https 都应被接受（不因协议校验报错）"""
        from src.web_scraper import WebScraper
        scraper = WebScraper()
        mock_store = _make_mock_store(seen_result=False)

        for scheme in ("http", "https"):
            with patch("src.web_scraper.get_url_store", return_value=mock_store), \
                 patch("src.web_scraper._extract_with_trafilatura", return_value="内容" * 30), \
                 patch("src.web_scraper._detect_language", return_value="zh"):
                docs = scraper.scrape(f"{scheme}://example.com/page", force=True)
            assert isinstance(docs, list)


# ==========================================
# 翻译集成
# ==========================================

class TestTranslationIntegration:
    def test_chinese_content_not_translated(self):
        """中文内容不触发翻译"""
        from src.web_scraper import WebScraper
        scraper = WebScraper()
        mock_store = _make_mock_store(seen_result=False)

        with patch("src.web_scraper.get_url_store", return_value=mock_store), \
             patch("src.web_scraper._extract_with_trafilatura", return_value="中文内容" * 30), \
             patch("src.web_scraper._detect_language", return_value="zh"), \
             patch("src.web_scraper.settings") as mock_settings:

            mock_settings.TRANSLATION_ENABLED = True
            mock_settings.CHUNK_SIZE = 800
            mock_settings.CHUNK_OVERLAP = 150

            with patch("src.translator.Translator") as mock_translator_cls:
                docs = scraper.scrape("https://example.com/zh-page")

        # 中文内容不应调用 Translator
        mock_translator_cls.assert_not_called()
        assert all(not d.metadata["translated"] for d in docs)

    def test_english_content_triggers_translation(self):
        """英文内容在 TRANSLATION_ENABLED=true 时触发翻译"""
        from src.web_scraper import WebScraper
        scraper = WebScraper()
        mock_store = _make_mock_store(seen_result=False)

        mock_translator = MagicMock()
        mock_translator.translate.return_value = "已翻译的中文内容" * 20

        with patch("src.web_scraper.get_url_store", return_value=mock_store), \
             patch("src.web_scraper._extract_with_trafilatura",
                   return_value="English manual content about pump maintenance." * 10), \
             patch("src.web_scraper._detect_language", return_value="en"), \
             patch("src.web_scraper.settings") as mock_settings, \
             patch("src.web_scraper.Translator", return_value=mock_translator):

            mock_settings.TRANSLATION_ENABLED = True
            mock_settings.CHUNK_SIZE = 800
            mock_settings.CHUNK_OVERLAP = 150

            docs = scraper.scrape("https://example.com/en-manual", translate=True)

        mock_translator.translate.assert_called_once()
        assert all(d.metadata["translated"] for d in docs)
