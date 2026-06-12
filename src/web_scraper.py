"""
网页内容抓取模块。

职责：
- 给定 URL，抓取正文内容并切分为 LangChain Document 列表
- 内置 URL 去重：已抓取的 URL 自动跳过
- 内置语言检测：非中文内容交给 translator 处理（TRANSLATION_ENABLED=true 时）

技术选型：
- trafilatura：纯 Python，正文提取准确率高，不需要浏览器
  适合大多数静态/服务端渲染页面（百科、博客、文档站）
- 不引入 crawl4ai（需要 Playwright，启动开销大），
  如果遇到 JS 渲染页面，用户可在 .env 中配置 SCRAPER_USE_BROWSER=true 扩展

降级策略：
  trafilatura 提取失败 → requests 直接取 raw HTML 的 <p> 标签文本
  网络请求失败 → 抛出 WebScraperError，由调用方决定是否跳过
"""
import logging
from typing import Optional
from urllib.parse import urlparse

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.url_store import URLStore

logger = logging.getLogger(__name__)
Translator = None

# 模块级全局 URLStore，API 进程共用同一个实例（单例）
_url_store: Optional[URLStore] = None


def get_url_store() -> URLStore:
    """懒加载全局 URLStore（测试时可通过 patch 替换）"""
    global _url_store
    if _url_store is None:
        import os
        store_path = os.path.join(settings.VECTORSTORE_PATH, "crawled_urls.json")
        _url_store = URLStore(store_path=store_path)
    return _url_store


class WebScraperError(Exception):
    """网页抓取失败（网络错误、内容提取失败等）"""
    pass


def _validate_url(url: str) -> None:
    """
    基本 URL 格式校验。

    只允许 http/https，拒绝 file:// / ftp:// 等协议，
    防止用户误传本地路径或恶意协议。
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise WebScraperError(f"不支持的协议: {parsed.scheme}，只允许 http/https")
    if not parsed.netloc:
        raise WebScraperError(f"无效 URL（缺少域名）: {url}")


def _extract_with_trafilatura(url: str) -> Optional[str]:
    """
    用 trafilatura 提取正文。

    trafilatura 会自动：
    - 去掉导航栏、广告、页脚
    - 保留正文段落、标题、列表
    - 处理编码
    返回 None 表示提取失败（空页面、反爬等）。
    """
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            no_fallback=False,  # 允许降级到备用算法
        )
        return text
    except ImportError:
        raise WebScraperError(
            "trafilatura 未安装，请运行: pip install trafilatura"
        )
    except Exception as e:
        logger.warning(f"trafilatura 提取失败: {e}")
        return None


def _extract_with_requests_fallback(url: str) -> Optional[str]:
    """
    trafilatura 失败时的降级方案：直接用 requests + 简单解析。

    只取 <p> 标签文本，过滤掉太短的段落（导航链接等）。
    质量不如 trafilatura，但保证不崩溃。
    """
    try:
        import requests
        from html.parser import HTMLParser

        class _PTagParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self._in_p = False
                self.paragraphs = []
                self._buf = []

            def handle_starttag(self, tag, attrs):
                if tag == "p":
                    self._in_p = True
                    self._buf = []

            def handle_endtag(self, tag):
                if tag == "p" and self._in_p:
                    text = "".join(self._buf).strip()
                    if len(text) > 20:  # 过滤掉太短的段落
                        self.paragraphs.append(text)
                    self._in_p = False

            def handle_data(self, data):
                if self._in_p:
                    self._buf.append(data)

        resp = requests.get(url, timeout=settings.SCRAPER_TIMEOUT, headers={
            "User-Agent": "Mozilla/5.0 (compatible; IndustrialRAGBot/1.0)"
        })
        resp.raise_for_status()
        parser = _PTagParser()
        parser.feed(resp.text)
        text = "\n\n".join(parser.paragraphs)
        return text if len(text) > 50 else None

    except Exception as e:
        logger.warning(f"降级抓取也失败: {e}")
        return None


class WebScraper:
    """
    网页内容抓取器。

    用法：
        scraper = WebScraper()
        docs = scraper.scrape("https://example.com/manual")
        # docs 是 List[Document]，可以直接传给 rag.add_documents()
    """

    def __init__(self):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )

    def scrape(
        self,
        url: str,
        force: bool = False,
        translate: bool = True,
    ) -> list[Document]:
        """
        抓取单个 URL，返回切分后的 Document 列表。

        Args:
            url:       目标网页地址
            force:     True 时跳过去重检查，强制重新抓取
            translate: True 时对非中文内容自动翻译（需要 TRANSLATION_ENABLED=true）

        Returns:
            List[Document]，每个 Document 带 metadata：
            {"source": url, "chunk_id": int, "lang": str, "translated": bool}

        Raises:
            WebScraperError: URL 格式非法、网络失败、内容为空
        """
        # 1. URL 格式校验
        _validate_url(url)

        # 2. 去重检查
        store = get_url_store()
        if not force and store.seen(url):
            logger.info(f"URL 已抓取过，跳过: {url}")
            return []

        # 3. 抓取正文
        logger.info(f"开始抓取: {url}")
        text = _extract_with_trafilatura(url)
        if not text:
            logger.warning(f"trafilatura 提取失败，尝试降级方案: {url}")
            text = _extract_with_requests_fallback(url)
        if not text:
            raise WebScraperError(f"无法提取有效内容: {url}")

        logger.info(f"成功提取正文，字符数: {len(text)}")

        # 4. 语言检测 + 翻译
        lang = _detect_language(text)
        translated = False

        if translate and settings.TRANSLATION_ENABLED and lang != "zh":
            try:
                global Translator
                translator_cls = Translator
                if translator_cls is None:
                    from src.translator import Translator as translator_cls
                    Translator = translator_cls
                translator = translator_cls()
                text = translator.translate(text)
                translated = True
                logger.info(f"已翻译（{lang} → zh）")
            except Exception as e:
                logger.warning(f"翻译失败，使用原文: {e}")

        # 5. 切分
        raw_doc = Document(page_content=text, metadata={})
        chunks = self._splitter.split_documents([raw_doc])

        # 6. 注入 metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "source": url,
                "chunk_id": i,
                "lang": lang,
                "translated": translated,
            })

        # 7. 标记为已抓取
        store.mark(url)
        logger.info(f"抓取完成，共 {len(chunks)} 个文本块: {url}")
        return chunks


def _detect_language(text: str) -> str:
    """
    检测文本语言，返回 ISO 639-1 代码（"zh"、"en"、"ja" 等）。

    langdetect 需要足够多的文本才准确，取前 500 字符检测即可。
    检测失败时返回 "unknown"，不崩溃。
    """
    try:
        from langdetect import detect
        sample = text[:500]
        lang = detect(sample)
        # langdetect 对中文返回 "zh-cn" 或 "zh-tw"，统一为 "zh"
        if lang.startswith("zh"):
            return "zh"
        return lang
    except ImportError:
        logger.warning("langdetect 未安装，语言检测跳过")
        return "unknown"
    except Exception:
        return "unknown"
