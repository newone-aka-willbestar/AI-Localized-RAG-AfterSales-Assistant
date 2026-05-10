"""
URL 去重存储模块。

职责：
- 记录已抓取过的 URL，防止重复入库
- 持久化到 JSON 文件，重启不丢数据
- URL 规范化（去掉 fragment、末尾斜杠统一）后取 SHA256 指纹

设计：
- 故意不用数据库，JSON 文件够用，零依赖，可直接 cat 查看内容
- 线程安全：写操作加 threading.Lock，API 并发时不会破坏文件
"""
import hashlib
import json
import logging
import threading
from pathlib import Path
from urllib.parse import urlparse, urlunparse

logger = logging.getLogger(__name__)


def _normalize_url(url: str) -> str:
    """
    URL 规范化，去除干扰因素后再比较。

    处理：
    - 去掉 fragment（#锚点不影响内容）
    - 统一 scheme 小写
    - path 末尾去掉多余的斜杠（/page/ == /page）
    """
    parsed = urlparse(url.strip())
    normalized = urlunparse((
        parsed.scheme.lower(),
        parsed.netloc.lower(),
        parsed.path.rstrip("/") or "/",
        parsed.params,
        parsed.query,
        ""  # 去掉 fragment
    ))
    return normalized


def _url_fingerprint(url: str) -> str:
    """取 SHA256 前 16 字节的 hex，作为 URL 的唯一指纹"""
    normalized = _normalize_url(url)
    return hashlib.sha256(normalized.encode()).hexdigest()[:32]


class URLStore:
    """
    已抓取 URL 的持久化存储。

    文件格式（JSON）：
    {
      "ab12cd34...": {
        "url": "https://example.com/page",
        "crawled_at": "2026-05-10T12:00:00"
      },
      ...
    }
    """

    def __init__(self, store_path: str = "./rag_cache/crawled_urls.json"):
        self._path = Path(store_path)
        self._lock = threading.Lock()
        self._data: dict = self._load()

    def _load(self) -> dict:
        """从磁盘加载，文件不存在时返回空 dict"""
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info(f"URL 去重库已加载，共 {len(data)} 条记录")
                return data
            except Exception as e:
                logger.warning(f"URL 去重库加载失败，将从空库开始: {e}")
        return {}

    def _save(self) -> None:
        """写回磁盘（调用前必须已持有 _lock）"""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def seen(self, url: str) -> bool:
        """判断 URL 是否已经抓取过"""
        fp = _url_fingerprint(url)
        return fp in self._data

    def mark(self, url: str) -> None:
        """标记 URL 为已抓取，并持久化"""
        from datetime import datetime, timezone
        fp = _url_fingerprint(url)
        with self._lock:
            self._data[fp] = {
                "url": _normalize_url(url),
                "crawled_at": datetime.now(timezone.utc).isoformat(),
            }
            self._save()

    def remove(self, url: str) -> bool:
        """
        从去重库中删除某条记录（用于重新抓取场景）。
        返回 True 表示确实删除了，False 表示原本不存在。
        """
        fp = _url_fingerprint(url)
        with self._lock:
            if fp in self._data:
                del self._data[fp]
                self._save()
                return True
        return False

    def all_urls(self) -> list[str]:
        """返回所有已抓取的规范化 URL 列表"""
        return [v["url"] for v in self._data.values()]

    def __len__(self) -> int:
        return len(self._data)
