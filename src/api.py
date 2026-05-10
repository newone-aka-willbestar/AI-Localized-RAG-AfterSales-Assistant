"""
FastAPI 后端入口。

关键设计说明：
1. RAG 是同步重操作（调大模型要几十秒），用 run_in_executor 放进线程池，
   不阻塞 FastAPI 的异步事件循环
2. 全局 rag 实例维护 all_documents 列表，保证多次上传后 BM25 不会丢失旧文档
3. CORS 只开放必要来源，不用 allow_origins=["*"]
4. LangSmith 追踪在 lifespan 最先初始化，确保在任何 LLM 调用前设好环境变量
"""
import asyncio
import gc
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import settings
from src.tracing import setup_tracing
from src.document_loader import DocumentLoader
from src.rag import RAG
from src.vector_store import VectorStore
from src.web_scraper import WebScraper, WebScraperError, get_url_store

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 生命周期钩子。

    启动顺序：
    1. LangSmith 追踪初始化（必须最先，确保后续所有 LLM 调用都被捕获）
    2. RAG 实例化（会触发 get_llm() 和缓存恢复，可能有 LLM 调用）

    这样即使启动阶段的 LLM 调用也会被 LangSmith 追踪到。
    """
    # Step 1: 初始化 LangSmith（在任何 LLM 调用之前）
    setup_tracing()

    # Step 2: 初始化全局资源
    global rag
    rag = RAG()

    logger.info("服务启动完成")
    yield
    # 关闭阶段（如需清理资源在此处理）
    logger.info("服务正在关闭")


app = FastAPI(title="华科制造 AI 智能客服", lifespan=lifespan)

# CORS：明确列出允许的来源，不用 * 全开
# 本地开发时前端跑在 8501，生产环境替换为真实域名
_cors_origins = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:8501,http://127.0.0.1:8501"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# 线程池：RAG 是同步代码，放进线程池执行，不阻塞异步事件循环
# max_workers=4 表示最多同时处理 4 个问答请求
_executor = ThreadPoolExecutor(max_workers=4)

# 全局 RAG 实例（由 lifespan 初始化，此处声明类型供静态分析）
rag: RAG = None  # type: ignore[assignment]


class QuestionRequest(BaseModel):
    question: str


class CrawlRequest(BaseModel):
    urls: List[str]
    force: bool = False       # True：忽略去重，强制重新抓取
    translate: bool = True    # True：非中文内容自动翻译（需 TRANSLATION_ENABLED=true）


async def verify_api_key(x_api_key: str = Header(None)):
    """API Key 鉴权中间件"""
    if x_api_key != settings.API_KEY:
        raise HTTPException(status_code=403, detail="无效 API Key")


@app.get("/health")
async def health():
    """
    健康检查接口。
    用途：Docker/Railway 检测服务是否就绪，面试时也能展示工程意识。
    """
    return {
        "status": "ok",
        "llm_provider": settings.LLM_PROVIDER,
        "retriever_ready": rag.final_retriever is not None,
    }


@app.post("/upload")
async def upload(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    """
    上传 PDF，解析后加入知识库。

    修复了原版的 BM25 覆盖 bug：
    原版：rag.init_retriever(新文档)  → BM25 只认识新文档，旧文档丢失
    现版：rag.add_documents(新文档)   → 累积所有文档，BM25 始终完整
    """
    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix != ".pdf":
        raise HTTPException(status_code=400, detail="只支持 PDF 格式")

    if file.size and file.size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大，最大支持 {settings.MAX_UPLOAD_SIZE // 1024 // 1024}MB"
        )

    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(await file.read())

        # 文档解析是 CPU 密集型同步操作，放线程池执行
        loop = asyncio.get_event_loop()
        loader = DocumentLoader()
        docs = await loop.run_in_executor(
            _executor,
            loader.load_and_split,
            tmp_path
        )

        # 向量库写入
        vector_store = VectorStore()
        await loop.run_in_executor(_executor, vector_store.add_documents, docs)

        # 累积式更新 RAG（关键修复：不是替换，是追加）
        await loop.run_in_executor(_executor, rag.add_documents, docs)

        return {
            "message": f"成功处理 {len(docs)} 个文本块",
            "filename": file.filename,
            "total_chunks": len(rag.all_documents),
        }

    except Exception as e:
        logger.error(f"上传失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        gc.collect()
        # asyncio.sleep 不阻塞事件循环，给 OS 时间释放文件句柄
        await asyncio.sleep(0.1)
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError as e:
                # 记录错误但不崩溃，Windows 偶尔会文件占用
                logger.warning(f"临时文件删除失败（不影响功能）: {e}")


@app.post("/crawl")
async def crawl(request: CrawlRequest, api_key: str = Depends(verify_api_key)):
    """
    批量抓取网页并入库。

    - 每次请求最多 SCRAPER_MAX_URLS_PER_REQUEST 个 URL
    - 已抓取过的 URL 默认跳过（force=True 强制重抓）
    - 非中文内容在 TRANSLATION_ENABLED=true 时自动翻译
    - 部分失败不影响其他 URL，错误信息单独返回

    返回格式：
    {
      "succeeded": [{"url": ..., "chunks": N}, ...],
      "skipped":   ["url1", ...],
      "failed":    [{"url": ..., "error": "..."}, ...]
    }
    """
    urls = request.urls
    if len(urls) > settings.SCRAPER_MAX_URLS_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"单次最多 {settings.SCRAPER_MAX_URLS_PER_REQUEST} 个 URL，"
                   f"当前传入 {len(urls)} 个"
        )
    if not urls:
        raise HTTPException(status_code=400, detail="urls 不能为空")

    loop = asyncio.get_event_loop()
    scraper = WebScraper()
    vector_store = VectorStore()

    succeeded = []
    skipped = []
    failed = []

    for url in urls:
        try:
            # scrape 是同步 IO 密集型，放线程池执行
            docs = await loop.run_in_executor(
                _executor,
                lambda u=url: scraper.scrape(u, force=request.force, translate=request.translate)
            )

            if not docs:
                # scrape 返回空列表 = URL 已被去重跳过
                skipped.append(url)
                continue

            # 写入向量库和 RAG 检索器
            await loop.run_in_executor(_executor, vector_store.add_documents, docs)
            await loop.run_in_executor(_executor, rag.add_documents, docs)

            succeeded.append({"url": url, "chunks": len(docs)})
            logger.info(f"URL 入库成功: {url}，{len(docs)} 块")

        except WebScraperError as e:
            failed.append({"url": url, "error": str(e)})
            logger.warning(f"抓取失败: {url} → {e}")
        except Exception as e:
            failed.append({"url": url, "error": f"内部错误: {e}"})
            logger.error(f"处理 URL 时出现意外错误: {url}", exc_info=True)

    return {
        "succeeded": succeeded,
        "skipped": skipped,
        "failed": failed,
        "total_chunks_in_kb": len(rag.all_documents),
    }


@app.get("/crawl/history")
async def crawl_history(api_key: str = Depends(verify_api_key)):
    """查看已抓取的 URL 列表"""
    store = get_url_store()
    return {
        "total": len(store),
        "urls": store.all_urls(),
    }


@app.post("/ask")
async def ask(request: QuestionRequest, api_key: str = Depends(verify_api_key)):
    """
    问答接口。

    rag.ask() 是同步函数（调大模型），用 run_in_executor 放进线程池，
    这样 FastAPI 在等待回答期间仍然可以处理其他请求。
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, rag.ask, request.question)
    return result
