"""
FastAPI 后端入口。

关键设计说明：
1. RAG 是同步重操作（调大模型要几十秒），用 run_in_executor 放进线程池，
   不阻塞 FastAPI 的异步事件循环
2. 全局 rag 实例维护 all_documents 列表，保证多次上传后 BM25 不会丢失旧文档
3. CORS 只开放必要来源，不用 allow_origins=["*"]
4. LangSmith 追踪在 lifespan 最先初始化，确保在任何 LLM 调用前设好环境变量
5. SessionMemory 以 session_id 为 key 保存多轮对话历史，解决指代消歧问题
"""
import asyncio
import gc
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import settings
from src.tracing import setup_tracing
from src.document_loader import DocumentLoader
from src.intent_classifier import IntentClassifier
from src.badcase_store import BadcaseStore
from src.memory import SessionMemory
from src.graph import build_ask_graph, make_initial_state
from src.rag import RAG
from src.report_generator import ReportGenerator, REPORT_TYPES
from src.vector_store import VectorStore
from src.web_scraper import WebScraper, WebScraperError, get_url_store
from src import audit_log

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
    global rag, intent_classifier, ask_graph
    rag = RAG()

    # Step 3: 意图分类器（复用 VectorStore 已加载的 Embedding 模型）
    try:
        embeddings = rag.vector_store.embeddings
        intent_classifier = IntentClassifier(embeddings=embeddings)
    except Exception as e:
        logger.warning(f"意图分类器初始化失败，将退化为纯规则模式: {e}")
        intent_classifier = IntentClassifier(embeddings=None)

    # Step 4: 编译 LangGraph 问答工作流
    # 依赖注入：把 rag / intent_classifier / session_memory 传入图，不在图内持有全局状态
    ask_graph = build_ask_graph(
        rag=rag,
        intent_classifier=intent_classifier,
        session_memory=session_memory,
    )

    # Step 5: 报表生成器（复用 rag 的检索器和 LLM）
    global report_generator
    report_generator = ReportGenerator(rag=rag)

    logger.info("服务启动完成")
    yield
    # 关闭阶段（如需清理资源在此处理）
    logger.info("服务正在关闭")


app = FastAPI(title="AI 数字员工", lifespan=lifespan)

# CORS：明确列出允许的来源，不用 * 全开
# 本地开发时 Gradio 前端跑在 7860，生产环境替换为真实域名
_cors_origins = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:7860,http://127.0.0.1:7860"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

# 线程池：RAG 是同步代码，放进线程池执行，不阻塞异步事件循环
# max_workers=4 表示最多同时处理 4 个问答请求
_executor = ThreadPoolExecutor(max_workers=4)

# 全局实例（由 lifespan 初始化）
rag: RAG = None                              # type: ignore[assignment]
intent_classifier: IntentClassifier = None   # type: ignore[assignment]
ask_graph = None                             # LangGraph 编译图，lifespan 初始化后就绪
report_generator: ReportGenerator = None     # type: ignore[assignment]
badcase_store: BadcaseStore = BadcaseStore() # 启动即就绪，路径从默认值取
session_memory: SessionMemory = SessionMemory()  # 多轮会话记忆，按 session_id 隔离


class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = None  # 前端生成的会话 ID，用于多轮记忆


class CrawlRequest(BaseModel):
    urls: List[str]
    force: bool = False       # True：忽略去重，强制重新抓取
    translate: bool = True    # True：非中文内容自动翻译（需 TRANSLATION_ENABLED=true）


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    feedback: str             # "bad" 或 "good"
    intent: str = ""
    sources: List[dict] = []
    note: str = ""            # 用户附加说明（可选）


class ReportRequest(BaseModel):
    topic: str                           # 报告主题（用于检索 + 标题）
    report_type: str = "summary"         # summary / keypoints / review / comparison / custom
    custom_instruction: str = ""         # 仅 report_type=custom 时有效
    max_sources: int = 8                 # 最多使用的文献片段数（1-20）


async def verify_api_key(x_api_key: str = Header(None)):
    """API Key 鉴权中间件"""
    if x_api_key != settings.API_KEY:
        raise HTTPException(status_code=403, detail="无效 API Key")


@app.get("/health")
async def health():
    """
    健康检查接口。

    返回字段：
    - status:            "ok" 表示服务就绪
    - llm_provider:      当前使用的 LLM 提供商
    - retriever_ready:   检索引擎是否初始化完毕（上传文档后才为 true）
    - doc_count:         知识库当前文档块数量
    - active_sessions:   当前活跃的多轮会话数
    - graph_ready:       LangGraph 工作流是否就绪
    - hyde_enabled:      HyDE 检索增强是否开启
    - feedback_stats:    累计反馈统计（总数 / 好评 / 差评）
    """
    # 顺带清理过期会话，保证 active_sessions 数字准确
    session_memory.evict_expired()

    return {
        "status":           "ok",
        "llm_provider":     settings.LLM_PROVIDER,
        "retriever_ready":  rag is not None and rag.final_retriever is not None,
        "doc_count":        len(rag.all_documents) if rag else 0,
        "active_sessions":  session_memory.session_count(),
        "graph_ready":      ask_graph is not None,
        "hyde_enabled":     settings.HYDE_ENABLED,
        "feedback_stats":   badcase_store.stats(),
    }


@app.post("/upload")
async def upload(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    """
    上传文档（PDF / Word），解析后加入知识库。

    支持格式：.pdf / .docx
    修复了原版的 BM25 覆盖 bug：
    原版：rag.init_retriever(新文档)  → BM25 只认识新文档，旧文档丢失
    现版：rag.add_documents(新文档)   → 累积所有文档，BM25 始终完整
    """
    from src.document_loader import SUPPORTED_EXTENSIONS
    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的格式 '{suffix}'，当前支持：{', '.join(SUPPORTED_EXTENSIONS)}"
        )

    content = await file.read()
    if len(content) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大，最大支持 {settings.MAX_UPLOAD_SIZE // 1024 // 1024}MB"
        )

    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(content)

        # 文档解析是 CPU 密集型同步操作，放线程池执行
        loop = asyncio.get_running_loop()
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
    if not urls:
        raise HTTPException(status_code=400, detail="urls 不能为空")
    if len(urls) > settings.SCRAPER_MAX_URLS_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"单次最多 {settings.SCRAPER_MAX_URLS_PER_REQUEST} 个 URL，"
                   f"当前传入 {len(urls)} 个"
        )

    loop = asyncio.get_running_loop()
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


@app.post("/feedback")
async def feedback(request: FeedbackRequest, api_key: str = Depends(verify_api_key)):
    """
    用户反馈接口（点赞/点踩）。

    bad 反馈会写入 SQLite，供后续分析和模型改进。
    good 反馈同样记录，用于统计满意率。
    """
    if request.feedback not in ("bad", "good"):
        raise HTTPException(status_code=400, detail="feedback 只接受 'bad' 或 'good'")

    record_id = badcase_store.record(
        question=request.question,
        answer=request.answer,
        feedback=request.feedback,
        intent=request.intent,
        sources=request.sources,
        note=request.note,
    )
    return {"id": record_id, "recorded": True}


@app.get("/feedback/stats")
async def feedback_stats(api_key: str = Depends(verify_api_key)):
    """查看反馈统计：总数、点赞数、点踩数"""
    return badcase_store.stats()


@app.get("/feedback/badcases")
async def list_badcases(limit: int = 50, api_key: str = Depends(verify_api_key)):
    """查看最近的 bad 反馈列表，用于分析改进"""
    return {"items": badcase_store.list_bad(limit=limit)}


@app.post("/ask")
async def ask(request: QuestionRequest, api_key: str = Depends(verify_api_key)):
    """
    问答接口，由 LangGraph StateGraph 驱动：

    流程图（src/graph.py）：
      classify → [chitchat | load_history → rag_ask] → save_memory

    - chitchat  → 跳过 RAG，直接 LLM 回复
    - 其他意图  → 完整 RAG 流程 + 历史上下文注入
    - session_id 可选，传入时开启多轮记忆，不传则单轮无状态（向后兼容）
    """
    loop = asyncio.get_running_loop()

    # 构建初始状态，交给图执行
    initial_state = make_initial_state(
        question=request.question,
        session_id=request.session_id,
    )

    # 计时 + 执行图（同步，放线程池避免阻塞事件循环）
    error_msg: Optional[str] = None
    final_state = None
    with audit_log.Timer() as timer:
        try:
            final_state = await loop.run_in_executor(
                _executor,
                lambda: ask_graph.invoke(initial_state),
            )
        except Exception as e:
            error_msg = str(e)
            logger.error(f"/ask 执行异常: {e}", exc_info=True)

    # 审计日志：不管成功失败都记录，fire-and-forget
    audit_log.record(
        question=request.question,
        intent=final_state["intent"] if final_state else "error",
        latency_ms=timer.elapsed_ms,
        answer_length=len(final_state["answer"]) if final_state else 0,
        source_count=len(final_state["sources"]) if final_state else 0,
        session_id=request.session_id,
        error=error_msg,
        provider=final_state["provider"] if final_state else settings.LLM_PROVIDER,
        has_history=bool(final_state.get("history")) if final_state else False,
    )

    if error_msg:
        raise HTTPException(status_code=500, detail="服务繁忙，请稍后再试")

    return {
        "answer":   final_state["answer"],
        "sources":  final_state["sources"],
        "provider": final_state["provider"],
        "intent":   final_state["intent"],
    }


@app.delete("/session/{session_id}")
async def clear_session(session_id: str, api_key: str = Depends(verify_api_key)):
    """
    清除指定会话的对话历史。
    用户点击"清空对话"或开始新会话时调用，释放服务端内存。
    """
    session_memory.clear(session_id)
    return {"cleared": True, "session_id": session_id}


@app.get("/session/stats")
async def session_stats(api_key: str = Depends(verify_api_key)):
    """查看当前活跃会话数（运维用）"""
    evicted = session_memory.evict_expired()
    return {
        "active_sessions": session_memory.session_count(),
        "evicted_this_call": evicted,
    }


@app.post("/report")
async def generate_report(request: ReportRequest, api_key: str = Depends(verify_api_key)):
    """
    报表生成接口。

    根据知识库中的文献内容，按指定类型生成结构化 Markdown 报告。
    面向论文写作场景，支持五种类型：
      summary     文献摘要
      keypoints   要点提炼
      review      文献综述
      comparison  对比分析
      custom      自定义（需填 custom_instruction）

    返回：
      report:   Markdown 格式报告正文
      sources:  参考来源列表
      type:     报告类型标签
      topic:    报告主题
    """
    if request.max_sources < 1 or request.max_sources > 20:
        raise HTTPException(status_code=400, detail="max_sources 范围：1-20")

    loop = asyncio.get_running_loop()

    with audit_log.Timer() as timer:
        result = await loop.run_in_executor(
            _executor,
            lambda: report_generator.generate(
                topic=request.topic,
                report_type=request.report_type,
                custom_instruction=request.custom_instruction,
                max_sources=request.max_sources,
            ),
        )

    audit_log.record(
        question=f"[报表] {request.topic}",
        intent=f"report:{request.report_type}",
        latency_ms=timer.elapsed_ms,
        answer_length=len(result.get("report", "")),
        source_count=len(result.get("sources", [])),
    )

    return result


@app.get("/report/types")
async def report_types():
    """获取支持的报告类型列表（前端选择器用）"""
    return {"types": [{"value": k, "label": v} for k, v in REPORT_TYPES.items()]}
