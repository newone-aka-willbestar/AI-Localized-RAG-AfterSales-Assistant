"""
LangSmith 链路追踪初始化模块。

LangSmith 的工作原理：
  LangChain 在检测到以下环境变量时，自动将所有 LLM 调用、Chain 执行、
  Retriever 查询的输入输出、延迟、token 数发送到 LangSmith 平台：
    - LANGCHAIN_TRACING_V2=true
    - LANGCHAIN_API_KEY=ls__xxx
    - LANGCHAIN_PROJECT=my-project（可选，用于在平台上分组）

  无需修改任何业务代码，对现有链路零侵入。

本模块额外提供：
  - setup_tracing()：在 FastAPI 启动时调用，统一设置环境变量
  - get_run_config()：生成带 run_name / metadata / tags 的配置字典，
    传入 chain.invoke(..., config=...) 后在 LangSmith 上看到更清晰的追踪树

自动追踪的内容（开箱即用）：
  ┌─────────────────────────────────────────────────────┐
  │ RAG.ask()                                           │
  │   ├── HyDE.generate()  [LLM 调用 #1]               │
  │   │     └── ChatPromptTemplate → ChatOllama         │
  │   ├── EnsembleRetriever.invoke()                    │
  │   │     ├── VectorStoreRetriever (Qdrant)           │
  │   │     └── BM25Retriever                           │
  │   ├── FlashrankRerank                               │
  │   └── 生成链 [LLM 调用 #2]                          │
  │         └── ChatPromptTemplate → ChatOllama         │
  └─────────────────────────────────────────────────────┘

  Translator.translate() 和 WebScraper.scrape() 通过 @traceable 装饰器
  也会出现在追踪树中。
"""
import logging
import os

from src.config import settings

logger = logging.getLogger(__name__)


def setup_tracing() -> bool:
    """
    读取 config 中的 LangSmith 配置并写入环境变量。

    LangChain 只认环境变量，不认 Python 对象，所以必须在 import langchain 之前
    （或至少在第一次调用 LLM 之前）设置好。FastAPI lifespan 启动时调用最合适。

    Returns:
        True  = 追踪已启用
        False = 追踪未启用（LANGCHAIN_TRACING_V2=false 或 API Key 为空）
    """
    if not settings.LANGCHAIN_TRACING_V2:
        logger.info("LangSmith 追踪未启用（LANGCHAIN_TRACING_V2=false）")
        return False

    if not settings.LANGCHAIN_API_KEY:
        logger.warning(
            "LANGCHAIN_TRACING_V2=true 但 LANGCHAIN_API_KEY 为空，追踪不会生效。"
            "请在 .env 中添加：LANGCHAIN_API_KEY=ls__你的key"
        )
        return False

    # LangChain 通过这四个环境变量控制追踪行为
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT

    logger.info(
        f"LangSmith 追踪已启用 → project: '{settings.LANGCHAIN_PROJECT}' "
        f"endpoint: {settings.LANGCHAIN_ENDPOINT}"
    )
    return True


def get_run_config(
    run_name: str,
    tags: list[str] | None = None,
    metadata: dict | None = None,
) -> dict:
    """
    生成 LangChain chain.invoke() 的 config 参数。

    用法：
        config = get_run_config("RAG.ask", tags=["prod"], metadata={"question": q})
        chain.invoke(input, config=config)

    在 LangSmith 上的效果：
        - run_name  → 追踪树节点名称（替代默认的 "RunnableSequence"）
        - tags      → 可在平台过滤（如按 "prod" / "dev" / "hyde-enabled" 筛选）
        - metadata  → 附加在 run 上的 KV 信息（如 question、doc_count）

    Args:
        run_name: 在 LangSmith 追踪树中显示的名称
        tags:     字符串标签列表，方便在平台过滤
        metadata: 附加的 KV 元数据，不参与 LLM 调用，仅供查询分析

    Returns:
        可直接传给 chain.invoke(config=...) 的字典
    """

    config: dict = {"run_name": run_name}

    # 默认 tag：当前 LLM 提供商，方便对比 Ollama vs DeepSeek 的效果差异
    default_tags = [f"provider:{settings.LLM_PROVIDER}"]
    if settings.HYDE_ENABLED:
        default_tags.append("hyde:on")
    else:
        default_tags.append("hyde:off")

    config["tags"] = default_tags + (tags or [])

    if metadata:
        config["metadata"] = metadata

    return config
