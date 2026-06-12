"""
LLM 工厂模块（含重试与降级）。

使用 LangChain 原生的 with_retry() + with_fallbacks() 实现，
返回标准 Runnable，完全兼容 prompt | llm | parser 链式调用。

重试策略：
  - 最多重试 3 次，指数退避（jitter 抖动避免惊群）
  - 只对 transient 网络异常重试

降级策略：
  - 主模型全部重试耗尽后，自动切到备用 Ollama
  - 若备用也失败，抛出最终异常（由 rag.py 的 except 兜底）
"""
import logging

from src.config import settings

logger = logging.getLogger(__name__)

# 可重试的异常类型（网络相关，不含业务错误）
_RETRYABLE: tuple = (
    ConnectionError,
    TimeoutError,
    OSError,
)

try:
    import httpx
    _RETRYABLE = _RETRYABLE + (httpx.TimeoutException, httpx.ConnectError)
except ImportError:
    pass


def _build_primary_llm():
    """按配置构建主 LLM"""
    provider = settings.LLM_PROVIDER

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        logger.info(f"[LLM] 主模型: Ollama / {settings.OLLAMA_MODEL}")
        return ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=settings.TEMPERATURE,
            timeout=60,
        )

    elif provider == "deepseek":
        from langchain_openai import ChatOpenAI
        if not settings.DEEPSEEK_API_KEY:
            raise ValueError("LLM_PROVIDER=deepseek 但 DEEPSEEK_API_KEY 未设置")
        logger.info(f"[LLM] 主模型: DeepSeek / {settings.DEEPSEEK_MODEL}")
        return ChatOpenAI(
            model=settings.DEEPSEEK_MODEL,
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
            temperature=settings.TEMPERATURE,
        )

    else:
        raise ValueError(f"不支持的 LLM_PROVIDER: '{provider}'")


def _build_fallback_llm():
    """降级备用 LLM：始终用本地 Ollama"""
    from langchain_ollama import ChatOllama
    logger.warning(f"[LLM] 备用模型: Ollama / {settings.OLLAMA_MODEL}")
    return ChatOllama(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=settings.TEMPERATURE,
        timeout=60,
    )


def get_llm_with_fallback():
    """
    返回带重试和降级能力的 LLM Runnable。

    使用 LangChain 原生机制：
      - with_retry()      → 指数退避重试，返回 RunnableRetry
      - with_fallbacks()  → 主备切换，返回 RunnableWithFallbacks

    两者都是标准 Runnable，完全兼容 prompt | llm | parser 写法。
    """
    primary = _build_primary_llm().with_retry(
        retry_if_exception_type=_RETRYABLE,
        stop_after_attempt=3,
        wait_exponential_jitter=True,
    )

    try:
        fallback = _build_fallback_llm().with_retry(
            retry_if_exception_type=_RETRYABLE,
            stop_after_attempt=2,
            wait_exponential_jitter=True,
        )
        return primary.with_fallbacks([fallback])
    except Exception as e:
        # Ollama 未安装时，降级构建失败，只用主模型
        logger.warning(f"[LLM] 备用模型构建失败，仅使用主模型: {e}")
        return primary
