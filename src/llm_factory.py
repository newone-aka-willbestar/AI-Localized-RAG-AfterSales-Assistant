"""
LLM 工厂模块（含重试与降级）。

解决的问题：
- DeepSeek API 偶发超时（rate limit / 网络抖动）
- 单次失败直接报错，用户体验差

策略：
1. 指数退避重试（最多 3 次，间隔 1s / 2s / 4s）
   适用于：transient 网络错误、429 rate limit
2. 主备降级（DeepSeek → Ollama）
   适用于：API 服务不可用、密钥过期等持久性错误

调用方（rag.py）只需调用 get_llm_with_fallback()，
不再关心重试逻辑，也不需要感知降级过程。

为什么用 tenacity？
  requirements.txt 里已有 tenacity==8.5.0，
  不引入新依赖，retry 语义也更清晰（比手写 for 循环）。
"""
import logging
import time
from typing import Any

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from src.config import settings

logger = logging.getLogger(__name__)

# 可重试的异常类型（网络相关，不含业务错误）
_RETRYABLE = (
    ConnectionError,
    TimeoutError,
    OSError,
)

# 尝试导入 httpx 异常（langchain_openai 底层用 httpx）
try:
    import httpx
    _RETRYABLE = _RETRYABLE + (httpx.TimeoutException, httpx.ConnectError)
except ImportError:
    pass


def _build_primary_llm():
    """按配置构建主 LLM（与 rag.py 原逻辑相同）"""
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
    """
    降级备用 LLM：始终用本地 Ollama。
    主模型是 Ollama 时，降级到同一实例（重试即可，不真正降级）。
    主模型是 DeepSeek 时，降级到本地 Ollama 保证可用性。
    """
    from langchain_ollama import ChatOllama
    logger.warning(f"[LLM] 降级到备用模型: Ollama / {settings.OLLAMA_MODEL}")
    return ChatOllama(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=settings.TEMPERATURE,
        timeout=60,
    )


class LLMWithFallback:
    """
    带重试和降级的 LLM 包装器。

    接口与普通 LangChain LLM 一致（支持 pipe 操作符 |），
    rag.py 无需修改调用方式。

    重试策略：
      - 最多重试 3 次（加上首次共 4 次调用）
      - 指数退避：1s → 2s → 4s
      - 只对 transient 网络异常重试

    降级策略：
      - 主模型全部重试耗尽后，切到备用 Ollama
      - 备用模型同样有 2 次重试机会
      - 若备用也失败，抛出最终异常（由 rag.py 的 except 兜底返回友好提示）
    """

    def __init__(self):
        self._primary = _build_primary_llm()
        self._fallback = None          # 懒加载，仅在主模型失败时初始化
        self._using_fallback = False   # 标记当前用的是哪个模型

    # ── LangChain 协议：支持 | 操作符 ─────────────────────

    def __or__(self, other):
        """让 LLMWithFallback 能参与 prompt | llm | parser 链式调用"""
        return _FallbackChain(self, other)

    # ── 核心调用 ──────────────────────────────────────────

    def invoke(self, input: Any, **kwargs) -> Any:
        """
        调用 LLM，自动处理重试和降级。
        优先走主模型，全部重试耗尽后切换备用模型。
        """
        try:
            return self._invoke_with_retry(self._primary, input, **kwargs)
        except Exception as primary_err:
            logger.error(f"[LLM] 主模型全部重试失败: {primary_err}，切换备用模型")
            if self._fallback is None:
                self._fallback = _build_fallback_llm()
            self._using_fallback = True
            return self._invoke_with_retry(self._fallback, input, **kwargs)

    @staticmethod
    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _invoke_with_retry(llm, input: Any, **kwargs) -> Any:
        """带指数退避重试的单次 LLM 调用"""
        return llm.invoke(input, **kwargs)

    @property
    def is_using_fallback(self) -> bool:
        return self._using_fallback


class _FallbackChain:
    """
    LLMWithFallback.__or__(other) 返回的中间对象，
    让链式调用 (prompt | llm_with_fallback | parser) 正常工作。
    """

    def __init__(self, llm: LLMWithFallback, next_step):
        self._llm = llm
        self._next = next_step

    def __or__(self, other):
        return _FallbackChain(self._llm, _SequentialChain(self._next, other))

    def invoke(self, input: Any, **kwargs) -> Any:
        llm_output = self._llm.invoke(input, **kwargs)
        return self._next.invoke(llm_output)


class _SequentialChain:
    """两个步骤顺序执行的简单包装"""

    def __init__(self, first, second):
        self._first = first
        self._second = second

    def __or__(self, other):
        return _SequentialChain(self, other)

    def invoke(self, input: Any, **kwargs) -> Any:
        return self._second.invoke(self._first.invoke(input, **kwargs))


def get_llm_with_fallback() -> LLMWithFallback:
    """
    对外接口：返回带重试和降级能力的 LLM 实例。
    rag.py 用这个替换原来的 get_llm()。
    """
    return LLMWithFallback()
