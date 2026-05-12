"""
LLM 工厂模块测试。

策略：
- Mock LangChain LLM 对象，不做真实 API 调用
- 验证：返回值是 Runnable、管道操作符兼容、降级构建失败时的容错
"""
import pytest
from unittest.mock import MagicMock, patch


# ==========================================
# get_llm_with_fallback 返回标准 Runnable
# ==========================================

class TestGetLLMWithFallback:

    def _mock_chat_llm(self):
        """构造一个具备 with_retry / with_fallbacks / invoke 的 mock LLM"""
        llm = MagicMock()
        # with_retry 返回自身（模拟链式调用）
        retried = MagicMock()
        retried.with_fallbacks = MagicMock(return_value=retried)
        retried.invoke = MagicMock(return_value=MagicMock(content="mock answer"))
        # 支持 | 操作符（LangChain Runnable 接口）
        retried.__or__ = MagicMock(return_value=MagicMock())
        llm.with_retry = MagicMock(return_value=retried)
        return llm, retried

    def test_returns_object_with_invoke(self):
        """返回的对象必须有 invoke 方法（Runnable 协议）"""
        with patch("src.llm_factory._build_primary_llm") as mock_primary, \
             patch("src.llm_factory._build_fallback_llm") as mock_fallback:
            primary_llm, primary_retried = self._mock_chat_llm()
            fallback_llm, fallback_retried = self._mock_chat_llm()
            mock_primary.return_value = primary_llm
            mock_fallback.return_value = fallback_llm

            from src.llm_factory import get_llm_with_fallback
            result = get_llm_with_fallback()
            assert hasattr(result, "invoke")

    def test_fallback_build_failure_still_returns_runnable(self):
        """备用模型构建失败时，只用主模型，不抛出异常"""
        with patch("src.llm_factory._build_primary_llm") as mock_primary, \
             patch("src.llm_factory._build_fallback_llm") as mock_fallback:
            primary_llm, primary_retried = self._mock_chat_llm()
            mock_primary.return_value = primary_llm
            mock_fallback.side_effect = RuntimeError("Ollama 未安装")

            from src.llm_factory import get_llm_with_fallback
            result = get_llm_with_fallback()
            assert result is not None
            assert hasattr(result, "invoke")

    def test_primary_llm_called_with_retry(self):
        """主模型应该经过 with_retry 包装"""
        with patch("src.llm_factory._build_primary_llm") as mock_primary, \
             patch("src.llm_factory._build_fallback_llm") as mock_fallback:
            primary_llm, primary_retried = self._mock_chat_llm()
            mock_primary.return_value = primary_llm
            mock_fallback.side_effect = RuntimeError("跳过备用")

            from src.llm_factory import get_llm_with_fallback
            get_llm_with_fallback()
            primary_llm.with_retry.assert_called_once()


# ==========================================
# _build_primary_llm 配置分支
# ==========================================

class TestBuildPrimaryLLM:

    def test_deepseek_provider_uses_chat_openai(self):
        with patch("src.llm_factory.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "deepseek"
            mock_settings.DEEPSEEK_API_KEY = "sk-test"
            mock_settings.DEEPSEEK_MODEL = "deepseek-chat"
            mock_settings.DEEPSEEK_BASE_URL = "https://api.deepseek.com"
            mock_settings.TEMPERATURE = 0.0

            # ChatOpenAI 在函数内懒加载，patch 原始模块路径
            with patch("langchain_openai.ChatOpenAI") as mock_cls:
                mock_cls.return_value = MagicMock()
                from src.llm_factory import _build_primary_llm
                _build_primary_llm()
                mock_cls.assert_called_once()

    def test_deepseek_raises_when_no_api_key(self):
        with patch("src.llm_factory.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "deepseek"
            mock_settings.DEEPSEEK_API_KEY = ""

            from src.llm_factory import _build_primary_llm
            with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
                _build_primary_llm()

    def test_ollama_provider_uses_chat_ollama(self):
        with patch("src.llm_factory.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "ollama"
            mock_settings.OLLAMA_MODEL = "qwen2:7b"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.TEMPERATURE = 0.0

            # ChatOllama 在函数内懒加载，patch 原始模块路径
            with patch("langchain_ollama.ChatOllama") as mock_cls:
                mock_cls.return_value = MagicMock()
                from src.llm_factory import _build_primary_llm
                _build_primary_llm()
                mock_cls.assert_called_once()

    def test_unknown_provider_raises_value_error(self):
        with patch("src.llm_factory.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "unknown_provider"
            from src.llm_factory import _build_primary_llm
            with pytest.raises(ValueError):
                _build_primary_llm()
