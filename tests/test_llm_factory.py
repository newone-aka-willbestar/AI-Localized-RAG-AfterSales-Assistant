"""
LLM 工厂（重试与降级）测试。

策略：
- Mock 底层 LLM，不做真实 API 调用
- 验证：正常调用透传、重试触发次数、降级切换、链式操作符
"""
import pytest
from unittest.mock import MagicMock, patch, call

from src.llm_factory import LLMWithFallback, get_llm_with_fallback, _FallbackChain


# ==========================================
# 正常路径
# ==========================================

class TestNormalPath:

    def test_invoke_returns_primary_result(self):
        """主模型正常时，直接返回其结果"""
        mock_primary = MagicMock()
        mock_primary.invoke.return_value = "主模型答案"

        llm = LLMWithFallback.__new__(LLMWithFallback)
        llm._primary = mock_primary
        llm._fallback = None
        llm._using_fallback = False

        result = llm._invoke_with_retry(mock_primary, "测试输入")
        assert result == "主模型答案"

    def test_is_using_fallback_false_initially(self):
        """初始状态不是降级模式"""
        with patch("src.llm_factory._build_primary_llm", return_value=MagicMock()):
            llm = LLMWithFallback()
        assert llm.is_using_fallback is False


# ==========================================
# 降级路径
# ==========================================

class TestFallback:

    def test_fallback_triggered_when_primary_fails(self):
        """主模型抛 ConnectionError，切换到备用模型"""
        mock_primary = MagicMock()
        mock_primary.invoke.side_effect = ConnectionError("网络断了")

        mock_fallback = MagicMock()
        mock_fallback.invoke.return_value = "备用答案"

        llm = LLMWithFallback.__new__(LLMWithFallback)
        llm._primary = mock_primary
        llm._fallback = None
        llm._using_fallback = False

        with patch("src.llm_factory._build_fallback_llm", return_value=mock_fallback):
            with patch.object(LLMWithFallback, "_invoke_with_retry",
                              side_effect=[ConnectionError("超时"), "备用答案"]):
                result = llm.invoke("输入")

        assert result == "备用答案"

    def test_is_using_fallback_true_after_fallback(self):
        """降级发生后，is_using_fallback 变为 True"""
        mock_primary = MagicMock()
        mock_fallback = MagicMock()
        mock_fallback.invoke.return_value = "降级答案"

        llm = LLMWithFallback.__new__(LLMWithFallback)
        llm._primary = mock_primary
        llm._fallback = None
        llm._using_fallback = False

        with patch("src.llm_factory._build_fallback_llm", return_value=mock_fallback):
            with patch.object(LLMWithFallback, "_invoke_with_retry",
                              side_effect=[RuntimeError("API 挂了"), "降级答案"]):
                llm.invoke("输入")

        assert llm.is_using_fallback is True

    def test_fallback_llm_lazily_initialized(self):
        """降级 LLM 只在首次需要时初始化"""
        mock_primary = MagicMock()
        mock_fallback = MagicMock()
        mock_fallback.invoke.return_value = "ok"

        llm = LLMWithFallback.__new__(LLMWithFallback)
        llm._primary = mock_primary
        llm._fallback = None
        llm._using_fallback = False

        with patch("src.llm_factory._build_fallback_llm", return_value=mock_fallback) as mock_build:
            # 主模型正常 → build_fallback_llm 不应被调用
            with patch.object(LLMWithFallback, "_invoke_with_retry", return_value="主模型答案"):
                llm.invoke("输入")
            mock_build.assert_not_called()


# ==========================================
# 链式操作符
# ==========================================

class TestChaining:

    def test_pipe_operator_returns_fallback_chain(self):
        """llm | parser 返回 _FallbackChain"""
        with patch("src.llm_factory._build_primary_llm", return_value=MagicMock()):
            llm = LLMWithFallback()
        parser = MagicMock()
        chain = llm | parser
        assert isinstance(chain, _FallbackChain)

    def test_fallback_chain_invoke_calls_both_steps(self):
        """_FallbackChain.invoke 先调用 LLM，再调用下一步"""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "llm_output"

        mock_next = MagicMock()
        mock_next.invoke.return_value = "final_output"

        chain = _FallbackChain(mock_llm, mock_next)
        result = chain.invoke("input")

        mock_llm.invoke.assert_called_once_with("input")
        mock_next.invoke.assert_called_once_with("llm_output")
        assert result == "final_output"

    def test_multiple_pipe_chaining(self):
        """prompt | llm | parser 三段链正常执行"""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "llm_out"

        mock_parser = MagicMock()
        mock_parser.invoke.return_value = "parsed"

        with patch("src.llm_factory._build_primary_llm", return_value=mock_llm):
            llm = LLMWithFallback()

        chain = llm | mock_parser
        result = chain.invoke("raw_input")

        assert result == "parsed"


# ==========================================
# get_llm_with_fallback 工厂
# ==========================================

class TestFactory:

    def test_get_llm_with_fallback_returns_instance(self):
        with patch("src.llm_factory._build_primary_llm", return_value=MagicMock()):
            llm = get_llm_with_fallback()
        assert isinstance(llm, LLMWithFallback)

    def test_each_call_returns_new_instance(self):
        """每次调用返回独立实例，互不影响状态"""
        with patch("src.llm_factory._build_primary_llm", return_value=MagicMock()):
            llm1 = get_llm_with_fallback()
            llm2 = get_llm_with_fallback()
        assert llm1 is not llm2
