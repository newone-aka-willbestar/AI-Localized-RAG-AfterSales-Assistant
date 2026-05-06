"""
测试 RAG 模块中的工厂函数与工具函数。

分两类：
1. sanitize_metadata - 纯函数，零外部依赖，任何环境都能跑
2. get_llm 工厂函数 - 需要 langchain，用 skipif 标记
"""
import pytest
import importlib
from unittest.mock import patch, MagicMock


# ==========================================
# sanitize_metadata 测试（零依赖，必须全部通过）
# ==========================================

class TestSanitizeMetadata:
    """测试 metadata 清洗函数（纯函数，无需任何 mock）"""

    def test_numpy_float32_converted_to_python_float(self):
        """numpy.float32 -> Python float"""
        numpy = pytest.importorskip("numpy", reason="需要 numpy")
        from src.rag import sanitize_metadata

        metadata = {"score": numpy.float32(0.95), "source": "test.pdf"}
        result = sanitize_metadata(metadata)

        assert isinstance(result["score"], float)
        assert not hasattr(result["score"], "item")
        assert abs(result["score"] - 0.95) < 0.001

    def test_numpy_int64_converted_to_python_int(self):
        """numpy.int64 -> Python int"""
        numpy = pytest.importorskip("numpy", reason="需要 numpy")
        from src.rag import sanitize_metadata

        metadata = {"chunk_id": numpy.int64(5)}
        result = sanitize_metadata(metadata)

        assert isinstance(result["chunk_id"], int)
        assert result["chunk_id"] == 5

    def test_nested_dict_recursively_sanitized(self):
        """嵌套字典里的 numpy 类型也要被清理"""
        numpy = pytest.importorskip("numpy", reason="需要 numpy")
        from src.rag import sanitize_metadata

        metadata = {
            "outer": "normal_string",
            "nested": {"score": numpy.float32(0.8), "rank": numpy.int64(1)}
        }
        result = sanitize_metadata(metadata)

        assert isinstance(result["nested"]["score"], float)
        assert isinstance(result["nested"]["rank"], int)

    def test_plain_python_values_pass_through_unchanged(self):
        """普通 Python 类型不应该被修改"""
        from src.rag import sanitize_metadata

        metadata = {"source": "manual.pdf", "page": 3, "has_table": True}
        result = sanitize_metadata(metadata)
        assert result == metadata

    def test_empty_metadata_returns_empty_dict(self):
        """空 metadata 应返回空字典，不报错"""
        from src.rag import sanitize_metadata
        assert sanitize_metadata({}) == {}


# ==========================================
# get_llm 工厂函数测试（需要 langchain_ollama）
# ==========================================

_has_langchain_ollama = importlib.util.find_spec("langchain_ollama") is not None
_has_langchain_openai = importlib.util.find_spec("langchain_openai") is not None


@pytest.mark.skipif(
    not _has_langchain_ollama,
    reason="未安装 langchain_ollama，跳过 LLM 工厂测试"
)
class TestGetLlm:
    """测试 LLM 工厂函数（不会真正连接 Ollama 或 DeepSeek）"""

    def _reload_modules(self):
        import src.config as cfg
        importlib.reload(cfg)
        import src.rag as rag_module
        importlib.reload(rag_module)
        return rag_module

    def test_ollama_provider_calls_chat_ollama(self, monkeypatch):
        """LLM_PROVIDER=ollama 时，get_llm() 应构造 ChatOllama"""
        import sys
        monkeypatch.setenv("LLM_PROVIDER", "ollama")

        mock_ollama_module = MagicMock()
        MockChatOllama = MagicMock()
        mock_ollama_module.ChatOllama = MockChatOllama

        with patch.dict(sys.modules, {"langchain_ollama": mock_ollama_module}):
            rag_module = self._reload_modules()
            rag_module.get_llm()
            assert MockChatOllama.called

    def test_deepseek_missing_key_raises_valueerror(self, monkeypatch):
        """LLM_PROVIDER=deepseek 但没填 API Key 时，必须抛出 ValueError"""
        monkeypatch.setenv("LLM_PROVIDER", "deepseek")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "")
        rag_module = self._reload_modules()

        with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
            rag_module.get_llm()

    @pytest.mark.skipif(
        not _has_langchain_openai,
        reason="未安装 langchain_openai，跳过 DeepSeek 测试"
    )
    def test_deepseek_with_key_calls_chat_openai(self, monkeypatch):
        """LLM_PROVIDER=deepseek 且有 API Key 时，使用 ChatOpenAI"""
        monkeypatch.setenv("LLM_PROVIDER", "deepseek")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-12345")
        rag_module = self._reload_modules()

        with patch("langchain_openai.ChatOpenAI") as MockChatOpenAI:
            MockChatOpenAI.return_value = MagicMock()
            rag_module.get_llm()
            assert MockChatOpenAI.called

    def test_unknown_provider_raises_valueerror(self, monkeypatch):
        """传入不支持的 provider 值时，get_llm 应该报错"""
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        rag_module = self._reload_modules()

        from src.config import settings
        original = settings.LLM_PROVIDER
        object.__setattr__(settings, "LLM_PROVIDER", "unsupported_vendor")
        try:
            with pytest.raises(ValueError):
                rag_module.get_llm()
        finally:
            object.__setattr__(settings, "LLM_PROVIDER", original)
