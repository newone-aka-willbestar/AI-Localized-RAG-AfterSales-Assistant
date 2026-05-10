"""
测试 API 层的关键约束：
1. 无硬编码密钥 - src/api.py 的 API Key 必须来自 settings（读环境变量）
2. BM25 累积 bug 修复 - add_documents 追加而非替换
3. /health 接口存在且返回正确字段
"""
import pytest
import importlib
from unittest.mock import patch, MagicMock


class TestNoHardcodedSecrets:
    """验证 src/api.py 里没有硬编码密钥"""

    def test_api_py_no_hardcoded_key(self):
        """src/api.py 里不应该出现硬编码的 API Key 值"""
        with open("src/api.py", "r", encoding="utf-8") as f:
            content = f.read()

        forbidden = ["your-secret-key-2026", "your-secret-key"]
        for secret in forbidden:
            assert secret not in content, (
                f"src/api.py 里发现硬编码密钥: '{secret}'，"
                f"应通过 settings.API_KEY 从环境变量读取"
            )

    def test_api_key_comes_from_settings(self):
        """verify_api_key 应该对比 settings.API_KEY，而不是硬编码字符串"""
        with open("src/api.py", "r", encoding="utf-8") as f:
            content = f.read()

        assert "settings.API_KEY" in content, \
            "src/api.py 的鉴权逻辑应该读取 settings.API_KEY"


class TestBM25AccumulationFix:
    """验证 BM25 覆盖 bug 已修复"""

    @pytest.fixture
    def mock_rag(self):
        """
        创建一个 RAG 实例，mock 掉所有外部依赖。
        只测 all_documents 的累积逻辑，不涉及真实模型。

        用 RAG.__new__(RAG) 绕过 __init__，手动填充所有属性——
        这样完全不触碰真实的 VectorStore / LLM，也不需要 patch 懒加载导入。
        """
        pytest.importorskip("langchain_core", reason="需要 langchain_core")
        from src.rag import RAG
        instance = RAG.__new__(RAG)
        instance.vector_store = MagicMock()
        instance.llm = MagicMock()
        instance.final_retriever = MagicMock()
        instance.all_documents = []
        instance.cache_path = "/tmp/test_cache.pkl"
        return instance

    def test_add_documents_accumulates(self, mock_rag):
        """
        核心测试：add_documents 应该累积，不应该替换。

        场景：上传手册A（3块）→ 上传手册B（2块）
        期望：all_documents 共有 5 块
        错误旧行为：all_documents 只有最后的 2 块
        """
        from langchain_core.documents import Document

        docs_a = [Document(page_content=f"手册A第{i}段") for i in range(3)]
        docs_b = [Document(page_content=f"手册B第{i}段") for i in range(2)]

        with patch.object(mock_rag, "init_retriever"):
            mock_rag.add_documents(docs_a)
            assert len(mock_rag.all_documents) == 3, "上传手册A后应有3块"

            mock_rag.add_documents(docs_b)
            assert len(mock_rag.all_documents) == 5, \
                "上传手册B后应累积到5块，而不是只有2块"

    def test_add_documents_calls_init_retriever_with_all_docs(self, mock_rag):
        """
        add_documents 重建检索器时，必须传入完整的 all_documents，
        而不是只传新的 docs。
        """
        from langchain_core.documents import Document

        docs_a = [Document(page_content="手册A")]
        docs_b = [Document(page_content="手册B")]

        with patch.object(mock_rag, "init_retriever") as mock_init:
            mock_rag.add_documents(docs_a)
            mock_rag.add_documents(docs_b)

            # 最后一次调用 init_retriever 时，应该传入了全部2个文档
            last_call_docs = mock_init.call_args[0][0]
            assert len(last_call_docs) == 2, \
                "init_retriever 应该收到全部2个文档，而不是只有1个"


class TestHealthEndpoint:
    """验证 /health 接口"""

    def test_health_endpoint_exists(self):
        """/health 接口应该存在于 api.py"""
        with open("src/api.py", "r", encoding="utf-8") as f:
            content = f.read()

        assert '"/health"' in content or "'/health'" in content, \
            "api.py 里应该有 /health 接口"

    def test_health_returns_provider_info(self):
        """/health 接口应该返回 llm_provider 字段"""
        with open("src/api.py", "r", encoding="utf-8") as f:
            content = f.read()

        assert "llm_provider" in content, \
            "/health 接口应该返回 llm_provider 字段，方便运维确认当前使用的模型"


class TestAsyncFix:
    """验证异步阻塞问题已修复"""

    def test_no_sync_sleep_in_api(self):
        """
        api.py 里不应该有同步的 time.sleep()。
        同步 sleep 在 async 函数里会阻塞整个事件循环。
        """
        with open("src/api.py", "r", encoding="utf-8") as f:
            content = f.read()

        assert "time.sleep" not in content, \
            "api.py 里有 time.sleep()，这在 async 函数里会阻塞事件循环，" \
            "应改为 await asyncio.sleep()"

    def test_run_in_executor_used_for_rag_ask(self):
        """
        rag.ask() 是同步函数，在 async endpoint 里必须用
        run_in_executor 包装，否则会阻塞事件循环。
        """
        with open("src/api.py", "r", encoding="utf-8") as f:
            content = f.read()

        assert "run_in_executor" in content, \
            "api.py 里应该用 run_in_executor 包装同步的 rag.ask() 调用"
