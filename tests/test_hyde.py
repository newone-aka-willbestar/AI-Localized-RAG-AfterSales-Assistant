"""
测试 HyDE 模块。

覆盖三个场景：
1. 正常生成假设文档
2. LLM 异常时降级返回原始问题
3. RAG.ask() 根据 HYDE_ENABLED 开关决定是否使用 HyDE
"""
import pytest
from unittest.mock import MagicMock, patch


# ==========================================
# 测试 HyDE.generate()
# ==========================================

class TestHyDeGenerate:
    """
    测试 HyDE.generate() 的三种情况。

    因为 langchain 依赖在 generate() 内部懒加载，
    patch 目标必须指向源头模块（langchain_core），而不是 src.hyde。
    技巧：让 mock chain 的 invoke() 直接返回我们想要的字符串，
    绕过 prompt | llm | parser 的复杂 | 运算符链。
    """

    def _make_chain_mock(self, return_value=None, side_effect=None):
        """
        构造一个支持 langchain | 运算符链的 mock。

        langchain 的链式调用是 (prompt | llm | parser).invoke(input)，
        每个 | 都返回一个新对象。这里让所有 | 都返回同一个 mock_chain，
        最后 invoke() 的行为由 return_value / side_effect 控制。
        """
        mock_chain = MagicMock()
        if side_effect:
            mock_chain.invoke.side_effect = side_effect
        else:
            mock_chain.invoke.return_value = return_value
        # __or__ 模拟 | 运算符，始终返回自身（链式）
        mock_chain.__or__ = MagicMock(return_value=mock_chain)
        return mock_chain

    def test_generate_returns_hypothetical_doc(self):
        """正常路径：generate() 返回 LLM 生成的假设文档"""
        pytest.importorskip("langchain_core", reason="需要 langchain_core")
        from src.hyde import HyDE

        hyde = HyDE(llm=MagicMock())
        expected = "[故障现象] 设备无法启动。\n[可能原因] 电源故障。"
        mock_chain = self._make_chain_mock(return_value=expected)

        with patch("langchain_core.prompts.ChatPromptTemplate.from_template",
                   return_value=mock_chain):
            result = hyde.generate("设备不启动怎么办？")

        assert result == expected

    def test_generate_fallback_on_exception(self):
        """
        降级路径：LLM 抛异常时返回原始问题，不崩溃。
        保证即使 HyDE 失败，主问答流程仍然正常。
        """
        pytest.importorskip("langchain_core", reason="需要 langchain_core")
        from src.hyde import HyDE

        hyde = HyDE(llm=MagicMock())
        original_question = "设备保修期是多久？"
        mock_chain = self._make_chain_mock(side_effect=RuntimeError("LLM 超时"))

        with patch("langchain_core.prompts.ChatPromptTemplate.from_template",
                   return_value=mock_chain):
            result = hyde.generate(original_question)

        assert result == original_question  # 降级到原始问题

    def test_generate_strips_whitespace(self):
        """generate() 去除 LLM 返回结果的首尾空白"""
        pytest.importorskip("langchain_core", reason="需要 langchain_core")
        from src.hyde import HyDE

        hyde = HyDE(llm=MagicMock())
        mock_chain = self._make_chain_mock(return_value="  \n假设文档内容\n  ")

        with patch("langchain_core.prompts.ChatPromptTemplate.from_template",
                   return_value=mock_chain):
            result = hyde.generate("测试问题")

        assert result == "假设文档内容"


# ==========================================
# 测试 RAG.ask() 与 HyDE 的集成
# ==========================================

class TestRagHydeIntegration:
    """
    验证 RAG 根据 HYDE_ENABLED 开关正确决定是否调用 HyDE。
    不启动真实 LLM 或 Qdrant，全部 mock。
    """

    def _make_mock_rag(self, hyde_enabled: bool):
        """构造一个完全 mock 的 RAG 实例"""
        from src.rag import RAG
        instance = RAG.__new__(RAG)
        instance.llm = MagicMock()
        instance.all_documents = []
        instance.cache_path = "/tmp/test.pkl"

        if hyde_enabled:
            mock_hyde = MagicMock()
            mock_hyde.generate.return_value = "假设文档：[故障现象]..."
            instance.hyde = mock_hyde
        else:
            instance.hyde = None

        # 设置一个假的 retriever，返回一个空列表（触发"未找到"分支）
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        instance.final_retriever = mock_retriever

        return instance

    def test_ask_uses_hyde_when_enabled(self):
        """
        HYDE_ENABLED=True 时，ask() 应该调用 hyde.generate() 并用其结果检索。
        """
        pytest.importorskip("langchain_core", reason="需要 langchain_core")
        rag = self._make_mock_rag(hyde_enabled=True)
        question = "设备保修期是多久？"

        rag.ask(question)

        # HyDE.generate 被调用了一次，传入了原始问题
        rag.hyde.generate.assert_called_once_with(question)

        # retriever 收到的是 HyDE 生成的假设文档，而不是原始问题
        retriever_input = rag.final_retriever.invoke.call_args[0][0]
        assert retriever_input == "假设文档：[故障现象]..."

    def test_ask_skips_hyde_when_disabled(self):
        """
        HYDE_ENABLED=False（hyde=None）时，ask() 直接用原始问题检索，不调用 HyDE。
        """
        pytest.importorskip("langchain_core", reason="需要 langchain_core")
        rag = self._make_mock_rag(hyde_enabled=False)
        question = "设备保修期是多久？"

        rag.ask(question)

        # retriever 收到的是原始问题
        retriever_input = rag.final_retriever.invoke.call_args[0][0]
        assert retriever_input == question

    def test_ask_still_works_when_hyde_falls_back(self):
        """
        HyDE 降级返回原始问题时，ask() 的主流程不应中断。
        """
        pytest.importorskip("langchain_core", reason="需要 langchain_core")
        rag = self._make_mock_rag(hyde_enabled=True)

        # 让 HyDE 降级，返回原始问题
        rag.hyde.generate.return_value = "原始问题"

        result = rag.ask("原始问题")

        # 应该正常返回结果（知识库为空的提示）
        assert "answer" in result
        assert "sources" in result
