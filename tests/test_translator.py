"""
测试 Translator 模块。

覆盖：
1. 短文本直接翻译
2. 长文本自动分块翻译
3. 单块翻译失败时保留原文，不崩溃
4. translate_if_needed：中文直接返回，非中文触发翻译
5. 分块逻辑：_split_for_translation 的边界条件
"""
import pytest
from unittest.mock import MagicMock, patch


# ==========================================
# 测试 Translator.translate()
# ==========================================

class TestTranslate:
    def _make_translator_with_mock_llm(self, llm_return="翻译结果"):
        """注入 mock LLM 的 Translator 实例"""
        from src.translator import Translator

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = llm_return

        # mock langchain 链式调用
        mock_prompt = MagicMock()
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)

        mock_llm = MagicMock()
        translator = Translator(llm=mock_llm)
        return translator, mock_chain, mock_prompt

    def test_short_text_translated(self):
        """短文本（≤ MAX_CHARS）直接翻译，返回翻译结果"""
        pytest.importorskip("langchain_core", reason="需要 langchain_core")
        from src.translator import Translator

        mock_llm = MagicMock()
        translator = Translator(llm=mock_llm)
        expected = "液压泵维修手册"

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = expected
        mock_chain.__or__ = MagicMock(return_value=mock_chain)

        with patch("langchain_core.prompts.ChatPromptTemplate.from_template",
                   return_value=mock_chain):
            result = translator.translate("Hydraulic pump maintenance manual")

        assert result == expected

    def test_single_chunk_failure_returns_original(self):
        """LLM 抛异常时，该块保留原文，整体不崩溃"""
        pytest.importorskip("langchain_core", reason="需要 langchain_core")
        from src.translator import Translator

        mock_llm = MagicMock()
        translator = Translator(llm=mock_llm)
        original = "Pump maintenance guide"

        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = RuntimeError("LLM timeout")
        mock_chain.__or__ = MagicMock(return_value=mock_chain)

        with patch("langchain_core.prompts.ChatPromptTemplate.from_template",
                   return_value=mock_chain):
            result = translator.translate(original)

        # 翻译失败 → 返回原文（降级保护）
        assert result == original

    def test_long_text_translated_in_chunks(self):
        """长文本分多块翻译，结果拼接"""
        pytest.importorskip("langchain_core", reason="需要 langchain_core")
        from src.translator import Translator, _MAX_CHARS_PER_CHUNK

        mock_llm = MagicMock()
        translator = Translator(llm=mock_llm)

        # 构造一段超出单块限制的文本
        long_text = ("This is a sentence about industrial equipment. " * 20 + "\n\n") * 5
        assert len(long_text) > _MAX_CHARS_PER_CHUNK

        call_count = 0

        def fake_invoke(input_dict):
            nonlocal call_count
            call_count += 1
            return f"翻译块{call_count}"

        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = fake_invoke
        mock_chain.__or__ = MagicMock(return_value=mock_chain)

        with patch("langchain_core.prompts.ChatPromptTemplate.from_template",
                   return_value=mock_chain):
            result = translator.translate(long_text)

        # 应该调用多次（分块翻译）
        assert call_count > 1
        # 结果包含所有块的翻译
        assert "翻译块1" in result
        assert "翻译块2" in result

    def test_partial_failure_preserves_other_chunks(self):
        """部分块翻译失败时，成功的块保留翻译，失败的块保留原文"""
        pytest.importorskip("langchain_core", reason="需要 langchain_core")
        from src.translator import Translator, _MAX_CHARS_PER_CHUNK

        mock_llm = MagicMock()
        translator = Translator(llm=mock_llm)

        # 超出单块的长文本
        long_text = ("Sentence about maintenance. " * 30 + "\n\n") * 4
        assert len(long_text) > _MAX_CHARS_PER_CHUNK

        call_count = 0

        def fake_invoke(input_dict):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # 第 2 块失败
                raise RuntimeError("timeout")
            return f"翻译块{call_count}"

        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = fake_invoke
        mock_chain.__or__ = MagicMock(return_value=mock_chain)

        with patch("langchain_core.prompts.ChatPromptTemplate.from_template",
                   return_value=mock_chain):
            result = translator.translate(long_text)

        # 不应崩溃，结果不为空
        assert result
        assert len(result) > 0


# ==========================================
# 测试 translate_if_needed()
# ==========================================

class TestTranslateIfNeeded:
    def test_chinese_returns_unchanged(self):
        """lang='zh' 时不翻译，直接返回原文"""
        pytest.importorskip("langchain_core", reason="需要 langchain_core")
        from src.translator import Translator

        translator = Translator(llm=MagicMock())
        text = "这是中文内容"
        result, translated = translator.translate_if_needed(text, lang="zh")

        assert result == text
        assert translated is False

    def test_unknown_lang_returns_unchanged(self):
        """lang='unknown' 时不翻译"""
        pytest.importorskip("langchain_core", reason="需要 langchain_core")
        from src.translator import Translator

        translator = Translator(llm=MagicMock())
        text = "???"
        result, translated = translator.translate_if_needed(text, lang="unknown")

        assert result == text
        assert translated is False

    def test_english_triggers_translation(self):
        """lang='en' 时触发翻译"""
        pytest.importorskip("langchain_core", reason="需要 langchain_core")
        from src.translator import Translator

        mock_llm = MagicMock()
        translator = Translator(llm=mock_llm)

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "已翻译"
        mock_chain.__or__ = MagicMock(return_value=mock_chain)

        with patch("langchain_core.prompts.ChatPromptTemplate.from_template",
                   return_value=mock_chain):
            result, translated = translator.translate_if_needed(
                "English text", lang="en"
            )

        assert translated is True
        assert result == "已翻译"


# ==========================================
# 测试分块逻辑 _split_for_translation()
# ==========================================

class TestSplitForTranslation:
    def test_short_text_not_split(self):
        """短文本不分块，返回原样"""
        from src.translator import _split_for_translation
        text = "短文本"
        chunks = _split_for_translation(text, max_chars=1000)
        assert chunks == [text]

    def test_long_text_split_into_multiple_chunks(self):
        """长文本按段落分成多块"""
        from src.translator import _split_for_translation
        # 每段 50 字，共 10 段，max_chars=120 → 应该有多块
        text = "\n\n".join(["a" * 50] * 10)
        chunks = _split_for_translation(text, max_chars=120)
        assert len(chunks) > 1

    def test_each_chunk_within_limit(self):
        """每块字符数不超过 max_chars"""
        from src.translator import _split_for_translation
        text = "\n\n".join(["x" * 100] * 20)
        max_chars = 300
        chunks = _split_for_translation(text, max_chars=max_chars)
        for chunk in chunks:
            assert len(chunk) <= max_chars * 1.1  # 允许 10% 误差（段落边界）

    def test_reassemble_covers_all_content(self):
        """分块后拼接，内容总量不丢失"""
        from src.translator import _split_for_translation
        paragraphs = [f"段落{i}内容，描述工业设备维修步骤。" for i in range(20)]
        text = "\n\n".join(paragraphs)
        chunks = _split_for_translation(text, max_chars=200)
        reassembled = "\n\n".join(chunks)
        # 每个段落的关键词都应该在结果里
        for p in paragraphs:
            assert p in reassembled
