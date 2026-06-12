"""
翻译流水线模块。

职责：
- 将非中文文本翻译为中文，使知识库能统一用中文检索
- 支持长文本分块翻译，避免超出 LLM 上下文窗口

设计决策：
1. 复用现有 LLM（DeepSeek / Ollama），不引入额外翻译 API
   - 优点：零新增外部依赖，成本可控
   - 适用：技术文档翻译，对话式 LLM 效果足够好
2. 分块翻译：每块 ≤ MAX_CHARS 字符，防止超出 token 限制
3. 降级保护：翻译失败返回原文，不中断主流程
4. 开关：TRANSLATION_ENABLED=false 时完全跳过，不增加延迟

使用场景：
- 英文学术论文与技术文档
- 日文、德文等其他语种文献
- 多语言网页内容入库前的统一处理
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# 每块最大字符数（约等于 ~500 tokens）
# 实际 token 数因语言而异，英文字符/token ≈ 4，中文 ≈ 1.5
# 保守设为 1500 字符，确保不超出大多数模型的单次输入限制
_MAX_CHARS_PER_CHUNK = 1500

_TRANSLATION_PROMPT = """\
你是一个专业的文档翻译专家。
请将以下文本翻译为简体中文。

要求：
1. 保留所有技术术语的准确性
2. 保留原文的段落结构和换行
3. 数字、单位、型号不要翻译，保持原样
4. 只输出翻译结果，不要加任何解释或前缀

原文：
{text}

翻译："""


def _split_for_translation(text: str, max_chars: int = _MAX_CHARS_PER_CHUNK) -> list[str]:
    """
    将长文本按段落切分为适合翻译的块。

    策略：
    1. 优先按 "\n\n"（段落）切分，保证语义完整
    2. 单个段落超出 max_chars 时，按句子（。！？.）进一步切分
    3. 保证每块 ≤ max_chars
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    current = []
    current_len = 0

    paragraphs = text.split("\n\n")
    for para in paragraphs:
        if current_len + len(para) + 2 <= max_chars:
            current.append(para)
            current_len += len(para) + 2
        else:
            if current:
                chunks.append("\n\n".join(current))
            # 如果单个段落本身超出限制，按句子再切
            if len(para) > max_chars:
                sub_chunks = _split_long_paragraph(para, max_chars)
                chunks.extend(sub_chunks[:-1])
                current = [sub_chunks[-1]]
                current_len = len(sub_chunks[-1])
            else:
                current = [para]
                current_len = len(para)

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _split_long_paragraph(para: str, max_chars: int) -> list[str]:
    """单个超长段落按句子边界切分"""
    import re
    # 中英文句子结束符
    sentences = re.split(r'(?<=[。！？.!?])\s*', para)
    chunks = []
    current = []
    current_len = 0

    for sent in sentences:
        if current_len + len(sent) <= max_chars:
            current.append(sent)
            current_len += len(sent)
        else:
            if current:
                chunks.append("".join(current))
            current = [sent]
            current_len = len(sent)

    if current:
        chunks.append("".join(current))

    return chunks if chunks else [para[:max_chars]]


class Translator:
    """
    文档翻译器，将任意语言文本翻译为简体中文。

    由 WebScraper 调用，不在模块顶层 import LLM（懒加载）。
    """

    def __init__(self, llm=None):
        """
        Args:
            llm: 可选，传入已有 LLM 实例（测试时注入 mock）
                 不传时自动调用 get_llm() 创建
        """
        self._llm: Optional[object] = llm

    def _get_llm(self):
        """懒加载 LLM，避免模块顶层 import"""
        if self._llm is None:
            from src.rag import get_llm
            self._llm = get_llm()
        return self._llm

    def translate(self, text: str) -> str:
        """
        将文本翻译为简体中文。

        对长文本自动分块，逐块翻译后拼接。
        任何块翻译失败时保留原文该块，保证整体不崩溃。

        Args:
            text: 待翻译文本（任意语言）

        Returns:
            翻译后的中文文本（或翻译失败时的原文）
        """
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        llm = self._get_llm()
        prompt = ChatPromptTemplate.from_template(_TRANSLATION_PROMPT)
        chain = prompt | llm | StrOutputParser()

        chunks = _split_for_translation(text)
        translated_chunks = []

        for i, chunk in enumerate(chunks):
            result = None
            try:
                result = chain.invoke(
                    {"text": chunk},
                    config={"run_name": f"Translator.chunk_{i+1}_of_{len(chunks)}"},
                )
            except TypeError as e:
                if "unexpected keyword argument 'config'" in str(e):
                    try:
                        result = chain.invoke({"text": chunk})
                    except Exception as inner_e:
                        logger.warning(
                            f"翻译块 {i+1}/{len(chunks)} 失败，保留原文: {inner_e}"
                        )
                        translated_chunks.append(chunk)
                        continue
                else:
                    raise
            except Exception as e:
                logger.warning(f"翻译块 {i+1}/{len(chunks)} 失败，保留原文: {e}")
                translated_chunks.append(chunk)
                continue

            try:
                translated_chunks.append(result.strip())
                logger.debug(f"翻译块 {i+1}/{len(chunks)} 完成")
            except Exception as e:
                logger.warning(f"翻译块 {i+1}/{len(chunks)} 失败，保留原文: {e}")
                translated_chunks.append(chunk)

        return "\n\n".join(translated_chunks)

    def translate_if_needed(self, text: str, lang: str) -> tuple[str, bool]:
        """
        仅在语言非中文时翻译，返回 (结果文本, 是否翻译了)。

        便捷方法，供 WebScraper 调用。
        """
        if lang == "zh" or lang == "unknown":
            return text, False
        translated = self.translate(text)
        return translated, True
