"""
报表生成模块。

职责：
从已入库的 PDF / 网页内容中，按指定模板生成结构化 Markdown 报告。
面向论文写作场景，支持五种报告类型：

  summary     文献摘要   —— 提炼单篇/多篇文献的核心内容
  keypoints   要点提炼   —— 按章节列出关键论点和数据
  review      文献综述   —— 综合多篇文献，输出可直接引用的综述段落
  comparison  对比分析   —— 横向比较多份文献的方法、结论、局限性
  custom      自定义     —— 用户自己写指令，系统按指令生成

输出格式：Markdown 字符串，前端直接渲染并提供下载。
"""
import logging

logger = logging.getLogger(__name__)

# ==========================================
# 各类型的 Prompt 模板
# ==========================================

_PROMPTS = {
    "summary": """\
你是一位学术助理。请根据以下[参考文献内容]，为用户生成一份结构清晰的**文献摘要报告**。

要求：
1. 用 Markdown 格式输出
2. 包含以下章节：研究背景、核心问题、研究方法、主要发现、结论与局限性
3. 语言简洁学术，每个章节 3-5 句话
4. 若参考内容不足以支撑某章节，请注明"资料不足"

[参考文献内容]
{context}

[用户主题]
{topic}

请生成报告：""",

    "keypoints": """\
你是一位学术助理。请根据以下[参考文献内容]，提炼出**核心要点清单**。

要求：
1. 用 Markdown 格式输出
2. 按逻辑分组，每组用二级标题
3. 每条要点一行，用"- "开头，附上关键数据或引用依据（如有）
4. 最后加一个"⚠️ 注意事项"章节，列出文献中提到的局限性或争议点

[参考文献内容]
{context}

[用户主题]
{topic}

请生成要点清单：""",

    "review": """\
你是一位学术写作助理。请根据以下[参考文献内容]，撰写一段可直接用于论文的**文献综述段落**。

要求：
1. 用 Markdown 格式，包含：综述正文 + 研究空白 + 本研究定位三个部分
2. 综述正文按时间或主题脉络梳理已有研究，指出各研究的贡献与不足
3. 研究空白指出现有文献尚未解决的问题
4. 本研究定位说明当前研究如何填补这一空白（可留空，用[待填写]标注）
5. 语言符合学术规范，避免口语化表达

[参考文献内容]
{context}

[用户主题]
{topic}

请生成文献综述：""",

    "comparison": """\
你是一位学术助理。请根据以下[参考文献内容]，生成一份**多文献对比分析报告**。

要求：
1. 用 Markdown 格式输出
2. 先生成一个对比表格（Markdown 表格语法），列：文献/来源、研究方法、主要结论、局限性
3. 表格之后，写 2-3 段对比分析文字，重点指出各文献的异同和互补性
4. 最后给出综合评价

[参考文献内容]
{context}

[用户主题]
{topic}

请生成对比分析：""",

    "custom": """\
你是一位学术助理。请根据以下[参考文献内容]，按照用户的[自定义指令]生成报告。

[参考文献内容]
{context}

[用户主题]
{topic}

[自定义指令]
{custom_instruction}

请生成报告：""",
}

REPORT_TYPES = {
    "summary":    "📄 文献摘要",
    "keypoints":  "🔑 要点提炼",
    "review":     "📚 文献综述",
    "comparison": "⚖️ 对比分析",
    "custom":     "✏️ 自定义",
}


class ReportGenerator:
    """
    报表生成器。

    依赖 RAG 的检索能力召回相关文献内容，
    再用 LLM 按模板生成结构化 Markdown 报告。
    """

    def __init__(self, rag):
        """
        Args:
            rag: RAG 实例，用于检索知识库内容
        """
        self._rag = rag

    def generate(
        self,
        topic: str,
        report_type: str = "summary",
        custom_instruction: str = "",
        max_sources: int = 8,
    ) -> dict:
        """
        生成报告。

        Args:
            topic:              报告主题（用于检索 + 注入 prompt）
            report_type:        报告类型，见 REPORT_TYPES
            custom_instruction: 仅 report_type="custom" 时有效
            max_sources:        最多使用的文献片段数

        Returns:
            {
              "report":   str,   # Markdown 报告正文
              "sources":  list,  # 检索到的来源列表
              "type":     str,   # 报告类型标签
              "topic":    str,
            }
        """
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from src.tracing import get_run_config

        if report_type not in _PROMPTS:
            report_type = "summary"

        if not self._rag.final_retriever:
            return {
                "report": "⚠️ **知识库为空**，请先上传 PDF 文档或抓取网页内容后再生成报告。",
                "sources": [],
                "type": REPORT_TYPES.get(report_type, report_type),
                "topic": topic,
            }

        try:
            # 1. 检索相关文献内容
            retrieved_docs = self._rag.final_retriever.invoke(
                topic,
                config=get_run_config(
                    "report-retrieval",
                    metadata={"topic": topic, "report_type": report_type},
                ),
            )

            if not retrieved_docs:
                return {
                    "report": "⚠️ **知识库中未找到与该主题相关的内容**，请尝试换个关键词或先上传相关文献。",
                    "sources": [],
                    "type": REPORT_TYPES.get(report_type, report_type),
                    "topic": topic,
                }

            # 限制来源数量，避免 context 过长
            docs_to_use = retrieved_docs[:max_sources]
            context = "\n\n---\n\n".join(
                f"【来源：{doc.metadata.get('source', '未知')}】\n{doc.page_content}"
                for doc in docs_to_use
            )

            # 2. 构建 prompt 并生成报告
            template = _PROMPTS[report_type]
            prompt = ChatPromptTemplate.from_template(template)

            chain = prompt | self._rag.llm | StrOutputParser()

            invoke_input = {
                "context": context,
                "topic": topic,
                "custom_instruction": custom_instruction or "请生成一份全面的分析报告",
            }

            report_text = chain.invoke(
                invoke_input,
                config=get_run_config(
                    "report-generation",
                    metadata={
                        "topic": topic,
                        "report_type": report_type,
                        "doc_count": len(docs_to_use),
                    },
                ),
            )

            # 3. 整理来源信息
            from src.rag import sanitize_metadata
            sources = []
            for doc in docs_to_use:
                meta = sanitize_metadata(doc.metadata)
                content = doc.page_content
                meta["content_excerpt"] = content[:300] + ("..." if len(content) > 300 else "")
                sources.append(meta)

            logger.info(
                f"报告生成完毕 | 类型={report_type} | 主题={topic[:30]} | 来源数={len(sources)}"
            )

            return {
                "report": report_text,
                "sources": sources,
                "type": REPORT_TYPES.get(report_type, report_type),
                "topic": topic,
            }

        except Exception as e:
            logger.error(f"报告生成失败: {e}", exc_info=True)
            return {
                "report": f"⚠️ **报告生成失败**，请稍后重试。\n\n错误信息：`{e}`",
                "sources": [],
                "type": REPORT_TYPES.get(report_type, report_type),
                "topic": topic,
            }
