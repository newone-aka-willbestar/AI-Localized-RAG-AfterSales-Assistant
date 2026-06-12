"""
文档解析模块。

支持格式：
  .pdf   → pymupdf4llm 转 Markdown（保留标题层级）+ fitz 兜底
  .docx  → python-docx 提取正文段落、标题、表格

解析后统一转为 LangChain Document 列表，交给下游分片器处理。
"""
import logging
from pathlib import Path
from typing import List

import fitz
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import settings

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


class DocumentLoader:
    def __init__(self):
        self.chunk_size = getattr(settings, "CHUNK_SIZE", 800)
        self.chunk_overlap = getattr(settings, "CHUNK_OVERLAP", 150)

        headers_to_split_on = [("#", "H1"), ("##", "H2")]
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""],
        )

    def load_and_split(self, file_path: str) -> List[Document]:
        file_path_obj = Path(file_path)
        suffix = file_path_obj.suffix.lower()
        logger.info(f"开始解析文档: {file_path_obj.name} (格式: {suffix})")

        if suffix == ".pdf":
            text = self._load_pdf(file_path)
        elif suffix == ".docx":
            text = self._load_docx(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {suffix}，当前支持: {SUPPORTED_EXTENSIONS}")

        # 统一分片流程
        header_splits = self.header_splitter.split_text(text)
        final_splits = self.text_splitter.split_documents(header_splits)

        for i, doc in enumerate(final_splits):
            doc.metadata.update({
                "source": file_path_obj.name,
                "chunk_id": i,
                "file_type": suffix.lstrip("."),
                "has_table": "|" in doc.page_content,
            })

        logger.info(f"解析完成: {file_path_obj.name} → {len(final_splits)} 个文本块")
        return final_splits

    # ── PDF 解析 ──────────────────────────────────────────

    def _load_pdf(self, file_path: str) -> str:
        """
        PDF → Markdown 字符串。
        主路径：pymupdf4llm（保留标题 / 列表 / 表格结构）
        备用路径：fitz 纯文本提取（pymupdf4llm 失败时自动降级）
        """
        try:
            md_text = pymupdf4llm.to_markdown(str(file_path))
            if not md_text or len(md_text) < 10:
                raise ValueError("Markdown 提取内容过少")
            return md_text
        except Exception as e:
            logger.warning(f"高级 PDF 解析失败，启用基础文本提取: {e}")
            doc = fitz.open(str(file_path))
            try:
                text = "\n\n".join([page.get_text() for page in doc])
            finally:
                doc.close()
            return text

    # ── Word 解析 ─────────────────────────────────────────

    def _load_docx(self, file_path: str) -> str:
        """
        Word .docx → Markdown 字符串。

        提取规则：
        - Heading 1/2/3 样式 → ## / ### / #### 标题
        - 普通段落 → 原文保留
        - 表格 → Markdown 表格语法（首行为表头）
        - 忽略页眉、页脚、图片（只取文字）

        为什么转成 Markdown？
          让 MarkdownHeaderTextSplitter 能按标题层级分片，
          和 PDF 走相同的下游处理流程，不需要额外分支。
        """
        try:
            from docx import Document as DocxDocument
        except ImportError:
            raise ImportError(
                "python-docx 未安装，请运行: pip install python-docx>=1.1.0"
            )

        docx = DocxDocument(file_path)
        lines: List[str] = []

        # 按文档顺序遍历段落和表格（docx 的 body 元素顺序即原始顺序）
        for element in docx.element.body:
            tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag

            if tag == "p":
                # 段落
                para_text = element.text_content() if hasattr(element, "text_content") else ""
                # 用 lxml 的 itertext 获取完整文本（含 run 中的内容）
                para_text = "".join(element.itertext()).strip()
                if not para_text:
                    continue

                # 识别标题样式
                style_name = ""
                pPr = element.find(f".//{{{element.nsmap.get('w', 'http://schemas.openxmlformats.org/wordprocessingml/2006/main')}}}pStyle")
                if pPr is not None:
                    style_name = pPr.get(f"{{{element.nsmap.get('w', 'http://schemas.openxmlformats.org/wordprocessingml/2006/main')}}}val", "")

                if style_name.startswith("Heading1") or style_name == "1":
                    lines.append(f"# {para_text}")
                elif style_name.startswith("Heading2") or style_name == "2":
                    lines.append(f"## {para_text}")
                elif style_name.startswith("Heading3") or style_name == "3":
                    lines.append(f"### {para_text}")
                else:
                    lines.append(para_text)

            elif tag == "tbl":
                # 表格 → Markdown 表格
                table_lines = self._docx_table_to_markdown(element)
                lines.extend(table_lines)
                lines.append("")  # 表格后空行

        return "\n\n".join(lines)

    def _docx_table_to_markdown(self, tbl_element) -> List[str]:
        """
        将 docx XML 表格元素转为 Markdown 表格行列表。
        首行视为表头，插入分隔线。
        """
        try:
            ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            rows = tbl_element.findall(f".//{{{ns}}}tr")
            if not rows:
                return []

            md_rows: List[str] = []
            for i, row in enumerate(rows):
                cells = row.findall(f".//{{{ns}}}tc")
                cell_texts = ["".join(c.itertext()).strip() for c in cells]
                md_rows.append("| " + " | ".join(cell_texts) + " |")
                if i == 0:
                    # 表头分隔线
                    md_rows.append("| " + " | ".join(["---"] * len(cell_texts)) + " |")
            return md_rows
        except Exception as e:
            logger.warning(f"表格转换失败，跳过: {e}")
            return []
