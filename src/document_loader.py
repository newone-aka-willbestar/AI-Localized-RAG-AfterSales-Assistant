import logging
import hashlib
from pathlib import Path
from typing import List

# 核心升级：使用 pymupdf4llm 将 PDF 完美转为 Markdown（保留表格结构）
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import settings

logger = logging.getLogger(__name__)

class DocumentLoader:
    """企业级文档解析器：支持表格还原、Markdown 层次拆分与语义增强"""

    def __init__(self):
        self.chunk_size = getattr(settings, "CHUNK_SIZE", 1000)
        self.chunk_overlap = getattr(settings, "CHUNK_OVERLAP", 200)

        # 1. 第一层拆分：按 Markdown 标题层级拆分，保证章节完整性
        # 面试亮点：这能让每个 Chunk 自动带上它所属的章节信息，避免“断章取义”
        headers_to_split_on = [
            ("#", "Header_1"),
            ("##", "Header_2"),
            ("###", "Header_3"),
        ]
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, 
            strip_headers=False
        )

        # 2. 第二层拆分：在章节内部按长度递归拆分
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )

    def _generate_file_hash(self, file_path: Path) -> str:
        """生成文件唯一标识，用于增量更新处理（生产环境必备）"""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def load_and_split(self, file_path: str) -> List[Document]:
        """PDF -> Markdown -> 语义切片 -> 元数据注入"""
        file_path = Path(file_path)
        logger.info(f"🚀 正在启动深度解析: {file_path.name}")

        try:
            md_text = pymupdf4llm.to_markdown(
                str(file_path), 
                show_progress=False
            )
        except Exception as e:
            logger.warning(f"⚠️ 高级解析失败，正在降级为基础解析: {e}")
            import fitz  # PyMuPDF
            doc = fitz.open(str(file_path))
            md_text = "" 
            for page in doc:
                md_text += page.get_text()
            doc.close()

        # A. 布局识别解析：将 PDF 转为 Markdown
        # 这种方式处理表格的效果远好于普通的 PyMuPDFLoader
        md_text = pymupdf4llm.to_markdown(str(file_path))
        
        # B. 基于标题层级的逻辑拆分
        header_splits = self.header_splitter.split_text(md_text)

        # C. 细粒度递归切片
        final_splits = self.text_splitter.split_documents(header_splits)

        # D. 工业级元数据注入
        file_hash = self._generate_file_hash(file_path)
        
        for i, doc in enumerate(final_splits):
            # 记录该块是否包含表格 (Markdown 表格通常包含 |---| )
            has_table = "|" in doc.page_content and "---" in doc.page_content
            
            doc.metadata.update({
                "source": file_path.name,
                "file_hash": file_hash,
                "chunk_id": i,
                "has_table": has_table,
                "content_length": len(doc.page_content),
                # 继承来自 MarkdownHeaderTextSplitter 的层级信息
                "section_hierarchy": {
                    "h1": doc.metadata.get("Header_1", "未知"),
                    "h2": doc.metadata.get("Header_2", "未知")
                }
            })

        logger.info(f"🎯 解析完成：生成 {len(final_splits)} 个语义块 (表格检测: {'有' if any(d.metadata['has_table'] for d in final_splits) else '无'})")
        return final_splits