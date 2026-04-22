import logging
import hashlib
import os
from pathlib import Path
from typing import List

import fitz  
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import settings

logger = logging.getLogger(__name__)

class DocumentLoader:
    def __init__(self):
        self.chunk_size = getattr(settings, "CHUNK_SIZE", 800)
        self.chunk_overlap = getattr(settings, "CHUNK_OVERLAP", 150)

        # 按标题拆分（针对 Markdown 格式）
        headers_to_split_on = [("#", "H1"), ("##", "H2")]
        self.header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
        
        # 递归拆分
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )

    def load_and_split(self, file_path: str) -> List[Document]:
        file_path_obj = Path(file_path)
        logger.info(f"🚀 开始解析文档: {file_path_obj.name}")

        md_text = ""
        try:
            # 尝试高级 Markdown 转换（可能会触发 ONNX 报错）
            # 关键：不传任何 layout 参数，降低报错概率
            md_text = pymupdf4llm.to_markdown(str(file_path))
            if not md_text or len(md_text) < 10:
                raise ValueError("Markdown 提取内容过少")
        except Exception as e:
            logger.warning(f"⚠️ 高级解析引擎异常，正在启动基础文本提取备份: {e}")
            # 【备份方案】直接使用基础 fitz 提取文本，保证系统绝对可用
            doc = fitz.open(str(file_path))
            md_text = "\n\n".join([page.get_text() for page in doc])
            doc.close() # 显式关闭句柄，防止 Windows 占用 

        # 执行分片
        header_splits = self.header_splitter.split_text(md_text)
        final_splits = self.text_splitter.split_documents(header_splits)

        # 元数据注入
        for i, doc in enumerate(final_splits):
            doc.metadata.update({
                "source": file_path_obj.name,
                "chunk_id": i,
                "has_table": "|" in doc.page_content
            })

        return final_splits