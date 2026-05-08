"""
RAG 核心引擎。

设计原则：
1. LLM 提供商通过 get_llm() 工厂函数抽象，不耦合具体实现
2. 所有重型依赖（langchain、向量库）采用懒加载，不在模块顶层 import
3. sanitize_metadata 是纯函数，零依赖，随时可测试
"""
import logging
import os
import pickle
from typing import Dict, Any, List, Optional

from src.config import settings

logger = logging.getLogger(__name__)


def get_llm():
    """
    LLM 工厂函数：根据 config 中的 LLM_PROVIDER 返回对应的模型对象。

    调用方不需要关心用的是哪家模型，只管拿到对象调用即可。
    新增提供商只需在这里加一个 elif，其他代码完全不需要改动。
    """
    provider = settings.LLM_PROVIDER

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        logger.info(f"使用本地 Ollama 模型: {settings.OLLAMA_MODEL}")
        return ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=settings.TEMPERATURE,
            timeout=60
        )

    elif provider == "deepseek":
        from langchain_openai import ChatOpenAI
        if not settings.DEEPSEEK_API_KEY:
            raise ValueError(
                "LLM_PROVIDER=deepseek，但 DEEPSEEK_API_KEY 未设置。"
                "请在 .env 文件中添加: DEEPSEEK_API_KEY=你的密钥"
            )
        logger.info(f"使用 DeepSeek API 模型: {settings.DEEPSEEK_MODEL}")
        return ChatOpenAI(
            model=settings.DEEPSEEK_MODEL,
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
            temperature=settings.TEMPERATURE,
        )

    else:
        raise ValueError(
            f"不支持的 LLM_PROVIDER: '{provider}'。可选值: 'ollama' 或 'deepseek'"
        )


def sanitize_metadata(metadata: dict) -> dict:
    """
    将 metadata 里的 Numpy 类型转为 Python 原生类型。

    背景：FlashrankRerank 会在 metadata 里注入 numpy.float32 类型的相关性分数，
    直接 JSON 序列化会报 "Object of type float32 is not JSON serializable"。

    这是纯函数（无副作用，无外部依赖），提取为模块级函数方便单独测试。
    """
    result = {}
    for k, v in metadata.items():
        if hasattr(v, "item"):
            result[k] = v.item()
        elif isinstance(v, dict):
            result[k] = sanitize_metadata(v)
        else:
            result[k] = v
    return result


class RAG:
    """
    RAG 问答引擎。

    检索流程：
    向量检索(k=5) ─┐
                   ├→ EnsembleRetriever → FlashrankRerank(top4) → LLM生成
    BM25检索(k=5)  ─┘

    BM25 bug 修复说明：
    原版 init_retriever(new_docs) 每次只用新上传的文档重建 BM25，
    导致多次上传后只有最后一批文档参与关键词检索。
    修复方案：用 all_documents 列表累积所有文档，
    新增文档时追加到列表再整体重建 BM25。
    """

    def __init__(self, documents: Optional[List] = None):
        from src.vector_store import VectorStore
        self.vector_store = VectorStore()
        self.llm = get_llm()
        self.final_retriever = None
        # all_documents 是 BM25 的数据源，累积所有上传过的文档
        self.all_documents: List = []
        self.cache_path = os.path.join(settings.VECTORSTORE_PATH, "docs_cache.pkl")

        if documents:
            self.init_retriever(documents)
        else:
            self._try_load_cache()

    def _try_load_cache(self):
        """系统启动时，自动从本地磁盘恢复 BM25 检索器"""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    cached_docs = pickle.load(f)
                logger.info(f"发现本地缓存 ({len(cached_docs)} 个文本块)，正在恢复检索引擎...")
                self.init_retriever(cached_docs, save_cache=False)
            except Exception as e:
                logger.error(f"缓存恢复失败: {e}")

    def add_documents(self, new_docs: List) -> None:
        """
        追加新文档并重建检索器。

        这是修复 BM25 覆盖 bug 的核心方法：
        - 把新文档追加到 all_documents（累积，不替换）
        - 用完整的 all_documents 重建 BM25 索引
        - 向量库在 api.py 里单独写入，这里只管 BM25 这侧
        """
        self.all_documents.extend(new_docs)
        self.init_retriever(self.all_documents)
        logger.info(f"知识库累计文档数: {len(self.all_documents)} 块")

    def init_retriever(self, all_documents: List, save_cache: bool = True):
        """构建双路检索 + 精排架构（懒加载所有 langchain 依赖）"""
        from langchain_community.retrievers.bm25 import BM25Retriever
        from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank

        try:
            try:
                from langchain.retrievers.ensemble import EnsembleRetriever
                from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
            except ImportError:
                from langchain_community.retrievers.ensemble import EnsembleRetriever
                from langchain_community.retrievers.contextual_compression import ContextualCompressionRetriever

            vector_retriever = self.vector_store.get_retriever()
            bm25_retriever = BM25Retriever.from_documents(all_documents)
            bm25_retriever.k = 5

            ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[0.6, 0.4]
            )

            compressor = FlashrankRerank()
            self.final_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=ensemble_retriever
            )

            if save_cache:
                os.makedirs(settings.VECTORSTORE_PATH, exist_ok=True)
                with open(self.cache_path, "wb") as f:
                    pickle.dump(all_documents, f)
                logger.info("文档已缓存，下次启动无需重新上传")

            logger.info("检索引擎就绪")

        except Exception as e:
            logger.error(f"检索器构建失败: {e}")
            raise

    def ask(self, question: str) -> Dict[str, Any]:
        """
        核心问答入口。
        返回格式: {"answer": str, "sources": list, "provider": str}
        """
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough

        if not self.final_retriever:
            return {
                "answer": "知识库为空，请先上传 PDF 文档或采集网页内容。",
                "sources": [],
                "provider": settings.LLM_PROVIDER
            }

        try:
            retrieved_docs = self.final_retriever.invoke(question)

            if not retrieved_docs:
                return {
                    "answer": "抱歉，知识库中未找到相关内容。",
                    "sources": [],
                    "provider": settings.LLM_PROVIDER
                }

            context = "\n\n".join([d.page_content for d in retrieved_docs])

            prompt = ChatPromptTemplate.from_template(
                """你是一个专业的工业售后专家。请仅根据[参考信息]回答问题。
如果参考信息中没有相关内容，请直接说"知识库中暂无此信息"，禁止猜测。

[参考信息]
{context}

[用户问题]
{input}"""
            )

            chain = (
                {"context": lambda x: context, "input": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )

            answer = chain.invoke(question)

            sanitized_sources = []
            for doc in retrieved_docs:
                meta = sanitize_metadata(doc.metadata)
                meta["content_excerpt"] = doc.page_content[:100] + "..."
                sanitized_sources.append(meta)

            return {
                "answer": answer,
                "sources": sanitized_sources,
                "provider": settings.LLM_PROVIDER
            }

        except Exception as e:
            logger.error(f"问答链路异常: {e}")
            return {
                "answer": "服务繁忙，请稍后再试。",
                "sources": [],
                "provider": settings.LLM_PROVIDER
            }
