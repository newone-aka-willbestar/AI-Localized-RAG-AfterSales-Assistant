"""
RAG 核心引擎。

设计原则：
1. LLM 通过 get_llm_with_fallback() 获取，内置重试（指数退避）和降级（DeepSeek→Ollama）
2. 所有重型依赖（langchain、向量库）采用懒加载，不在模块顶层 import
3. sanitize_metadata 是纯函数，零依赖，随时可测试
"""
import logging
import os
import pickle
from typing import Dict, Any, List, Optional

from src.config import settings
from src.llm_factory import get_llm_with_fallback

logger = logging.getLogger(__name__)


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
        self.llm = get_llm_with_fallback()
        self.final_retriever = None
        # all_documents 是 BM25 的数据源，累积所有上传过的文档
        self.all_documents: List = []
        self.cache_path = os.path.join(settings.VECTORSTORE_PATH, "docs_cache.pkl")

        # HyDE：注入当前 LLM，检索前将问题变换为假设文档
        # 通过 HYDE_ENABLED=false 可以关闭，方便 A/B 对比效果
        if settings.HYDE_ENABLED:
            from src.hyde import HyDE
            self.hyde: Optional[HyDE] = HyDE(self.llm)
            logger.info("HyDE 检索增强已启用")
        else:
            self.hyde = None
            logger.info("HyDE 检索增强已关闭")

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
                # 必须同步恢复 all_documents：否则重启后再上传新文档时，
                # add_documents 会只用新文档重建 BM25，旧文档全部丢失
                self.all_documents = list(cached_docs)
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

            compressor = FlashrankRerank(
                model=settings.RERANK_MODEL_NAME,
                top_n=settings.RERANK_TOP_K,
            )
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

    def chitchat(self, question: str) -> Dict[str, Any]:
        """
        闲聊路径：跳过 RAG，直接用 LLM 回复。
        意图分类为 chitchat 时调用，不消耗检索资源。
        """
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        prompt = ChatPromptTemplate.from_template(
            "你是一个 AI 数字员工助手。请友好地回复用户的问候或闲聊，"
            "并适时引导他们上传文档或提出知识库相关问题。\n\n用户: {input}"
        )
        chain = prompt | self.llm | StrOutputParser()
        try:
            answer = chain.invoke({"input": question})
        except Exception as e:
            logger.warning(f"闲聊回复失败: {e}")
            answer = "您好！有什么知识库相关的问题需要我帮助吗？您也可以先上传文档或采集网页内容。"
        return {"answer": answer, "sources": [], "provider": settings.LLM_PROVIDER}

    def ask(self, question: str, history: str = "") -> Dict[str, Any]:
        """
        核心问答入口。

        Args:
            question: 用户当前轮次的问题
            history:  由 SessionMemory.get_history() 提供的历史对话文本。
                      非空时注入 prompt，让 LLM 能理解指代（"这个"、"刚才说的那个"）。
        返回格式: {"answer": str, "sources": list, "provider": str}
        """
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough
        from src.tracing import get_run_config

        if not self.final_retriever:
            return {
                "answer": "知识库为空，请先上传 PDF 文档或采集网页内容。",
                "sources": [],
                "provider": settings.LLM_PROVIDER
            }

        try:
            # HyDE 变换：将短句问题扩展为假设文档，提升向量检索召回率
            # hyde.generate() 内置降级保护，失败时自动返回原始问题
            retrieval_query = (
                self.hyde.generate(question)
                if self.hyde is not None
                else question
            )
            retrieved_docs = self.final_retriever.invoke(
                retrieval_query,
                config=get_run_config("retriever", metadata={"question": question}),
            )

            if not retrieved_docs:
                return {
                    "answer": "抱歉，知识库中未找到相关内容。",
                    "sources": [],
                    "provider": settings.LLM_PROVIDER
                }

            context = "\n\n".join([d.page_content for d in retrieved_docs])

            # 会话历史区块：有历史时插入，让 LLM 能理解指代词和上下文
            history_section = f"\n{history}\n" if history else ""

            prompt = ChatPromptTemplate.from_template(
                """你是一个专业的 AI 数字员工助手。请仅根据[参考信息]回答问题。
如果参考信息中没有相关内容，请直接说"知识库中暂无此信息"，禁止猜测。
{history}
[参考信息]
{context}

[用户问题]
{input}"""
            )

            chain = (
                {
                    "context": lambda x: context,
                    "history": lambda x: history_section,
                    "input": RunnablePassthrough(),
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )

            answer = chain.invoke(
                question,
                config=get_run_config(
                    "answer-generation",
                    metadata={
                        "question": question,
                        "doc_count": len(retrieved_docs),
                        "hyde_used": self.hyde is not None,
                        "has_history": bool(history),
                    },
                ),
            )

            sanitized_sources = []
            for doc in retrieved_docs:
                meta = sanitize_metadata(doc.metadata)
                content = doc.page_content
                # 摘要保留前 300 字：既供前端来源面板展示，也是评估忠实度时
                # LLM 裁判判断"答案声明是否有据可查"的上下文。截得过短（如 100 字）
                # 会让忠实度被系统性低估。
                meta["content_excerpt"] = content[:300] + ("..." if len(content) > 300 else "")
                sanitized_sources.append(meta)

            return {
                "answer": answer,
                "sources": sanitized_sources,
                "provider": settings.LLM_PROVIDER
            }

        except Exception as e:
            logger.error(f"问答链路异常: {e}", exc_info=True)
            raise
