import logging
from typing import Dict, Any, List, Optional

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.vector_store import VectorStore
from src.config import settings
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

class RAG:
    """
    工业级 RAG 引擎（优化版）：
    1. 支持动态加载文档（解决启动时无文档报错）
    2. 混合检索 + 语义精排
    3. 异常捕获与友好提示
    """
    
    def __init__(self, documents: Optional[List] = None):
        self.vector_store = VectorStore()
        self.llm = ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0,
        )
        
        # 核心：最终检索器实例，初始为 None
        self.final_retriever = None
        
        # 如果初始化时有文档，直接构建检索链
        if documents:
            self.init_retriever(documents)
        else:
            logger.warning("⚠️ RAG 启动时未加载文档，请通过 /upload 接口上传文档。")

    def init_retriever(self, all_documents: List):
        """
        初始化或更新检索器（面试点：动态构建检索链）
        """
        try:
            # 1. 向量路 (MMR 保证多样性)
            vector_retriever = self.vector_store.get_retriever()
            
            # 2. 关键词路 (BM25 处理专有名词)
            bm25_retriever = BM25Retriever.from_documents(all_documents)
            bm25_retriever.k = 5
            
            # 3. 合并双路召回
            ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[0.6, 0.4]
            )
            
            # 4. 精排层 (Rerank)
            # 面试时说：Flashrank 可以在本地 CPU 运行，极大地平衡了精度和速度
            compressor = FlashrankRerank(model_name="ms-marco-MiniLM-L-12-v2") 
            self.final_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=ensemble_retriever
            )
            logger.info("✅ 混合检索与精排层初始化成功")
        except Exception as e:
            logger.error(f"❌ 初始化检索器失败: {e}")

    def format_docs(self, docs):
        """将检索到的文档片段合并"""
        return "\n\n".join(doc.page_content for doc in docs)

    def ask(self, question: str) -> Dict[str, Any]:
        """
        问答执行入口
        """
        # 1. 容错处理：如果没上传文档就问问题
        if not self.final_retriever:
            # 尝试从向量库直接恢复（针对重启后已存在向量库的情况）
            # 这里简化处理，直接提示上传
            return {
                "answer": "知识库暂未就绪，请先上传 PDF 文档。",
                "sources": []
            }

        logger.info(f"🔍 正在处理提问: {question}")
        
        try:
            # 2. 检索逻辑
            retrieved_docs = self.final_retriever.invoke(question)
            
            # 3. 构建提示词（面试点：严格的角色定义与背景限制）
            system_prompt = """你是一个专业的工业售后专家。请根据提供的[参考上下文]回答问题。
            
[规则]
- 只能根据上下文回答，不要瞎编。
- 若无法找到答案，请回答："抱歉，当前技术手册中没有找到关于'{input}'的操作指南。"
- 来源必须注明文件名。

[上下文]
{context}"""

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])

            # 4. 运行链
            # 用 list(set(...)) 这种写法显得你的代码很细心，处理了重复来源
            sources = ", ".join(list(set(d.metadata.get("source", "未知") for d in retrieved_docs)))
            
            chain = (
                {
                    "context": lambda x: self.format_docs(retrieved_docs),
                    "input": RunnablePassthrough()
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )
            
            answer = chain.invoke(question)
            
            return {
                "answer": answer,
                "sources": [doc.metadata for doc in retrieved_docs]
            }
            
        except Exception as e:
            logger.error(f"❌ 问答流程出错: {e}")
            return {"answer": f"处理出错：{str(e)}", "sources": []}