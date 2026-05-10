"""
向量数据库封装层。

从 ChromaDB 迁移到 Qdrant 的说明：
- ChromaDB 是进程内嵌入式库，并发写入容易出文件锁，生产环境不可靠
- Qdrant 是独立 HTTP 服务（Docker 启动），支持并发读写、水平扩展
- 对外接口（add_documents / get_retriever）完全不变，rag.py 和 api.py 零改动

依赖服务：docker compose up -d  （启动本地 Qdrant）
"""
import logging
import os
from typing import List, Optional

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.config import settings

# 国内镜像加速 Hugging Face 下载
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# langchain_huggingface 和 langchain_qdrant 在旧版 langchain_core 下顶层 import
# 会触发 pydantic_v1 兼容问题，统一移到函数内部懒加载

logger = logging.getLogger(__name__)


def _build_embeddings():
    """
    加载 Embedding 模型（懒加载：在此函数内部才 import 重型依赖）。

    优先使用 models/ 目录下的离线模型，不存在则从镜像站下载。
    抽出独立函数方便测试时 mock。
    """
    from langchain_huggingface import HuggingFaceEmbeddings  # 懒加载，避免 pydantic_v1 问题

    local_path = os.path.abspath(settings.EMBEDDING_MODEL_PATH)
    vocab_file = os.path.join(local_path, "vocab.txt")

    if os.path.exists(vocab_file):
        model_name = local_path
        logger.info(f"使用本地 Embedding 模型: {model_name}")
    else:
        model_name = "BAAI/bge-small-zh-v1.5"
        logger.warning(f"本地模型不存在 ({local_path})，正在通过镜像站下载...")

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


class VectorStore:
    """
    Qdrant 向量库封装。

    设计要点：
    1. 接受可选的 QdrantClient 注入（方便测试传入内存客户端）
    2. 启动时自动检测 Collection 是否存在，不存在则创建
    3. 对外接口与原 ChromaDB 版本完全一致

    本地开发：先运行 docker compose up -d 启动 Qdrant
    生产部署：设置 QDRANT_HOST / QDRANT_PORT 指向云端 Qdrant
    """

    def __init__(self, client: Optional[QdrantClient] = None):
        # 1. Embedding 模型
        self.embeddings = _build_embeddings()

        # 2. Qdrant 客户端（支持外部注入，方便单元测试）
        if client is not None:
            self.client = client
        else:
            self.client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
            )
            logger.info(
                f"已连接 Qdrant: {settings.QDRANT_HOST}:{settings.QDRANT_PORT}"
            )

        # 3. 确保 Collection 存在（首次启动时自动建表）
        self._ensure_collection()

        # 4. LangChain Qdrant 封装（懒加载）
        from langchain_qdrant import QdrantVectorStore  # 懒加载，避免 pydantic_v1 问题
        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=settings.QDRANT_COLLECTION,
            embedding=self.embeddings,
        )

    def _ensure_collection(self) -> None:
        """
        检查 Collection 是否存在，不存在时按配置的向量维度自动创建。

        为什么要显式创建：Qdrant 不像 ChromaDB 可以隐式建库，
        必须预先声明向量维度和距离计算方式。
        使用余弦相似度（COSINE）与 bge 模型的训练目标一致。
        """
        existing = {c.name for c in self.client.get_collections().collections}
        if settings.QDRANT_COLLECTION not in existing:
            self.client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=settings.EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(
                f"已创建 Qdrant Collection '{settings.QDRANT_COLLECTION}' "
                f"(dim={settings.EMBEDDING_DIM}, distance=COSINE)"
            )

    def add_documents(self, documents: List[Document]) -> None:
        """将文档写入 Qdrant（向量化 + 持久化，Qdrant 服务负责存储）"""
        self.vectorstore.add_documents(documents)
        logger.info(f"已向 Qdrant 写入 {len(documents)} 个文本块")

    def get_retriever(self):
        """
        返回 MMR 检索器。

        MMR（Maximal Marginal Relevance）在保证相关性的同时
        降低结果冗余度，比纯相似度排序效果更好。
        """
        return self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": settings.RETRIEVAL_TOP_K},
        )
