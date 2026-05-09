"""
测试 VectorStore（Qdrant 版）。

所有测试都用 QdrantClient(":memory:") 内存模式，
不依赖 Docker，CI 也能运行。
Embedding 模型用 FakeEmbeddings 代替，不加载真实权重。
"""
import pytest
from unittest.mock import patch
from typing import List


# ==========================================
# 伪 Embedding（必须继承 langchain Embeddings 基类）
# ==========================================

class FakeEmbeddings:
    """
    符合 langchain Embeddings 接口的轻量假实现。

    langchain_qdrant 1.1.0 在初始化时会 isinstance 检查，
    纯 MagicMock 无法通过，所以用真实子类代替。
    返回固定 512 维向量，每个文档用 index 偏移避免全零重复。
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[float(idx + 1) * 0.01] * 512 for idx, _ in enumerate(texts)]

    def embed_query(self, text: str) -> List[float]:
        return [0.05] * 512


def _make_langchain_embeddings():
    """
    动态让 FakeEmbeddings 通过 langchain_qdrant 的 isinstance 检查。

    langchain_qdrant 检查 isinstance(emb, langchain_core.embeddings.Embeddings)，
    为了不在模块顶层 import langchain_core（避免 pydantic_v1 问题），
    在这里动态添加基类。
    """
    from langchain_core.embeddings import Embeddings

    class _FakeEmbeddingsWithBase(FakeEmbeddings, Embeddings):
        pass

    return _FakeEmbeddingsWithBase()


# ==========================================
# Fixtures
# ==========================================

@pytest.fixture
def mem_client():
    """Qdrant 内存客户端——不需要 Docker，测试结束自动清理"""
    qdrant_client = pytest.importorskip("qdrant_client", reason="需要 qdrant-client")
    return qdrant_client.QdrantClient(":memory:")


@pytest.fixture
def fake_embeddings():
    """符合 langchain Embeddings 接口的 512 维伪模型"""
    pytest.importorskip("langchain_core", reason="需要 langchain_core")
    return _make_langchain_embeddings()


@pytest.fixture
def store(mem_client, fake_embeddings):
    """
    注入内存 Qdrant + 伪 Embedding 的 VectorStore 实例。

    patch _build_embeddings 让 VectorStore.__init__ 拿到 fake_embeddings，
    再传入 mem_client 跳过真实 Qdrant 连接。

    注意：patch 要求目标模块已经被 import，所以先 import 再用 patch.object。
    """
    pytest.importorskip("langchain_qdrant", reason="需要 langchain-qdrant")
    import src.vector_store as vs_module  # 确保模块加载，patch 才能找到目标
    with patch.object(vs_module, "_build_embeddings", return_value=fake_embeddings):
        return vs_module.VectorStore(client=mem_client)


# ==========================================
# 测试：Collection 自动创建
# ==========================================

class TestCollectionSetup:
    def test_collection_created_on_init(self, store):
        """VectorStore 初始化后，Qdrant 里应该存在对应的 Collection"""
        from src.config import settings
        collections = {c.name for c in store.client.get_collections().collections}
        assert settings.QDRANT_COLLECTION in collections, (
            f"Collection '{settings.QDRANT_COLLECTION}' 未自动创建"
        )

    def test_collection_not_duplicated(self, mem_client, fake_embeddings):
        """多次初始化 VectorStore 不应报错（Collection 已存在时应幂等）"""
        pytest.importorskip("langchain_qdrant", reason="需要 langchain-qdrant")
        import src.vector_store as vs_module
        with patch.object(vs_module, "_build_embeddings", return_value=fake_embeddings):
            vs_module.VectorStore(client=mem_client)  # 第一次
            vs_module.VectorStore(client=mem_client)  # 第二次，不应抛异常


# ==========================================
# 测试：文档写入与检索
# ==========================================

class TestAddAndRetrieve:
    def test_add_documents_increases_count(self, store):
        """add_documents 后，Collection 里的向量数量应该增加"""
        from langchain_core.documents import Document
        docs = [
            Document(page_content="设备保修期为一年"),
            Document(page_content="错误代码 E05 表示传感器故障"),
        ]
        before = store.client.count(store.client.get_collections().collections[0].name).count
        store.add_documents(docs)
        after = store.client.count(store.client.get_collections().collections[0].name).count
        assert after == before + 2, "写入2个文档后向量数量应增加2"

    def test_get_retriever_returns_non_none(self, store):
        """get_retriever 应该返回一个可用的检索器对象"""
        from langchain_core.documents import Document
        store.add_documents([Document(page_content="测试文档")])
        retriever = store.get_retriever()
        assert retriever is not None

    def test_add_empty_list_does_not_crash(self, store):
        """传入空列表不应崩溃"""
        store.add_documents([])  # 不应抛异常


# ==========================================
# 测试：接口契约不变（rag.py 依赖的方法签名）
# ==========================================

class TestInterfaceContract:
    def test_has_add_documents_method(self, store):
        """VectorStore 必须有 add_documents 方法（rag.py 依赖）"""
        assert callable(getattr(store, "add_documents", None))

    def test_has_get_retriever_method(self, store):
        """VectorStore 必须有 get_retriever 方法（rag.py 依赖）"""
        assert callable(getattr(store, "get_retriever", None))

    def test_retriever_supports_k_parameter(self, store):
        """get_retriever 返回的对象应接受 search_kwargs 配置"""
        from langchain_core.documents import Document
        store.add_documents([Document(page_content="测试")])
        retriever = store.get_retriever()
        # 检索器有 search_kwargs 属性，说明 k 参数被正确传递
        assert hasattr(retriever, "search_kwargs"), \
            "检索器应有 search_kwargs 属性"
