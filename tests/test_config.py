"""
配置模块测试。

策略：
- 验证默认值是否符合预期
- 验证关键字段类型和范围
- 不做任何真实网络/文件操作
"""
import pytest
from src.config import Settings


# ==========================================
# 默认值验证
# ==========================================

class TestDefaults:

    def setup_method(self):
        # 用空环境创建 Settings，只用代码内的默认值
        self.s = Settings(
            _env_file=None,
            DEEPSEEK_API_KEY="",
            LANGCHAIN_API_KEY="",
            API_KEY="test-key",
        )

    def test_default_llm_provider(self):
        assert self.s.LLM_PROVIDER == "ollama"

    def test_default_ollama_model(self):
        assert self.s.OLLAMA_MODEL == "qwen2:7b"

    def test_default_qdrant_port(self):
        assert self.s.QDRANT_PORT == 6333

    def test_default_embedding_dim(self):
        assert self.s.EMBEDDING_DIM == 512

    def test_default_chunk_size_positive(self):
        assert self.s.CHUNK_SIZE > 0

    def test_default_retrieval_top_k_positive(self):
        assert self.s.RETRIEVAL_TOP_K > 0

    def test_default_rerank_top_k_less_than_retrieval(self):
        assert self.s.RERANK_TOP_K <= self.s.RETRIEVAL_TOP_K

    def test_default_temperature_in_range(self):
        assert 0.0 <= self.s.TEMPERATURE <= 2.0

    def test_default_hyde_enabled(self):
        assert isinstance(self.s.HYDE_ENABLED, bool)

    def test_default_langsmith_tracing_disabled(self):
        assert self.s.LANGCHAIN_TRACING_V2 is False

    def test_default_translation_disabled(self):
        assert self.s.TRANSLATION_ENABLED is False

    def test_max_upload_size_is_reasonable(self):
        # 应在 1MB ~ 100MB 之间
        assert 1 * 1024 * 1024 <= self.s.MAX_UPLOAD_SIZE <= 100 * 1024 * 1024


# ==========================================
# 字段类型验证
# ==========================================

class TestFieldTypes:

    def setup_method(self):
        self.s = Settings(_env_file=None, API_KEY="test-key")

    def test_qdrant_port_is_int(self):
        assert isinstance(self.s.QDRANT_PORT, int)

    def test_chunk_size_is_int(self):
        assert isinstance(self.s.CHUNK_SIZE, int)

    def test_temperature_is_float(self):
        assert isinstance(self.s.TEMPERATURE, float)

    def test_embedding_model_path_is_str(self):
        assert isinstance(self.s.EMBEDDING_MODEL_PATH, str)

    def test_llm_provider_is_valid_choice(self):
        assert self.s.LLM_PROVIDER in ("ollama", "deepseek")
