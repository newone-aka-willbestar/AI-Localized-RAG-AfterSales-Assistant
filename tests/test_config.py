"""
测试配置模块 (src/config.py)

单元测试核心思路：
- 不依赖外部服务（不需要 Ollama 运行，不需要网络）
- 每个测试只验证一件事
- 测试名字说清楚"测什么、期望什么结果"
"""
import pytest
from pydantic import ValidationError


class TestSettingsDefaults:
    """测试配置的默认值是否正确"""

    def test_default_provider_is_ollama(self):
        """默认 LLM 提供商应该是 ollama"""
        from src.config import Settings
        s = Settings()
        assert s.LLM_PROVIDER == "ollama"

    def test_default_temperature_is_zero(self):
        """售后场景 temperature 默认应为 0，保证回答稳定"""
        from src.config import Settings
        s = Settings()
        assert s.TEMPERATURE == 0.0

    def test_default_chunk_size(self):
        """默认切片大小应为 800"""
        from src.config import Settings
        s = Settings()
        assert s.CHUNK_SIZE == 800

    def test_default_retrieval_top_k(self):
        """默认双路召回数量应为 10"""
        from src.config import Settings
        s = Settings()
        assert s.RETRIEVAL_TOP_K == 10

    def test_rerank_top_k_less_than_retrieval(self):
        """精排保留数量必须 <= 召回数量，否则精排没有意义"""
        from src.config import Settings
        s = Settings()
        assert s.RERANK_TOP_K <= s.RETRIEVAL_TOP_K


class TestSettingsFromEnv:
    """测试从环境变量覆盖配置"""

    def test_can_switch_to_deepseek_via_env(self, monkeypatch):
        """通过环境变量把 LLM_PROVIDER 切换为 deepseek"""
        monkeypatch.setenv("LLM_PROVIDER", "deepseek")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-123")
        from src.config import Settings
        s = Settings()
        assert s.LLM_PROVIDER == "deepseek"
        assert s.DEEPSEEK_API_KEY == "test-key-123"

    def test_invalid_provider_raises_error(self, monkeypatch):
        """填了不支持的 LLM_PROVIDER 值，应该在启动时就报错"""
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        from src.config import Settings
        with pytest.raises(ValidationError):
            Settings()

    def test_custom_chunk_size_from_env(self, monkeypatch):
        """环境变量可以覆盖 CHUNK_SIZE"""
        monkeypatch.setenv("CHUNK_SIZE", "500")
        from src.config import Settings
        s = Settings()
        assert s.CHUNK_SIZE == 500


class TestSettingsValidation:
    """测试配置的业务逻辑校验"""

    def test_translation_provider_values(self):
        """翻译提供商只允许特定值"""
        from src.config import Settings
        s = Settings()
        assert s.TRANSLATION_PROVIDER in ("ollama", "deepseek", "none")

    def test_max_upload_size_is_reasonable(self):
        """上传文件大小限制应该在合理范围内（1MB ~ 100MB）"""
        from src.config import Settings
        s = Settings()
        assert 1 * 1024 * 1024 <= s.MAX_UPLOAD_SIZE <= 100 * 1024 * 1024
