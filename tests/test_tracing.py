"""
测试 LangSmith 追踪模块。

测试策略：
- 不真实连接 LangSmith（无需网络，无需 API Key）
- 验证环境变量被正确设置/不被设置
- 验证配置缺失时的警告行为
- 验证 get_run_config 生成的结构正确
"""
import os
import pytest
from unittest.mock import patch, MagicMock


class TestSetupTracing:
    def test_disabled_when_flag_false(self):
        """LANGCHAIN_TRACING_V2=false 时不设置环境变量，返回 False"""
        from src.tracing import setup_tracing

        with patch("src.tracing.settings") as mock_settings:
            mock_settings.LANGCHAIN_TRACING_V2 = False

            # 清除可能存在的环境变量，避免测试污染
            env_backup = os.environ.pop("LANGCHAIN_TRACING_V2", None)
            try:
                result = setup_tracing()
            finally:
                if env_backup is not None:
                    os.environ["LANGCHAIN_TRACING_V2"] = env_backup

        assert result is False
        # 确认环境变量没被设置
        assert os.environ.get("LANGCHAIN_TRACING_V2") != "true" or env_backup == "true"

    def test_disabled_when_api_key_empty(self):
        """LANGCHAIN_TRACING_V2=true 但 API Key 为空 → 返回 False，打 warning"""
        from src.tracing import setup_tracing

        with patch("src.tracing.settings") as mock_settings:
            mock_settings.LANGCHAIN_TRACING_V2 = True
            mock_settings.LANGCHAIN_API_KEY = ""

            result = setup_tracing()

        assert result is False

    def test_enabled_sets_env_vars(self):
        """配置完整时，环境变量被正确设置，返回 True"""
        from src.tracing import setup_tracing

        # 保存并清除现有环境变量，测试后还原
        keys = ["LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY",
                "LANGCHAIN_PROJECT", "LANGCHAIN_ENDPOINT"]
        backup = {k: os.environ.pop(k, None) for k in keys}

        try:
            with patch("src.tracing.settings") as mock_settings:
                mock_settings.LANGCHAIN_TRACING_V2 = True
                mock_settings.LANGCHAIN_API_KEY = "ls__test_key_abc123"
                mock_settings.LANGCHAIN_PROJECT = "test-project"
                mock_settings.LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"

                result = setup_tracing()

            assert result is True
            assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
            assert os.environ["LANGCHAIN_API_KEY"] == "ls__test_key_abc123"
            assert os.environ["LANGCHAIN_PROJECT"] == "test-project"
            assert os.environ["LANGCHAIN_ENDPOINT"] == "https://api.smith.langchain.com"

        finally:
            # 还原环境变量，不污染其他测试
            for k, v in backup.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_idempotent_safe_to_call_twice(self):
        """多次调用 setup_tracing 不崩溃（FastAPI 热重载场景）"""
        from src.tracing import setup_tracing

        with patch("src.tracing.settings") as mock_settings:
            mock_settings.LANGCHAIN_TRACING_V2 = False

            setup_tracing()
            setup_tracing()  # 第二次不应报错


class TestGetRunConfig:
    def test_run_name_is_set(self):
        """run_name 正确写入 config"""
        from src.tracing import get_run_config

        with patch("src.tracing.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "ollama"
            mock_settings.HYDE_ENABLED = True

            config = get_run_config("RAG.ask")

        assert config["run_name"] == "RAG.ask"

    def test_default_tags_include_provider(self):
        """默认 tags 包含 provider 信息"""
        from src.tracing import get_run_config

        with patch("src.tracing.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "deepseek"
            mock_settings.HYDE_ENABLED = False

            config = get_run_config("test")

        assert "provider:deepseek" in config["tags"]

    def test_hyde_tag_reflects_setting(self):
        """hyde:on / hyde:off 标签随 HYDE_ENABLED 变化"""
        from src.tracing import get_run_config

        with patch("src.tracing.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "ollama"
            mock_settings.HYDE_ENABLED = True
            config_on = get_run_config("test")

            mock_settings.HYDE_ENABLED = False
            config_off = get_run_config("test")

        assert "hyde:on" in config_on["tags"]
        assert "hyde:off" in config_off["tags"]

    def test_extra_tags_merged(self):
        """额外传入的 tags 与默认 tags 合并"""
        from src.tracing import get_run_config

        with patch("src.tracing.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "ollama"
            mock_settings.HYDE_ENABLED = True

            config = get_run_config("test", tags=["prod", "v2"])

        assert "prod" in config["tags"]
        assert "v2" in config["tags"]
        assert "provider:ollama" in config["tags"]

    def test_metadata_included(self):
        """传入 metadata 时被原样写入 config"""
        from src.tracing import get_run_config

        with patch("src.tracing.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "ollama"
            mock_settings.HYDE_ENABLED = True

            config = get_run_config(
                "RAG.ask",
                metadata={"question": "设备保修多久？", "doc_count": 4}
            )

        assert config["metadata"]["question"] == "设备保修多久？"
        assert config["metadata"]["doc_count"] == 4

    def test_no_metadata_key_when_not_provided(self):
        """不传 metadata 时 config 中不应有 metadata 键（避免空值干扰）"""
        from src.tracing import get_run_config

        with patch("src.tracing.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "ollama"
            mock_settings.HYDE_ENABLED = True

            config = get_run_config("test")

        assert "metadata" not in config
