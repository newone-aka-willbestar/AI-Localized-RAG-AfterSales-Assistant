import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    """
    统一配置管理中心。
    所有参数优先从 .env 文件读取，其次使用下方默认值。
    修改配置只需改 .env 文件，代码本身不需要动。
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore"
    )

    # ==========================================
    # LLM 提供商选择（核心开关）
    # ==========================================
    # 可选值: "ollama" | "deepseek"
    LLM_PROVIDER: Literal["ollama", "deepseek"] = "ollama"

    # ==========================================
    # Ollama 本地模型配置
    # ==========================================
    OLLAMA_MODEL: str = "qwen2:7b"
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # ==========================================
    # DeepSeek API 配置
    # ==========================================
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_MODEL: str = "deepseek-chat"
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"

    # ==========================================
    # 通用 LLM 参数
    # ==========================================
    TEMPERATURE: float = 0.0

    # ==========================================
    # Embedding 模型（向量化，始终本地运行）
    # ==========================================
    EMBEDDING_MODEL_PATH: str = "./models/bge-small-zh-v1.5"

    # ==========================================
    # RAG 检索参数
    # ==========================================
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
    RETRIEVAL_TOP_K: int = 10
    RERANK_TOP_K: int = 4
    RERANK_MODEL_NAME: str = "ms-marco-MiniLM-L-12-v2"

    # ==========================================
    # 存储路径
    # ==========================================
    VECTORSTORE_PATH: str = "./chroma_db"

    # ==========================================
    # 翻译配置（第二阶段使用）
    # ==========================================
    TRANSLATION_PROVIDER: Literal["ollama", "deepseek", "none"] = "ollama"
    TRANSLATION_TARGET_LANG: str = "zh"

    # ==========================================
    # 安全与系统配置
    # ==========================================
    API_KEY: str = "your-secret-key-2026"
    LOG_LEVEL: str = "INFO"
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB


settings = Settings()
