from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore"
    )

    # LLM 配置
    OLLAMA_MODEL: str = "qwen2:7b"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    TEMPERATURE: float = 0.3

    # Embedding（关键！改成这个中文模型）
    EMBEDDING_MODEL: str = "BAAI/bge-small-zh-v1.5"

    # RAG 参数
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 4
    USE_HYDE: bool = True

    # 向量库
    VECTORSTORE_PATH: str = "./chroma_db"

    # 安全
    API_KEY: Optional[str] = None

    # 日志
    LOG_LEVEL: str = "INFO"

settings = Settings()