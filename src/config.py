import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    """
    企业级配置管理：支持从 .env 文件自动读取，具备类型检查与默认值。
    面试点：使用 Pydantic 管理配置可以确保系统启动时即完成参数校验，避免运行中出错。
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore"
    )

    # --- 大模型 (LLM) 配置 ---
    # 默认使用通义千问 Qwen2，也可方便切换为 DeepSeek
    OLLAMA_MODEL: str = "qwen2:7b"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    TEMPERATURE: float = 0.0  # 售后场景建议设为 0，保证回答稳定性

    # --- 向量模型 (Embedding) 配置 ---
    # 路径建议使用环境变量，方便在不同机器部署
    EMBEDDING_MODEL_PATH: str = "./models/bge-small-zh-v1.5"
    
    # --- RAG 流程参数 ---
    # 文本切片大小：500-1000 是工业界处理中文手册的常用范围
    CHUNK_SIZE: int = 800 
    CHUNK_OVERLAP: int = 150
    # 初始召回数量 (Vector + BM25)
    RETRIEVAL_TOP_K: int = 10
    # 最终精排后保留给 LLM 的数量
    RERANK_TOP_K: int = 4
    # 精排模型名称 (Flashrank)
    RERANK_MODEL_NAME: str = "ms-marco-MiniLM-L-12-v2"

    # --- 存储配置 ---
    VECTORSTORE_PATH: str = "./chroma_db"
    
    # --- 安全与系统配置 (面试加分点) ---
    # 模拟企业级 API 访问令牌
    API_KEY: str = "your-secret-key-2026"
    LOG_LEVEL: str = "INFO"
    # 限制上传文件大小 (10MB)，防止服务器崩溃
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024 

# 实例化配置对象
settings = Settings()