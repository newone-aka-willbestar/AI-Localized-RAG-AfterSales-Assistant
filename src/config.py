from dotenv import load_dotenv
import os
load_dotenv()

class Config:
    LLM_MODEL = "qwen2:4b"          # 改成 Ollama
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    TOP_K = 4
    USE_HYDE = True
    VECTORSTORE = "chroma"
    PERSIST_DIR = "./chroma_db"