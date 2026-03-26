# 华科制造智能售后客服系统 | RAG 本地智能客服

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38-FF4B4B)](https://streamlit.io/)

一个**完全本地化**的制造业智能售后客服系统，支持上传产品手册 PDF 并实时问答。


## ✨ 项目亮点

- 完全本地部署（数据不出域，适合制造业企业）
- 支持 PDF 文档上传 → 自动分块 → 向量化 → 实时检索
- 使用中文优化 Embedding 模型（BAAI/bge-small-zh-v1.5）
- 解决大陆网络环境下 huggingface 下载难题
- 企业级售后 Prompt 工程设计
- 前后端分离（FastAPI + Streamlit）

## 🛠️ 技术栈

- **后端**：FastAPI + Uvicorn
- **前端**：Streamlit
- **LLM**：Ollama + Qwen2-7B（本地）
- **向量数据库**：ChromaDB
- **Embedding**：BAAI/bge-small-zh-v1.5
- **RAG 框架**：LangChain（langchain-classic 兼容）
- **文档处理**：PyPDFLoader + RecursiveCharacterTextSplitter

#快速启动方案

```bash
# 1. 启动 Ollama
ollama serve

# 2. 启动后端
uvicorn src.api:app --reload

# 3. 启动前端
streamlit run app.py

# 核心解决的问题

大陆网络下 embedding 模型下载慢 → 使用 hf-mirror + 手动预下载
Ollama 被 VPN 代理导致 502 → 使用 httpx.Client(proxies=None, trust_env=False)
不同 embedding 维度冲突 → 删除 chroma_db 后重新向量化
LangChain 版本兼容问题 → 使用 langchain-classic
Rag的热重载问题与模型本地化

#未来改造计划
向量库管理后台（CRUD + 版本控制）
Redis 缓存高频答案
流式输出（StreamingResponse）
LangSmith / LangFuse 全链路追踪
JWT 权限 + 多租户支持