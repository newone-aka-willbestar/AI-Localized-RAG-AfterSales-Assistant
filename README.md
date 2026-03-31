# AI 智能售后客服系统（本地化 RAG）

**全本地部署 · 数据不出域 · 专为制造业设计**  
支持上传 PDF 产品手册 → 自动分块向量化 → 实时智能问答，完美解决售后工程师“翻手册找答案”的痛点。

## ✨ 核心特性
- **100% 本地化部署**：Ollama + Qwen2-7B + ChromaDB，无需联网，无数据泄露风险
- **中文优化 RAG 流程**：使用 BAAI/bge-small-zh-v1.5 嵌入模型，解决大陆网络下载问题
- **前后端分离架构**：FastAPI 后端（API 服务） + Streamlit 前端（交互界面）
- **企业级 Prompt 工程**：针对制造业售后场景专门设计的提示词，回答专业且准确
- **Docker 一键部署**：支持 docker-compose 快速启动

## 🛠 技术栈
- **Backend**：FastAPI + Uvicorn
- **Frontend**：Streamlit
- **LLM**：Ollama + Qwen2-7B（本地）
- **向量数据库**：ChromaDB
- **Embedding**：BAAI/bge-small-zh-v1.5（已预下载）
- **RAG 框架**：LangChain
- **文档处理**：PyPDFLoader + RecursiveCharacterTextSplitter

## 🚀 快速启动（3 步）
```bash
# 1. 启动 Ollama
ollama serve

# 2. 启动后端
uvicorn src.api:app --reload

# 3. 启动前端
streamlit run app.py
Docker 部署（推荐）：
Bashdocker-compose up -d
## 📊 系统架构（Mermaid 图）

## 🎯 面试亮点（你可以直接说）

解决了中国网络环境下 Hugging Face 下载慢的实际痛点（预下载模型 + 本地 embedding）
完整实现了 RAG 全流程（加载 → 分割 → 嵌入 → 检索 → 生成）
强调数据安全与合规，适合制造业/金融等对隐私要求高的企业
前后端分离 + Docker 部署，体现生产级思维

## 📸 项目截图（建议你现在补）

## 后续计划
集成 LangGraph 实现多 Agent（文档智能体 + 问答智能体 + 反馈智能体）
添加 Redis 缓存 + 流式输出（Streaming）
支持多租户与用户认证

Star 一下支持我继续迭代吧！ ⭐