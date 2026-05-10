# 工业售后 AI 智能客服

面向制造业售后场景的本地化 RAG 问答系统。支持 PDF 文档解析、网页抓取入库、混合检索、重排序、HyDE 检索增强、LangSmith 链路追踪。

## 技术栈

| 类别 | 技术 |
|---|---|
| 后端 API | FastAPI + Uvicorn |
| 前端 UI | Streamlit |
| 大模型 | DeepSeek API / Ollama（本地） |
| 向量数据库 | Qdrant（Docker） |
| Embedding | BAAI/bge-small-zh-v1.5 |
| RAG 框架 | LangChain |
| 检索增强 | HyDE + BM25 + FlashrankRerank |
| 网页抓取 | trafilatura |
| 链路追踪 | LangSmith（可选） |

## 快速开始

### 前提条件

- Python 3.11+
- Docker Desktop（运行状态）
- DeepSeek API Key **或** 本地 Ollama

### 1. 安装依赖

```bash
pip install -r requirements.txt
pip install trafilatura langdetect
```

### 2. 配置环境变量

复制示例文件并填写：

```bash
cp .env.example .env
```

最少需要填写的字段（以 DeepSeek 为例）：

```env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-你的key
DEEPSEEK_MODEL=deepseek-chat
```

### 3. 启动 Qdrant

```bash
docker compose up qdrant -d
```

验证：访问 http://localhost:6333/dashboard

### 4. 启动服务

```bash
# 终端1：后端 API
uvicorn src.api:app --reload --port 8000

# 终端2：前端 UI（可选）
streamlit run app.py
```

### 5. 验证

```bash
curl http://localhost:8000/health
# {"status":"ok","llm_provider":"deepseek","retriever_ready":false}
```

## API 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查 |
| POST | `/upload` | 上传 PDF 入库 |
| POST | `/crawl` | 批量抓取网页入库 |
| GET | `/crawl/history` | 查看已抓取 URL |
| POST | `/ask` | 问答 |

所有写操作需要请求头：`x-api-key: <API_KEY>`

### 示例：上传 PDF

```bash
curl -X POST http://localhost:8000/upload \
  -H "x-api-key: your-secret-key-2026" \
  -F "file=@手册.pdf"
```

### 示例：抓取网页

```bash
curl -X POST http://localhost:8000/crawl \
  -H "x-api-key: your-secret-key-2026" \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://example.com/manual"]}'
```

### 示例：提问

```bash
curl -X POST http://localhost:8000/ask \
  -H "x-api-key: your-secret-key-2026" \
  -H "Content-Type: application/json" \
  -d '{"question": "设备保修期是多久？"}'
```

## 系统架构

```
用户提问
  │
  ▼
HyDE 变换（问题 → 假设文档）
  │
  ├─ 向量检索（Qdrant）──┐
  │                      ├─ EnsembleRetriever → FlashrankRerank → LLM 生成答案
  └─ BM25 检索 ──────────┘
```

## 可选功能

### 开启 LangSmith 追踪

在 `.env` 中填写：

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__你的key
LANGCHAIN_PROJECT=industrial-rag
```

重启服务后，所有 LLM 调用、检索过程的延迟和 Token 消耗会上报到 [smith.langchain.com](https://smith.langchain.com)。

### 开启网页翻译

抓取到英文/日文等非中文网页时自动翻译：

```env
TRANSLATION_ENABLED=true
```

翻译使用 `LLM_PROVIDER` 指定的模型，每次抓取会增加额外 LLM 调用。

### 关闭 HyDE

```env
HYDE_ENABLED=false
```

关闭后问答速度加快（减少一次 LLM 调用），但检索召回率可能下降。

## 运行测试

```bash
pytest tests/ -v
```

## 评估

```bash
python evaluate.py --key your-secret-key-2026
```

结果保存至 `test/evaluation_report.json`，可在 Streamlit 评估看板中查看。
