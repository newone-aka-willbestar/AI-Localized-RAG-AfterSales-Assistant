import streamlit as st
import requests
import json
import os
import uuid

st.set_page_config(page_title="AI 论文助理", page_icon="📚", layout="wide")

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
DEFAULT_API_KEY = os.environ.get("API_KEY", "")
REPORT_PATH = "test/evaluation_report.json"

REPORT_TYPE_OPTIONS = {
    "📄 文献摘要":  "summary",
    "🔑 要点提炼":  "keypoints",
    "📚 文献综述":  "review",
    "⚖️ 对比分析":  "comparison",
    "✏️ 自定义":   "custom",
}

# ==========================================
# 侧边栏
# ==========================================
with st.sidebar:
    st.title("📚 AI 论文助理")
    menu = st.radio(
        "选择功能",
        ["💬 智能对话", "📊 报表生成", "🔬 系统评估"],
    )
    st.divider()

    st.subheader("系统设置")
    api_key = st.text_input("API Key", value=DEFAULT_API_KEY, type="password")

    # --- PDF 上传 ---
    st.subheader("📄 上传 PDF 文献")
    uploaded_file = st.file_uploader("选择 PDF 文件", type=["pdf"])
    if uploaded_file and st.button("解析并入库", type="primary"):
        with st.spinner("正在解析文档并构建索引..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            headers = {"x-api-key": api_key}
            try:
                resp = requests.post(f"{API_BASE_URL}/upload", files=files, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"✅ 成功入库 {data.get('total_chunks', '?')} 个文本块")
                else:
                    st.error(f"处理失败: {resp.json().get('detail', resp.text)}")
            except Exception as e:
                st.error(f"连接 API 失败: {e}")

    st.divider()

    # --- 网页抓取 ---
    st.subheader("🌐 抓取网页文献")
    url_input = st.text_area(
        "输入网页 URL（每行一个）",
        placeholder="https://arxiv.org/abs/xxxx\nhttps://example.com/paper",
        height=100,
    )
    force_recrawl = st.checkbox("强制重新抓取", value=False)
    if st.button("抓取并入库"):
        urls = [u.strip() for u in url_input.strip().splitlines() if u.strip()]
        if not urls:
            st.warning("请先输入至少一个 URL")
        else:
            with st.spinner(f"正在抓取 {len(urls)} 个网页..."):
                headers = {"x-api-key": api_key, "Content-Type": "application/json"}
                try:
                    resp = requests.post(
                        f"{API_BASE_URL}/crawl",
                        json={"urls": urls, "force": force_recrawl, "translate": False},
                        headers=headers,
                        timeout=120,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        succeeded = data.get("succeeded", [])
                        skipped = data.get("skipped", [])
                        failed = data.get("failed", [])
                        if succeeded:
                            st.success(f"✅ 成功入库 {len(succeeded)} 个网页")
                            for item in succeeded:
                                st.caption(f"  · {item['url']} → {item['chunks']} 块")
                        if skipped:
                            st.info(f"⏭️ {len(skipped)} 个 URL 已抓取过，跳过")
                        if failed:
                            st.error(f"❌ {len(failed)} 个抓取失败")
                            for item in failed:
                                st.caption(f"  · {item['url']}: {item['error']}")
                    else:
                        st.error(f"请求失败: {resp.json().get('detail', resp.text)}")
                except Exception as e:
                    st.error(f"连接 API 失败: {e}")


# ==========================================
# 页面 A：智能对话
# ==========================================
if menu == "💬 智能对话":
    st.title("💬 智能对话")
    st.caption("基于知识库的多轮问答 · 支持上下文追问")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("🔍 查看参考原文"):
                    for src in msg["sources"]:
                        st.info(
                            f"📄 来源: {src.get('source', '未知')}\n\n"
                            f"{src.get('content_excerpt', '...')}"
                        )

    col_input, col_clear = st.columns([5, 1])
    with col_clear:
        if st.button("🗑️ 清空对话", use_container_width=True):
            try:
                requests.delete(
                    f"{API_BASE_URL}/session/{st.session_state.session_id}",
                    headers={"x-api-key": api_key},
                    timeout=5,
                )
            except Exception:
                pass
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()

    if prompt := st.chat_input("输入问题，支持多轮追问..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("检索知识库中..."):
                try:
                    headers = {"x-api-key": api_key}
                    resp = requests.post(
                        f"{API_BASE_URL}/ask",
                        json={"question": prompt, "session_id": st.session_state.session_id},
                        headers=headers,
                        timeout=180,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        answer = data.get("answer", "未能生成答案")
                        sources = data.get("sources", [])
                        intent = data.get("intent", "")
                        st.markdown(answer)
                        if intent:
                            st.caption(f"意图：{intent}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                            "intent": intent,
                            "question": prompt,
                        })
                    else:
                        st.error(f"API 响应异常: {resp.status_code}")
                except Exception as e:
                    st.error(f"发生错误: {e}")


# ==========================================
# 页面 B：报表生成
# ==========================================
elif menu == "📊 报表生成":
    st.title("📊 报表生成")
    st.caption("从已上传的 PDF / 网页中，自动生成结构化学术报告 · 支持直接下载")

    # --- 表单区 ---
    with st.form("report_form"):
        topic = st.text_input(
            "报告主题",
            placeholder="例：大语言模型在医学影像诊断中的应用",
        )

        col1, col2 = st.columns([2, 1])
        with col1:
            report_type_label = st.selectbox(
                "报告类型",
                list(REPORT_TYPE_OPTIONS.keys()),
            )
        with col2:
            max_sources = st.slider("最多引用文献片段数", 2, 20, 8)

        report_type = REPORT_TYPE_OPTIONS[report_type_label]

        custom_instruction = ""
        if report_type == "custom":
            custom_instruction = st.text_area(
                "自定义指令",
                placeholder="例：请重点分析该方法的局限性，并与传统方法做对比，用中文输出",
                height=100,
            )

        submitted = st.form_submit_button("🚀 生成报告", type="primary", use_container_width=True)

    # --- 生成 ---
    if submitted:
        if not topic.strip():
            st.warning("请输入报告主题")
        else:
            with st.spinner(f"正在检索文献并生成{report_type_label}报告，可能需要 30-60 秒..."):
                try:
                    headers = {"x-api-key": api_key}
                    resp = requests.post(
                        f"{API_BASE_URL}/report",
                        json={
                            "topic": topic,
                            "report_type": report_type,
                            "custom_instruction": custom_instruction,
                            "max_sources": max_sources,
                        },
                        headers=headers,
                        timeout=300,
                    )

                    if resp.status_code == 200:
                        data = resp.json()
                        report_text = data.get("report", "")
                        sources = data.get("sources", [])

                        # 展示报告
                        st.divider()
                        st.subheader(f"{report_type_label}：{topic}")
                        st.markdown(report_text)

                        # 下载按钮
                        st.divider()
                        col_dl1, col_dl2 = st.columns(2)
                        with col_dl1:
                            st.download_button(
                                label="⬇️ 下载 Markdown (.md)",
                                data=report_text.encode("utf-8"),
                                file_name=f"report_{topic[:20]}.md",
                                mime="text/markdown",
                                use_container_width=True,
                            )
                        with col_dl2:
                            st.download_button(
                                label="⬇️ 下载纯文本 (.txt)",
                                data=report_text.encode("utf-8"),
                                file_name=f"report_{topic[:20]}.txt",
                                mime="text/plain",
                                use_container_width=True,
                            )

                        # 参考来源
                        if sources:
                            with st.expander(f"📚 参考来源（共 {len(sources)} 条）"):
                                for i, src in enumerate(sources, 1):
                                    st.markdown(
                                        f"**{i}.** 📄 `{src.get('source', '未知')}`\n\n"
                                        f"> {src.get('content_excerpt', '...')}"
                                    )
                    else:
                        st.error(f"生成失败: {resp.json().get('detail', resp.text)}")

                except Exception as e:
                    st.error(f"连接 API 失败: {e}")

    # 使用说明
    else:
        st.info(
            "**使用流程：**\n"
            "1. 在左侧侧边栏上传 PDF 文献或抓取相关网页\n"
            "2. 在上方填写报告主题和类型\n"
            "3. 点击「生成报告」，等待 AI 检索并撰写\n"
            "4. 报告以 Markdown 格式展示，可直接下载"
        )
        st.markdown("""
| 报告类型 | 适合场景 |
|----------|----------|
| 📄 文献摘要 | 快速了解一篇/多篇论文的核心内容 |
| 🔑 要点提炼 | 整理论文中的关键论点和数据 |
| 📚 文献综述 | 生成论文「相关工作」章节的草稿 |
| ⚖️ 对比分析 | 横向比较多篇文献的方法和结论 |
| ✏️ 自定义 | 按你的具体需求生成任意结构报告 |
""")


# ==========================================
# 页面 C：系统评估
# ==========================================
elif menu == "🔬 系统评估":
    st.title("🔬 系统性能评估")

    # 健康状态
    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=5).json()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("知识库文档块", health.get("doc_count", 0))
        c2.metric("活跃会话数", health.get("active_sessions", 0))
        c3.metric("LLM 提供商", health.get("llm_provider", "-"))
        c4.metric("检索引擎", "✅ 就绪" if health.get("retriever_ready") else "⚠️ 未就绪")
        st.divider()
    except Exception:
        st.warning("无法连接到 API 服务")

    if os.path.exists(REPORT_PATH):
        try:
            with open(REPORT_PATH, "r", encoding="utf-8") as f:
                report = json.load(f)

            summary = report.get("summary", {})
            c1, c2, c3 = st.columns(3)
            c1.metric("端到端准确率", summary.get("accuracy", "N/A"))
            c2.metric("平均响应耗时", f"{summary.get('avg_latency_sec', 0)}s")
            c3.metric("测试用例总数", summary.get("total_questions", 0))

            st.divider()
            st.subheader("详细评测流水")
            for item in report.get("details", []):
                status = "✅ 通过" if item["is_correct"] else "❌ 失败"
                with st.expander(f"{status} | {item['question']}"):
                    st.write(f"**AI 回答:** {item['answer']}")
                    st.write(f"**耗时:** {item['latency']}s")
                    if item.get("sources"):
                        st.json(item["sources"])
        except Exception as e:
            st.error(f"读取报告出错: {e}")
    else:
        st.warning("⚠️ 暂无评估报告")
        st.info("运行以下命令生成报告：")
        st.code("python evaluate.py --key your-secret-key-2026")
