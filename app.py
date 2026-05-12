import streamlit as st
import requests
import json
import os
import uuid

try:
    from src.docx_exporter import markdown_to_docx_bytes
    _DOCX_AVAILABLE = True
except Exception:
    _DOCX_AVAILABLE = False

st.set_page_config(
    page_title="AI 数字员工",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# 全局样式
# ==========================================
st.markdown("""
<style>
/* 隐藏默认 header / footer */
#MainMenu, footer, header {visibility: hidden;}

/* 侧边栏背景 */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
[data-testid="stSidebar"] * {color: #e2e8f0 !important;}
[data-testid="stSidebar"] .stButton > button {
    background: #3b82f6;
    color: white !important;
    border: none;
    border-radius: 8px;
    width: 100%;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #2563eb;
}
[data-testid="stSidebar"] hr {border-color: #334155 !important;}

/* radio 菜单 */
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 8px 14px;
    margin: 3px 0;
    cursor: pointer;
    transition: all 0.2s;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #334155;
    border-color: #3b82f6;
}

/* 主区域卡片 */
.card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

/* 页面大标题 */
.page-header {
    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #06b6d4 100%);
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 24px;
    color: white !important;
}
.page-header h1 {color: white !important; margin: 0; font-size: 1.8rem;}
.page-header p  {color: #bfdbfe !important; margin: 4px 0 0; font-size: 0.95rem;}

/* metric 卡片优化 */
[data-testid="metric-container"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 12px !important;
}

/* 聊天气泡 */
[data-testid="stChatMessage"] {border-radius: 12px; margin-bottom: 8px;}

/* 主按钮 */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #3b82f6, #06b6d4);
    border: none;
    border-radius: 8px;
    color: white;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
    transition: opacity 0.2s;
}
.stButton > button[kind="primary"]:hover {opacity: 0.9;}

/* expander 美化 */
[data-testid="stExpander"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    margin-bottom: 8px;
}

/* 状态徽章 */
.badge-ok  {background:#dcfce7;color:#166534;border-radius:20px;padding:2px 10px;font-size:0.8rem;font-weight:600;}
.badge-err {background:#fee2e2;color:#991b1b;border-radius:20px;padding:2px 10px;font-size:0.8rem;font-weight:600;}
.badge-na  {background:#f1f5f9;color:#475569;border-radius:20px;padding:2px 10px;font-size:0.8rem;font-weight:600;}
</style>
""", unsafe_allow_html=True)

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
    st.markdown("""
    <div style='text-align:center;padding:16px 0 8px'>
        <div style='font-size:2.5rem'>🤖</div>
        <div style='font-size:1.2rem;font-weight:700;color:#f1f5f9'>AI 数字员工</div>
        <div style='font-size:0.78rem;color:#94a3b8;margin-top:2px'>智能知识库助手</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    menu = st.radio(
        "功能导航",
        ["💬 智能对话", "📊 报表生成", "🔬 系统评估"],
        label_visibility="collapsed",
    )
    st.divider()

    api_key = st.text_input("🔑 API Key", value=DEFAULT_API_KEY, type="password")

    st.markdown("**📄 上传文献**")
    uploaded_file = st.file_uploader(
        "支持 PDF / Word", type=["pdf", "docx"], label_visibility="collapsed"
    )
    if uploaded_file and st.button("解析并入库", type="primary"):
        with st.spinner("解析中…"):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/octet-stream")}
            headers = {"x-api-key": api_key}
            try:
                resp = requests.post(f"{API_BASE_URL}/upload", files=files, headers=headers, timeout=120)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"✅ 入库 {data.get('total_chunks', '?')} 个文本块")
                else:
                    st.error(f"失败: {resp.json().get('detail', resp.text)[:80]}")
            except Exception as e:
                st.error(f"连接失败: {e}")

    st.divider()
    st.markdown("**🌐 抓取网页**")
    url_input = st.text_area(
        "每行一个 URL",
        placeholder="https://arxiv.org/abs/xxxx",
        height=80,
        label_visibility="collapsed",
    )
    col_crawl, col_force = st.columns([3, 2])
    with col_force:
        force_recrawl = st.checkbox("强制重抓", value=False)
    with col_crawl:
        do_crawl = st.button("开始抓取")
    if do_crawl:
        urls = [u.strip() for u in url_input.strip().splitlines() if u.strip()]
        if not urls:
            st.warning("请输入 URL")
        else:
            with st.spinner(f"抓取 {len(urls)} 个页面…"):
                headers = {"x-api-key": api_key, "Content-Type": "application/json"}
                try:
                    resp = requests.post(
                        f"{API_BASE_URL}/crawl",
                        json={"urls": urls, "force": force_recrawl, "translate": False},
                        headers=headers, timeout=120,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        ok = len(data.get("succeeded", []))
                        sk = len(data.get("skipped", []))
                        fl = len(data.get("failed", []))
                        if ok: st.success(f"✅ {ok} 个成功")
                        if sk: st.info(f"⏭️ {sk} 个已跳过")
                        if fl: st.error(f"❌ {fl} 个失败")
                    else:
                        st.error("请求失败")
                except Exception as e:
                    st.error(f"连接失败: {e}")

    # 底部系统状态
    st.divider()
    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=3).json()
        status_color = "#22c55e" if health.get("retriever_ready") else "#f59e0b"
        st.markdown(f"""
        <div style='font-size:0.8rem;color:#94a3b8'>
            <div>● <span style='color:{status_color}'>
                {'检索引擎就绪' if health.get('retriever_ready') else '等待文档上传'}
            </span></div>
            <div style='margin-top:4px'>📚 {health.get('doc_count',0)} 个文本块
            &nbsp;|&nbsp; 💬 {health.get('active_sessions',0)} 个会话</div>
            <div style='margin-top:2px'>🤖 {health.get('llm_provider','-').upper()}</div>
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        st.markdown("<div style='font-size:0.8rem;color:#ef4444'>⚠️ 服务未连接</div>", unsafe_allow_html=True)


# ==========================================
# 页面 A：智能对话
# ==========================================
if menu == "💬 智能对话":
    st.markdown("""
    <div class='page-header'>
        <h1>💬 智能对话</h1>
        <p>基于知识库的多轮问答 · 上传文献后直接提问</p>
    </div>
    """, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    # 工具栏
    col_info, col_clear = st.columns([5, 1])
    with col_info:
        st.caption(f"会话 ID：`{st.session_state.session_id[:8]}…`  |  共 {len(st.session_state.messages)} 条消息")
    with col_clear:
        if st.button("🗑️ 清空", use_container_width=True):
            try:
                requests.delete(
                    f"{API_BASE_URL}/session/{st.session_state.session_id}",
                    headers={"x-api-key": api_key}, timeout=5,
                )
            except Exception:
                pass
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()

    # 欢迎提示（无消息时）
    if not st.session_state.messages:
        st.markdown("""
        <div class='card' style='text-align:center;padding:40px;border-style:dashed;border-color:#cbd5e1'>
            <div style='font-size:2.5rem'>📚</div>
            <div style='font-size:1.1rem;font-weight:600;color:#334155;margin:8px 0'>知识库已就绪，开始提问吧</div>
            <div style='color:#94a3b8;font-size:0.9rem'>先在左侧上传 PDF 或抓取网页，再输入问题</div>
        </div>
        """, unsafe_allow_html=True)

    # 历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander(f"🔍 查看 {len(msg['sources'])} 条参考原文"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(
                            f"**{i}.** `{src.get('source', '未知')}`\n\n"
                            f"> {src.get('content_excerpt', '...')[:200]}"
                        )
            if msg["role"] == "assistant" and msg.get("intent"):
                st.caption(f"意图识别：{msg['intent']}")

    # 输入框
    if prompt := st.chat_input("输入问题，支持多轮追问…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("检索知识库中…"):
                try:
                    resp = requests.post(
                        f"{API_BASE_URL}/ask",
                        json={"question": prompt, "session_id": st.session_state.session_id},
                        headers={"x-api-key": api_key},
                        timeout=180,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        answer  = data.get("answer", "未能生成答案")
                        sources = data.get("sources", [])
                        intent  = data.get("intent", "")
                        st.markdown(answer)
                        if intent:
                            st.caption(f"意图识别：{intent}")
                        st.session_state.messages.append({
                            "role": "assistant", "content": answer,
                            "sources": sources, "intent": intent,
                        })
                    else:
                        err = resp.json().get("detail", resp.text) if resp.headers.get("content-type","").startswith("application/json") else resp.text
                        st.error(f"API 错误 {resp.status_code}：{err[:120]}")
                except Exception as e:
                    st.error(f"请求失败：{e}")


# ==========================================
# 页面 B：报表生成
# ==========================================
elif menu == "📊 报表生成":
    st.markdown("""
    <div class='page-header'>
        <h1>📊 报表生成</h1>
        <p>从已上传的文献中自动生成结构化学术报告，支持 Markdown / Word 下载</p>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        with st.form("report_form"):
            topic = st.text_input(
                "📌 报告主题",
                placeholder="例：深度学习在农作物病害识别中的应用进展",
            )
            col1, col2 = st.columns([3, 2])
            with col1:
                report_type_label = st.selectbox("📋 报告类型", list(REPORT_TYPE_OPTIONS.keys()))
            with col2:
                max_sources = st.slider("引用片段上限", 2, 20, 8)

            report_type = REPORT_TYPE_OPTIONS[report_type_label]
            custom_instruction = ""
            if report_type == "custom":
                custom_instruction = st.text_area(
                    "✏️ 自定义指令",
                    placeholder="例：重点分析该方法的局限性，并与传统方法对比，用中文输出",
                    height=90,
                )

            submitted = st.form_submit_button("🚀 生成报告", type="primary", use_container_width=True)

    if submitted:
        if not topic.strip():
            st.warning("请输入报告主题")
        else:
            with st.spinner(f"正在检索文献并生成报告，约需 30-60 秒…"):
                try:
                    resp = requests.post(
                        f"{API_BASE_URL}/report",
                        json={
                            "topic": topic, "report_type": report_type,
                            "custom_instruction": custom_instruction,
                            "max_sources": max_sources,
                        },
                        headers={"x-api-key": api_key},
                        timeout=300,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        report_text = data.get("report", "")
                        sources     = data.get("sources", [])

                        st.markdown(f"""
                        <div class='card'>
                            <div style='font-size:0.8rem;color:#64748b;margin-bottom:4px'>{report_type_label}</div>
                            <div style='font-size:1.3rem;font-weight:700;color:#1e293b'>{topic}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(report_text)

                        # 下载区
                        st.divider()
                        safe_name = topic[:20].replace(" ", "_").replace("/", "_")
                        col_d1, col_d2, col_d3 = st.columns(3)
                        with col_d1:
                            st.download_button(
                                "⬇️ Markdown (.md)",
                                data=report_text.encode("utf-8"),
                                file_name=f"report_{safe_name}.md",
                                mime="text/markdown",
                                use_container_width=True,
                            )
                        with col_d2:
                            st.download_button(
                                "⬇️ 纯文本 (.txt)",
                                data=report_text.encode("utf-8"),
                                file_name=f"report_{safe_name}.txt",
                                mime="text/plain",
                                use_container_width=True,
                            )
                        with col_d3:
                            if _DOCX_AVAILABLE:
                                try:
                                    docx_bytes = markdown_to_docx_bytes(report_text, title=f"{report_type_label}：{topic}")
                                    st.download_button(
                                        "⬇️ Word (.docx)",
                                        data=docx_bytes,
                                        file_name=f"report_{safe_name}.docx",
                                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                        use_container_width=True,
                                    )
                                except Exception as ex:
                                    st.warning(f"Word 导出失败: {ex}")
                            else:
                                st.info("安装 python-docx 可导出 Word")

                        if sources:
                            with st.expander(f"📚 参考来源（{len(sources)} 条）"):
                                for i, src in enumerate(sources, 1):
                                    st.markdown(
                                        f"**{i}.** `{src.get('source', '未知')}`\n\n"
                                        f"> {src.get('content_excerpt', '...')[:200]}"
                                    )
                    else:
                        st.error(f"生成失败：{resp.json().get('detail', resp.text)[:120]}")
                except Exception as e:
                    st.error(f"连接失败：{e}")
    else:
        # 类型说明卡片
        cols = st.columns(5)
        for i, (label, _) in enumerate(REPORT_TYPE_OPTIONS.items()):
            desc = ["快速提炼论文核心内容", "整理关键论点与数据",
                    "生成「相关工作」草稿", "横向比较方法与结论", "按需求生成任意格式"][i]
            cols[i].markdown(f"""
            <div class='card' style='text-align:center;min-height:100px'>
                <div style='font-size:1.4rem'>{label.split()[0]}</div>
                <div style='font-weight:600;font-size:0.85rem;margin:4px 0'>{' '.join(label.split()[1:])}</div>
                <div style='color:#94a3b8;font-size:0.78rem'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ==========================================
# 页面 C：系统评估
# ==========================================
elif menu == "🔬 系统评估":
    st.markdown("""
    <div class='page-header'>
        <h1>🔬 系统性能评估</h1>
        <p>5 项量化指标 · 语义相似度 / LLM裁判 / 忠实度 / 相关度 / 耗时百分位</p>
    </div>
    """, unsafe_allow_html=True)

    # 实时健康状态
    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=5).json()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📚 文档块总数", health.get("doc_count", 0))
        c2.metric("💬 活跃会话", health.get("active_sessions", 0))
        c3.metric("🤖 LLM", health.get("llm_provider", "-").upper())
        c4.metric("⚡ 检索引擎", "就绪" if health.get("retriever_ready") else "未就绪",
                  delta="✅" if health.get("retriever_ready") else "⚠️")
    except Exception:
        st.warning("⚠️ 无法连接 API 服务，请确认后端已启动")

    st.divider()

    if not os.path.exists(REPORT_PATH):
        st.markdown("""
        <div class='card' style='text-align:center;padding:40px;border-style:dashed'>
            <div style='font-size:2rem'>📭</div>
            <div style='font-size:1rem;font-weight:600;color:#334155;margin:8px 0'>暂无评估报告</div>
            <div style='color:#94a3b8;font-size:0.88rem'>在终端运行评估脚本后刷新页面</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        try:
            with open(REPORT_PATH, "r", encoding="utf-8") as f:
                report = json.load(f)
            summary = report.get("summary", {})
            details = report.get("details", [])

            # ── 核心指标 ──────────────────────────────────
            acc = summary.get("accuracy", 0)
            sem = summary.get("avg_semantic", 0)
            judge = summary.get("avg_llm_judge")
            faith = summary.get("avg_faithfulness")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🎯 准确率", f"{acc*100:.1f}%",
                      help=f"语义相似度 ≥ {summary.get('correct_threshold',0.75)} 视为正确")
            c2.metric("🧠 平均语义相似度", f"{sem:.3f}",
                      help="Embedding 余弦相似度，0~1")
            c3.metric("⚖️ LLM 裁判分",
                      f"{judge:.3f}" if judge is not None else "N/A",
                      help="0~3 分归一化，N/A 表示未启用")
            c4.metric("🔍 忠实度",
                      f"{faith:.3f}" if faith is not None else "N/A",
                      help="回答有据可查的声明占比")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("📝 测试用例", summary.get("total", 0))
            c6.metric("⏱️ P50 耗时", f"{summary.get('p50_latency_ms',0):.0f} ms")
            c7.metric("⏱️ P90 耗时", f"{summary.get('p90_latency_ms',0):.0f} ms")
            c8.metric("⏱️ P99 耗时", f"{summary.get('p99_latency_ms',0):.0f} ms")

            # ── 分类得分 ──────────────────────────────────
            if summary.get("category_scores"):
                st.divider()
                st.subheader("📂 各类别语义得分")
                cat_data = summary["category_scores"]
                cols = st.columns(min(len(cat_data), 4))
                for i, (cat, score) in enumerate(cat_data.items()):
                    bar = int(score * 10)
                    color = "#22c55e" if score >= 0.75 else "#f59e0b" if score >= 0.6 else "#ef4444"
                    cols[i % 4].markdown(f"""
                    <div class='card' style='text-align:center;padding:12px'>
                        <div style='font-size:0.78rem;color:#64748b'>{cat}</div>
                        <div style='font-size:1.4rem;font-weight:700;color:{color}'>{score:.3f}</div>
                        <div style='background:#e2e8f0;border-radius:4px;height:6px;margin-top:6px'>
                            <div style='background:{color};width:{bar*10}%;height:6px;border-radius:4px'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # ── 详细流水 ──────────────────────────────────
            st.divider()
            st.subheader("🔍 详细评测结果")

            n_correct = sum(1 for d in details if d.get("is_correct"))
            n_wrong   = len(details) - n_correct
            fc1, fc2, fc3 = st.columns(3)
            fc1.markdown(f"<span class='badge-ok'>✅ 正确 {n_correct} 条</span>", unsafe_allow_html=True)
            fc2.markdown(f"<span class='badge-err'>❌ 错误 {n_wrong} 条</span>", unsafe_allow_html=True)
            fc3.markdown(f"<span class='badge-na'>📋 共 {len(details)} 条</span>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            for item in sorted(details, key=lambda x: x.get("semantic_score", 0)):
                ok    = item.get("is_correct", False)
                sem_s = item.get("semantic_score", 0)
                judge_s = item.get("llm_judge_score", -1)
                faith_s = item.get("faithfulness")

                badge = f"<span class='badge-ok'>✅ 正确</span>" if ok else f"<span class='badge-err'>❌ 错误</span>"
                score_parts = [f"语义 {sem_s:.2f}"]
                if judge_s >= 0: score_parts.append(f"LLM {judge_s}/3")
                if faith_s is not None: score_parts.append(f"忠实 {faith_s:.2f}")

                with st.expander(f"{('✅' if ok else '❌')}  {item['question'][:55]}  —  {' | '.join(score_parts)}"):
                    st.markdown(badge, unsafe_allow_html=True)
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**📋 标准答案**")
                        st.info(item.get("ground_truth", "无"))
                    with col_b:
                        st.markdown("**🤖 AI 回答**")
                        ans = item.get("answer", "")
                        if ok:
                            st.success(ans[:400])
                        else:
                            st.error(ans[:400])
                    if item.get("llm_judge_reason"):
                        st.caption(f"💬 {item['llm_judge_reason']}")
                    st.caption(
                        f"⏱️ {item.get('latency_ms',0):.0f}ms  "
                        f"| 📄 {item.get('sources_count',0)} 条来源  "
                        f"| 🏷️ {item.get('category','')}"
                    )

            # 下载
            st.divider()
            st.download_button(
                "⬇️ 下载完整评估报告 (JSON)",
                data=json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="evaluation_report.json",
                mime="application/json",
            )

        except Exception as e:
            st.error(f"读取报告出错: {e}")

    st.divider()
    st.subheader("▶️ 运行评估")
    col_cmd1, col_cmd2 = st.columns(2)
    with col_cmd1:
        st.code("python evaluate.py --key your-secret-key-2026", language="bash")
        st.caption("完整评估（含 LLM 裁判，约 2-5 分钟）")
    with col_cmd2:
        st.code("python evaluate.py --key your-secret-key-2026 --no-llm-judge", language="bash")
        st.caption("快速评估（仅语义相似度，约 30 秒）")
