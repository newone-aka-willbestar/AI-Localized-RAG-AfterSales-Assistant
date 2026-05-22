"""
Gradio 前端入口。

启动：
    python app_gradio.py          # 默认 7860 端口
    gradio app_gradio.py          # 热重载模式（开发用）

环境变量：
    API_BASE_URL  后端地址（默认 http://localhost:8000）
    API_KEY       默认 API Key（可在 UI 里覆盖）
"""
import json
import os
import uuid
from pathlib import Path

import gradio as gr
import requests

API_BASE  = os.environ.get("API_BASE_URL", "http://localhost:8000")
REPORT_PATH = Path("test/evaluation_report.json")

REPORT_TYPES = {
    "📄 文献摘要":  "summary",
    "🔑 要点提炼":  "keypoints",
    "📚 文献综述":  "review",
    "⚖️ 对比分析":  "comparison",
    "✏️ 自定义":    "custom",
}

# ──────────────────────────────────────────────────────────────
# CSS：在 Soft 主题基础上微调
# ──────────────────────────────────────────────────────────────
CSS = """
/* 隐藏 Gradio 底栏 */
footer { display: none !important; }

/* 整体宽度上限 */
.gradio-container { max-width: 1280px !important; margin: auto; }

/* 顶部品牌区 */
.brand-header {
    background: linear-gradient(135deg, #1e40af 0%, #2563eb 50%, #0891b2 100%);
    border-radius: 14px;
    padding: 20px 28px;
    margin-bottom: 8px;
    color: white !important;
}
.brand-header h1 { color: white !important; margin: 0; font-size: 1.75rem; font-weight: 800; }
.brand-header p  { color: #bfdbfe !important; margin: 4px 0 0; font-size: 0.9rem; }

/* 聊天气泡稍微圆一点 */
.message.user   { background: #eff6ff !important; border-radius: 12px !important; }
.message.bot    { background: #f8fafc !important; border-radius: 12px !important; }

/* 参考来源块 */
.source-block {
    background: #f1f5f9;
    border-left: 3px solid #3b82f6;
    border-radius: 0 6px 6px 0;
    padding: 6px 10px;
    margin: 4px 0;
    font-size: 0.85rem;
    color: #475569;
}

/* Tab 标签字体加粗 */
.tab-nav button { font-weight: 600; }

/* 状态标记 */
.status-ok  { color: #16a34a; font-weight: 700; }
.status-err { color: #dc2626; font-weight: 700; }
.status-warn{ color: #d97706; font-weight: 700; }
"""


# ──────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────

def _headers(api_key: str) -> dict:
    return {"x-api-key": api_key}


def get_health(api_key: str) -> str:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        if r.status_code == 200:
            h = r.json()
            ready  = h.get("retriever_ready", False)
            dot    = "🟢" if ready else "🟡"
            status = "检索引擎就绪" if ready else "等待文档上传"
            return (
                f"{dot} **{status}**　　"
                f"📚 {h.get('doc_count', 0)} 块　"
                f"💬 {h.get('active_sessions', 0)} 会话　"
                f"🤖 {h.get('llm_provider', '-').upper()}　"
                f"⚡ HyDE {'ON' if h.get('hyde_enabled') else 'OFF'}"
            )
    except Exception as e:
        pass
    return "🔴 **后端未连接**，请确认 `uvicorn src.api:app --reload --port 8000` 已运行"


def upload_doc(file, api_key: str) -> str:
    if file is None:
        return "⚠️ 请先选择文件"
    try:
        p = Path(file.name)
        with p.open("rb") as fobj:
            r = requests.post(
                f"{API_BASE}/upload",
                files={"file": (p.name, fobj, "application/octet-stream")},
                headers=_headers(api_key),
                timeout=120,
            )
        if r.status_code == 200:
            d = r.json()
            return f"✅ 入库成功 · **{d.get('total_chunks', '?')}** 个文本块 · `{d.get('filename', p.name)}`"
        detail = r.json().get("detail", r.text) if "application/json" in r.headers.get("content-type", "") else r.text
        return f"❌ {detail[:120]}"
    except Exception as e:
        return f"❌ 连接失败：{e}"


def crawl_urls(url_text: str, api_key: str) -> str:
    urls = [u.strip() for u in url_text.strip().splitlines() if u.strip()]
    if not urls:
        return "⚠️ 请输入至少一个 URL"
    try:
        r = requests.post(
            f"{API_BASE}/crawl",
            json={"urls": urls, "force": False, "translate": False},
            headers=_headers(api_key),
            timeout=120,
        )
        if r.status_code == 200:
            d = r.json()
            parts = []
            if d.get("succeeded"): parts.append(f"✅ {len(d['succeeded'])} 个成功")
            if d.get("skipped"):   parts.append(f"⏭️ {len(d['skipped'])} 个已跳过")
            if d.get("failed"):    parts.append(f"❌ {len(d['failed'])} 个失败")
            return "  |  ".join(parts) or "无变化"
        return f"❌ 请求失败 HTTP {r.status_code}"
    except Exception as e:
        return f"❌ 连接失败：{e}"


# ──────────────────────────────────────────────────────────────
# 对话
# ──────────────────────────────────────────────────────────────

def chat_respond(message: str, history: list, api_key: str, session_id: str):
    """
    Gradio generator 模式：先 yield 带占位符的 history，等 API 返回后再逐词流出。
    history 格式：[[user, bot], ...]
    """
    if not message.strip():
        yield history, ""
        return
    if not api_key:
        history = history + [[message, "⚠️ 请先填写顶部的 API Key"]]
        yield history, ""
        return

    # 立即把用户消息追加，bot 先显示省略号
    history = history + [[message, "…"]]
    yield history, ""

    try:
        r = requests.post(
            f"{API_BASE}/ask",
            json={"question": message, "session_id": session_id},
            headers=_headers(api_key),
            timeout=180,
        )
        if r.status_code == 200:
            d       = r.json()
            answer  = d.get("answer", "（无答案）")
            sources = d.get("sources", [])
            intent  = d.get("intent", "")

            # 把来源附在回答末尾（Markdown 格式）
            if sources:
                src_lines = [f"> **{i}.** `{s.get('source','未知')}`  \n> {s.get('content_excerpt','')[:120]}…"
                             for i, s in enumerate(sources, 1)]
                answer += "\n\n---\n**📎 参考来源**\n\n" + "\n\n".join(src_lines)
            if intent:
                answer += f"\n\n<sub>意图：`{intent}`</sub>"

            # 逐词流式输出，增强交互感
            words = answer.split()
            streamed = ""
            for i, w in enumerate(words):
                streamed += w + (" " if i < len(words) - 1 else "")
                history[-1][1] = streamed
                if i % 4 == 0:          # 每 4 个词刷新一次
                    yield history, ""
            history[-1][1] = answer
            yield history, ""

        else:
            detail = r.json().get("detail", r.text) if "application/json" in r.headers.get("content-type", "") else r.text
            history[-1][1] = f"❌ API 错误 {r.status_code}：{detail[:120]}"
            yield history, ""

    except Exception as e:
        history[-1][1] = f"❌ 请求失败：{e}"
        yield history, ""


def clear_chat(session_id: str, api_key: str):
    """清空对话、重置 session_id，并刷新会话标签"""
    try:
        requests.delete(f"{API_BASE}/session/{session_id}", headers=_headers(api_key), timeout=5)
    except Exception:
        pass
    new_sid = str(uuid.uuid4())
    return [], new_sid, f"<sub style='color:#94a3b8'>会话 ID：`{new_sid[:8]}…`</sub>"


# ──────────────────────────────────────────────────────────────
# 报表生成
# ──────────────────────────────────────────────────────────────

def toggle_custom_field(choice: str):
    return gr.update(visible="自定义" in choice)


def generate_report(topic: str, rtype_label: str, custom: str, max_src: int, api_key: str):
    """返回 (markdown_text, download_path_or_None)"""
    if not topic.strip():
        return "⚠️ 请先输入报告主题", None

    rtype = REPORT_TYPES.get(rtype_label, "summary")
    try:
        r = requests.post(
            f"{API_BASE}/report",
            json={
                "topic": topic, "report_type": rtype,
                "custom_instruction": custom, "max_sources": int(max_src),
            },
            headers=_headers(api_key),
            timeout=300,
        )
        if r.status_code == 200:
            d    = r.json()
            text = d.get("report", "")

            # 保存 .md 供下载
            out_dir = Path("test")
            out_dir.mkdir(exist_ok=True)
            safe = topic[:20].replace(" ", "_").replace("/", "_")
            out_path = out_dir / f"report_{safe}.md"
            out_path.write_text(text, encoding="utf-8")

            return text, str(out_path)
        detail = r.json().get("detail", r.text)
        return f"❌ 生成失败：{detail[:150]}", None
    except Exception as e:
        return f"❌ 连接失败：{e}", None


# ──────────────────────────────────────────────────────────────
# 系统评估
# ──────────────────────────────────────────────────────────────

def load_eval_report():
    """读取 evaluation_report.json，返回 (markdown, download_path_or_None)"""
    if not REPORT_PATH.exists():
        return (
            "### 📭 暂无评估报告\n\n"
            "在终端运行以下命令后点击「刷新」：\n\n"
            "```bash\n"
            "python evaluate.py --key YOUR_KEY --no-llm-judge   # 快速模式\n"
            "python evaluate.py --key YOUR_KEY                  # 完整评估\n"
            "python evaluate.py --key YOUR_KEY --ablation       # 消融实验\n"
            "```",
            None,
        )

    try:
        report  = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        summary = report.get("summary", {})
        details = report.get("details", [])

        acc   = summary.get("accuracy", 0)
        sem   = summary.get("avg_semantic", 0)
        judge = summary.get("avg_llm_judge")
        faith = summary.get("avg_faithfulness")
        total = summary.get("total", 0)
        right = summary.get("correct", 0)
        thr   = summary.get("correct_threshold", 0.75)
        date  = summary.get("test_date", "—")

        # ── 汇总卡片 ──
        acc_color = "#16a34a" if acc >= thr else "#dc2626"
        md = f"""## 📊 评估结果 <sub style="font-size:0.8rem;color:#94a3b8">· {date}</sub>

<table>
<tr>
  <td align="center"><b>🎯 准确率</b><br>
    <span style="font-size:2rem;font-weight:800;color:{acc_color}">{acc*100:.1f}%</span><br>
    <sub>{right}/{total} 条通过 · 阈值 ≥ {thr}</sub>
  </td>
  <td align="center"><b>🧠 语义相似度</b><br>
    <span style="font-size:2rem;font-weight:800">{sem:.3f}</span><br>
    <sub>Embedding 余弦均值</sub>
  </td>
  <td align="center"><b>⚖️ LLM 裁判</b><br>
    <span style="font-size:2rem;font-weight:800">{"N/A" if judge is None else f"{judge:.3f}"}</span><br>
    <sub>0~3 分归一化</sub>
  </td>
  <td align="center"><b>🔍 忠实度</b><br>
    <span style="font-size:2rem;font-weight:800">{"N/A" if faith is None else f"{faith:.3f}"}</span><br>
    <sub>幻觉程度（1=无幻觉）</sub>
  </td>
</tr>
</table>

| P50 耗时 | P90 耗时 | P99 耗时 |
|----------|----------|----------|
| {summary.get('p50_latency_ms',0):.0f} ms | {summary.get('p90_latency_ms',0):.0f} ms | {summary.get('p99_latency_ms',0):.0f} ms |
"""

        # ── 分类得分 ──
        if summary.get("category_scores"):
            md += "\n### 📂 分类语义得分\n\n"
            md += "| 类别 | 分数 | 进度 |\n|------|------|------|\n"
            for cat, score in summary["category_scores"].items():
                bar_len = int(score * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                color = "🟢" if score >= 0.75 else "🟡" if score >= 0.6 else "🔴"
                md += f"| {cat} | {color} {score:.3f} | `{bar}` |\n"

        # ── 明细 ──
        if details:
            md += "\n### 🔍 逐条评测\n\n"
            md += "| # | 类别 | 问题 | 语义分 | 判定 | 耗时 |\n"
            md += "|---|------|------|--------|------|------|\n"
            for i, d in enumerate(sorted(details, key=lambda x: -x.get("semantic_score", 0)), 1):
                ok  = "✅" if d.get("is_correct") else "❌"
                q   = d.get("question", "")
                q   = q[:40] + "…" if len(q) > 40 else q
                cat = d.get("category", "")
                sem_s = d.get("semantic_score", 0)
                lat = d.get("latency_ms", 0)
                md += f"| {i} | {cat} | {q} | {sem_s:.3f} | {ok} | {lat:.0f}ms |\n"

        # 保存供下载
        dl_path = REPORT_PATH.with_name("_eval_download.json")
        dl_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return md, str(dl_path)

    except Exception as e:
        return f"❌ 读取报告出错：{e}", None


def load_ablation_report():
    """读取消融实验报告"""
    abl_path = REPORT_PATH.with_name("ablation_report.json")
    if not abl_path.exists():
        return "暂无消融实验报告，运行 `python evaluate.py --ablation --key YOUR_KEY` 生成。"
    try:
        data    = json.loads(abl_path.read_text(encoding="utf-8"))
        results = data.get("results", [])
        if not results:
            return "报告文件为空。"

        baseline_acc = results[0]["summary"].get("accuracy", 0)
        baseline_sem = results[0]["summary"].get("avg_semantic", 0)

        md = f"### 🔬 消融实验对比 <sub>（基准：{results[0]['name']}）</sub>\n\n"
        md += "| 配置 | 准确率 | Δ准确率 | 语义相似 | Δ语义 | 均耗时 | P90 |\n"
        md += "|------|--------|---------|----------|-------|--------|-----|\n"

        for i, item in enumerate(results):
            s    = item["summary"]
            acc  = s.get("accuracy", 0)
            sem  = s.get("avg_semantic", 0)
            lat  = s.get("avg_latency_ms", 0)
            p90  = s.get("p90_latency_ms", 0)
            name = ("★ " if i == 0 else "　") + item["name"]
            d_acc = "—" if i == 0 else f"{(acc - baseline_acc)*100:+.1f}%"
            d_sem = "—" if i == 0 else f"{sem - baseline_sem:+.4f}"
            md += f"| {name} | {acc*100:.1f}% | {d_acc} | {sem:.4f} | {d_sem} | {lat:.0f}ms | {p90:.0f}ms |\n"

        return md
    except Exception as e:
        return f"❌ 读取消融报告出错：{e}"


# ──────────────────────────────────────────────────────────────
# UI 构建
# ──────────────────────────────────────────────────────────────

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="sky",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    button_primary_background_fill="linear-gradient(135deg, *primary_500, *secondary_400)",
    button_primary_background_fill_hover="linear-gradient(135deg, *primary_600, *secondary_500)",
)

with gr.Blocks(theme=theme, title="AI 数字员工", css=CSS) as demo:

    # ── 全局 State ──────────────────────────────────────────────
    session_id = gr.State(str(uuid.uuid4()))

    # ── 品牌 Header ─────────────────────────────────────────────
    gr.HTML("""
    <div class="brand-header">
        <h1>🤖 AI 数字员工</h1>
        <p>基于 RAG 的本地知识库问答 · 学术报表生成 · 多模型容错 · 全自动评估</p>
    </div>
    """)

    # ── 全局工具栏 ──────────────────────────────────────────────
    with gr.Row(equal_height=True):
        api_key_box = gr.Textbox(
            label="🔑 API Key",
            type="password",
            value=os.environ.get("API_KEY", ""),
            placeholder="输入后端 API Key…",
            scale=3, min_width=220,
            container=True,
        )
        health_btn = gr.Button("🔄 服务状态", variant="secondary", scale=1, min_width=110)
        health_md  = gr.Markdown(value="", scale=6)

    health_btn.click(get_health, inputs=api_key_box, outputs=health_md)
    demo.load(get_health, inputs=api_key_box, outputs=health_md)

    # ── Tabs ────────────────────────────────────────────────────
    with gr.Tabs(elem_classes="tab-nav"):

        # ════════════════════════════════════════════
        # Tab 1 · 智能对话
        # ════════════════════════════════════════════
        with gr.Tab("💬 智能对话"):
            with gr.Row(equal_height=False):

                # 左侧：知识库管理面板
                with gr.Column(scale=1, min_width=260):
                    with gr.Accordion("📄 上传文献", open=True):
                        file_input     = gr.File(label="PDF / Word", file_types=[".pdf", ".docx"])
                        upload_btn     = gr.Button("解析并入库 ▶", variant="primary", size="sm")
                        upload_status  = gr.Markdown(label="")

                    with gr.Accordion("🌐 网页抓取", open=False):
                        url_box       = gr.Textbox(
                            label="每行一个 URL",
                            placeholder="https://arxiv.org/abs/xxxx\nhttps://…",
                            lines=4,
                        )
                        crawl_btn     = gr.Button("开始抓取 ▶", variant="secondary", size="sm")
                        crawl_status  = gr.Markdown(label="")

                    with gr.Accordion("ℹ️ 使用说明", open=False):
                        gr.Markdown("""
1. 填写顶部 **API Key**
2. 上传 PDF / Word 或抓取网页入库
3. 在右侧输入框提问，支持多轮追问
4. 绿色状态 = 检索引擎就绪
                        """)

                # 右侧：聊天区域
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(
                        label="对话区",
                        height=460,
                        bubble_full_width=False,
                        show_copy_button=True,
                        render_markdown=True,
                        placeholder=(
                            "<div style='text-align:center;padding:60px;color:#94a3b8'>"
                            "<div style='font-size:3rem'>📚</div>"
                            "<div style='font-size:1.1rem;font-weight:600;margin:8px 0'>知识库已就绪</div>"
                            "<div style='font-size:0.9rem'>先上传文献或抓取网页，再在下方输入问题</div>"
                            "</div>"
                        ),
                    )
                    with gr.Row():
                        msg_box  = gr.Textbox(
                            placeholder="输入问题，支持多轮追问… (Enter 发送，Shift+Enter 换行)",
                            scale=7, show_label=False, container=False,
                            lines=1, max_lines=4,
                        )
                        send_btn = gr.Button("发送 ▶", variant="primary", scale=1, min_width=90)
                    with gr.Row():
                        clear_chat_btn = gr.Button("🗑️ 清空对话", size="sm", variant="secondary")
                        session_lbl    = gr.Markdown("", scale=4)

            # 事件绑定
            upload_btn.click(upload_doc, [file_input, api_key_box], upload_status)
            crawl_btn.click(crawl_urls, [url_box, api_key_box], crawl_status)

            send_btn.click(
                chat_respond,
                inputs=[msg_box, chatbot, api_key_box, session_id],
                outputs=[chatbot, msg_box],
            )
            msg_box.submit(
                chat_respond,
                inputs=[msg_box, chatbot, api_key_box, session_id],
                outputs=[chatbot, msg_box],
            )
            clear_chat_btn.click(
                clear_chat,
                inputs=[session_id, api_key_box],
                outputs=[chatbot, session_id, session_lbl],
            )

        # ════════════════════════════════════════════
        # Tab 2 · 报表生成
        # ════════════════════════════════════════════
        with gr.Tab("📊 报表生成"):
            with gr.Row():

                # 左侧：配置表单
                with gr.Column(scale=2, min_width=280):
                    gr.Markdown("### ⚙️ 报告配置")
                    topic_box = gr.Textbox(
                        label="📌 报告主题",
                        placeholder="例：深度学习在农作物病害识别中的应用进展",
                        lines=2,
                    )
                    rtype_box = gr.Dropdown(
                        label="📋 报告类型",
                        choices=list(REPORT_TYPES.keys()),
                        value="📄 文献摘要",
                    )
                    custom_box = gr.Textbox(
                        label="✏️ 自定义指令",
                        placeholder="例：重点分析局限性，与传统方法对比，中文输出",
                        lines=3,
                        visible=False,
                    )
                    max_src_slider = gr.Slider(
                        minimum=2, maximum=20, step=1, value=8,
                        label="最多引用片段数",
                    )
                    gen_btn = gr.Button("🚀 生成报告", variant="primary", size="lg")

                    gr.Markdown("""
---
**报告类型说明**

| 类型 | 用途 |
|------|------|
| 📄 文献摘要 | 快速提炼论文核心 |
| 🔑 要点提炼 | 整理关键论点数据 |
| 📚 文献综述 | 生成「相关工作」草稿 |
| ⚖️ 对比分析 | 横向对比方法结论 |
| ✏️ 自定义 | 按指令生成任意格式 |
                    """)

                # 右侧：报告输出
                with gr.Column(scale=3):
                    gr.Markdown("### 📝 报告输出")
                    report_output = gr.Markdown(
                        value="*生成报告后在此显示…*",
                        label="",
                        height=520,
                    )
                    report_dl = gr.File(label="⬇️ 下载 Markdown 报告", visible=False)

            rtype_box.change(toggle_custom_field, rtype_box, custom_box)

            def _gen_report_and_update_dl(topic, rtype, custom, max_src, api_key):
                text, path = generate_report(topic, rtype, custom, max_src, api_key)
                dl_update = gr.update(value=path, visible=bool(path))
                return text, dl_update

            gen_btn.click(
                _gen_report_and_update_dl,
                inputs=[topic_box, rtype_box, custom_box, max_src_slider, api_key_box],
                outputs=[report_output, report_dl],
            )

        # ════════════════════════════════════════════
        # Tab 3 · 系统评估
        # ════════════════════════════════════════════
        with gr.Tab("🔬 系统评估"):
            with gr.Row():
                refresh_btn = gr.Button("🔄 刷新报告", variant="secondary")
                gr.Markdown(
                    "运行 `python evaluate.py --key YOUR_KEY` 生成报告后点击刷新",
                    scale=4,
                )
                eval_dl = gr.File(label="⬇️ 下载 JSON", visible=False, scale=1)

            # 主评估报告
            eval_md = gr.Markdown()

            gr.Markdown("---")

            # 消融实验报告（折叠）
            with gr.Accordion("🔬 消融实验对比", open=False):
                gr.Markdown(
                    "运行 `python evaluate.py --ablation --key YOUR_KEY` 生成消融报告"
                )
                ablation_refresh_btn = gr.Button("🔄 刷新消融报告", size="sm")
                ablation_md = gr.Markdown()

            # 事件
            def _refresh_eval():
                md, dl = load_eval_report()
                return md, gr.update(value=dl, visible=bool(dl))

            refresh_btn.click(_refresh_eval, outputs=[eval_md, eval_dl])
            demo.load(_refresh_eval, outputs=[eval_md, eval_dl])
            ablation_refresh_btn.click(load_ablation_report, outputs=ablation_md)
            demo.load(load_ablation_report, outputs=ablation_md)


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_PORT", "7860")),
        share=False,
        show_api=False,
    )
