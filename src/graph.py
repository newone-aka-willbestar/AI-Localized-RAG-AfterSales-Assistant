"""
LangGraph 问答工作流。

将原来散落在 api.py 里的过程式路由逻辑，改造为显式有向图：

    START
      │
    [classify]          ← 意图分类
      │
    ┌─┴──────────────┐
    │                │
  chitchat         [load_history]   ← 从 SessionMemory 取历史
    │                │
    │             [rag_ask]         ← HyDE + 检索 + 精排 + LLM 生成
    │                │
    └────┬───────────┘
         │
    [save_memory]       ← 本轮问答写回 SessionMemory
         │
        END

设计原则：
1. 依赖注入（rag, intent_classifier, session_memory 由外部传入），不在模块顶层持有状态
2. 每个节点只返回它修改的 state 字段（partial update），不重写整个 state
3. 节点函数都是纯同步函数，由 api.py 的线程池包裹后调用，不影响异步事件循环
4. build_ask_graph() 在服务启动时调用一次，得到编译好的图对象复用
"""
import logging
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)


# ==========================================
# 状态定义
# ==========================================

class AskState(TypedDict):
    """
    问答工作流的完整状态。
    每个节点读取所需字段，返回需要更新的字段，LangGraph 自动合并。
    """
    # 输入
    question: str
    session_id: Optional[str]

    # 意图分类节点输出
    intent: str
    confidence: float
    method: str          # "rule" | "embedding" | "default" | "fallback"

    # 历史加载节点输出
    history: str

    # 生成节点输出
    answer: str
    sources: list
    provider: str


# ==========================================
# 图构建
# ==========================================

def build_ask_graph(rag, intent_classifier, session_memory):
    """
    构建并编译问答 StateGraph。

    Args:
        rag:               RAG 实例（提供 ask() 和 chitchat()）
        intent_classifier: IntentClassifier 实例
        session_memory:    SessionMemory 实例

    Returns:
        编译好的 CompiledGraph，可直接调用 .invoke(state)
    """

    # ── 节点函数 ───────────────────────────────────────────

    def node_classify(state: AskState) -> dict:
        """
        意图分类节点。
        输入：question
        输出：intent, confidence, method
        """
        result = intent_classifier.classify(state["question"])
        logger.info(
            f"[graph] 意图分类: {result.intent} "
            f"(confidence={result.confidence:.2f}, method={result.method})"
        )
        return {
            "intent": result.intent,
            "confidence": result.confidence,
            "method": result.method,
        }

    def node_load_history(state: AskState) -> dict:
        """
        历史加载节点（仅 RAG 路径经过）。
        从 SessionMemory 取出历史对话文本，供后续生成节点注入 prompt。
        输入：session_id
        输出：history
        """
        session_id = state.get("session_id")
        history = session_memory.get_history(session_id) if session_id else ""
        if history:
            logger.debug(f"[graph] 加载历史记录（session={session_id}）")
        return {"history": history}

    def node_rag_ask(state: AskState) -> dict:
        """
        RAG 生成节点。
        调用完整 RAG 流程：HyDE → 检索 → 精排 → LLM 生成。
        输入：question, history
        输出：answer, sources, provider
        """
        result = rag.ask(
            state["question"],
            history=state.get("history", ""),
        )
        logger.info(f"[graph] RAG 生成完毕，来源数: {len(result.get('sources', []))}")
        return {
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "provider": result.get("provider", ""),
        }

    def node_chitchat(state: AskState) -> dict:
        """
        闲聊生成节点。
        跳过检索，直接 LLM 回复，省检索开销。
        输入：question
        输出：answer, sources（空）, provider
        """
        result = rag.chitchat(state["question"])
        logger.info("[graph] 闲聊路径生成完毕")
        return {
            "answer": result.get("answer", ""),
            "sources": [],
            "provider": result.get("provider", ""),
        }

    def node_save_memory(state: AskState) -> dict:
        """
        记忆保存节点（汇聚节点，两条路径都经过）。
        将本轮问答写回 SessionMemory，无 session_id 时跳过。
        输入：session_id, question, answer, intent
        输出：无（不修改 state）
        """
        session_id = state.get("session_id")
        if session_id and state.get("answer"):
            session_memory.add_turn(
                session_id=session_id,
                question=state["question"],
                answer=state["answer"],
                intent=state.get("intent", ""),
            )
            logger.debug(f"[graph] 本轮问答已写入记忆（session={session_id}）")
        return {}

    # ── 路由函数 ───────────────────────────────────────────

    def route_by_intent(state: AskState) -> str:
        """
        根据意图分类结果决定走哪条路径：
        - chitchat              → 直接生成，跳过 RAG 检索
        - knowledge_query /
          operation / complaint → 先加载历史，再 RAG 检索生成
        """
        return "chitchat" if state["intent"] == "chitchat" else "load_history"

    # ── 图结构 ─────────────────────────────────────────────

    graph = StateGraph(AskState)

    # 注册节点
    graph.add_node("classify",     node_classify)
    graph.add_node("load_history", node_load_history)
    graph.add_node("rag_ask",      node_rag_ask)
    graph.add_node("chitchat",     node_chitchat)
    graph.add_node("save_memory",  node_save_memory)

    # 入口
    graph.set_entry_point("classify")

    # 意图分类后的条件分支
    graph.add_conditional_edges(
        "classify",
        route_by_intent,
        {
            "load_history": "load_history",
            "chitchat":     "chitchat",
        },
    )

    # RAG 路径：历史 → 检索生成 → 存记忆
    graph.add_edge("load_history", "rag_ask")
    graph.add_edge("rag_ask",      "save_memory")

    # 闲聊路径：生成 → 存记忆
    graph.add_edge("chitchat",    "save_memory")

    # 汇聚 → 结束
    graph.add_edge("save_memory", END)

    compiled = graph.compile()
    logger.info("问答 StateGraph 编译完成")
    return compiled


# ==========================================
# 工具函数
# ==========================================

def make_initial_state(question: str, session_id: Optional[str] = None) -> AskState:
    """
    创建带默认值的初始状态，避免调用方手写每个字段。

    用法：
        state = make_initial_state(question, session_id)
        result = ask_graph.invoke(state)
    """
    return AskState(
        question=question,
        session_id=session_id,
        intent="",
        confidence=0.0,
        method="",
        history="",
        answer="",
        sources=[],
        provider="",
    )
