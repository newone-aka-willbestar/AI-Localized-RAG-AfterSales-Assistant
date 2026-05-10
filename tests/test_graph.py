"""
LangGraph 问答工作流测试。

策略：
- 用 Mock 替换 rag / intent_classifier / session_memory，不依赖真实模型
- 验证路由逻辑：chitchat 走闲聊路径，其他意图走 RAG 路径
- 验证节点间状态流转：每个节点的输出字段正确写入 state
- 验证记忆交互：add_turn / get_history 被正确调用
- 验证无 session_id 时的无状态模式
"""
import pytest
from unittest.mock import MagicMock, call

from src.graph import build_ask_graph, make_initial_state, AskState
from src.intent_classifier import IntentResult


# ==========================================
# 测试夹具
# ==========================================

def make_mocks(intent: str = "knowledge_query", answer: str = "测试答案"):
    """
    创建依赖三件套的 Mock。
    intent:  intent_classifier 将返回的意图
    answer:  rag.ask / rag.chitchat 将返回的答案
    """
    mock_rag = MagicMock()
    mock_rag.ask.return_value = {
        "answer": answer,
        "sources": [{"source": "manual.pdf"}],
        "provider": "deepseek",
    }
    mock_rag.chitchat.return_value = {
        "answer": answer,
        "sources": [],
        "provider": "deepseek",
    }

    mock_classifier = MagicMock()
    mock_classifier.classify.return_value = IntentResult(
        intent=intent, confidence=0.95, method="rule"
    )

    mock_memory = MagicMock()
    mock_memory.get_history.return_value = ""

    return mock_rag, mock_classifier, mock_memory


# ==========================================
# 图构建
# ==========================================

class TestBuildGraph:

    def test_graph_compiles_without_error(self):
        rag, clf, mem = make_mocks()
        graph = build_ask_graph(rag, clf, mem)
        assert graph is not None

    def test_graph_has_invoke_method(self):
        rag, clf, mem = make_mocks()
        graph = build_ask_graph(rag, clf, mem)
        assert callable(getattr(graph, "invoke", None))


# ==========================================
# make_initial_state
# ==========================================

class TestMakeInitialState:

    def test_required_fields_present(self):
        state = make_initial_state("问题")
        assert state["question"] == "问题"
        assert state["session_id"] is None
        assert state["intent"] == ""
        assert state["history"] == ""
        assert state["answer"] == ""
        assert state["sources"] == []
        assert state["provider"] == ""

    def test_session_id_set_when_provided(self):
        state = make_initial_state("问题", session_id="abc-123")
        assert state["session_id"] == "abc-123"


# ==========================================
# 路由逻辑
# ==========================================

class TestRouting:

    def test_chitchat_intent_calls_chitchat_not_ask(self):
        """chitchat 意图 → 走闲聊路径，rag.ask() 不被调用"""
        rag, clf, mem = make_mocks(intent="chitchat")
        graph = build_ask_graph(rag, clf, mem)

        graph.invoke(make_initial_state("你好"))

        rag.chitchat.assert_called_once()
        rag.ask.assert_not_called()

    def test_knowledge_query_calls_ask_not_chitchat(self):
        """knowledge_query 意图 → 走 RAG 路径，rag.chitchat() 不被调用"""
        rag, clf, mem = make_mocks(intent="knowledge_query")
        graph = build_ask_graph(rag, clf, mem)

        graph.invoke(make_initial_state("E05错误怎么处理"))

        rag.ask.assert_called_once()
        rag.chitchat.assert_not_called()

    def test_operation_intent_calls_rag_ask(self):
        rag, clf, mem = make_mocks(intent="operation")
        graph = build_ask_graph(rag, clf, mem)
        graph.invoke(make_initial_state("怎么安装设备"))
        rag.ask.assert_called_once()

    def test_complaint_intent_calls_rag_ask(self):
        rag, clf, mem = make_mocks(intent="complaint")
        graph = build_ask_graph(rag, clf, mem)
        graph.invoke(make_initial_state("质量太差了"))
        rag.ask.assert_called_once()


# ==========================================
# 状态流转
# ==========================================

class TestStateTransitions:

    def test_final_state_contains_answer(self):
        rag, clf, mem = make_mocks(intent="knowledge_query", answer="检查电机负载")
        graph = build_ask_graph(rag, clf, mem)
        result = graph.invoke(make_initial_state("E05是什么"))
        assert result["answer"] == "检查电机负载"

    def test_final_state_contains_intent(self):
        rag, clf, mem = make_mocks(intent="operation")
        graph = build_ask_graph(rag, clf, mem)
        result = graph.invoke(make_initial_state("如何安装"))
        assert result["intent"] == "operation"

    def test_final_state_contains_sources_for_rag(self):
        rag, clf, mem = make_mocks(intent="knowledge_query")
        graph = build_ask_graph(rag, clf, mem)
        result = graph.invoke(make_initial_state("保修期多久"))
        assert result["sources"] == [{"source": "manual.pdf"}]

    def test_final_state_sources_empty_for_chitchat(self):
        rag, clf, mem = make_mocks(intent="chitchat")
        graph = build_ask_graph(rag, clf, mem)
        result = graph.invoke(make_initial_state("你好"))
        assert result["sources"] == []

    def test_final_state_contains_provider(self):
        rag, clf, mem = make_mocks(intent="knowledge_query")
        graph = build_ask_graph(rag, clf, mem)
        result = graph.invoke(make_initial_state("问题"))
        assert result["provider"] == "deepseek"

    def test_classify_called_with_question(self):
        rag, clf, mem = make_mocks(intent="knowledge_query")
        graph = build_ask_graph(rag, clf, mem)
        graph.invoke(make_initial_state("具体问题文本"))
        clf.classify.assert_called_once_with("具体问题文本")


# ==========================================
# 会话记忆交互
# ==========================================

class TestMemoryInteraction:

    def test_add_turn_called_after_rag(self):
        """RAG 路径完成后，本轮问答写入记忆"""
        rag, clf, mem = make_mocks(intent="knowledge_query", answer="RAG答案")
        graph = build_ask_graph(rag, clf, mem)
        graph.invoke(make_initial_state("问题", session_id="sid-001"))

        mem.add_turn.assert_called_once_with(
            session_id="sid-001",
            question="问题",
            answer="RAG答案",
            intent="knowledge_query",
        )

    def test_add_turn_called_after_chitchat(self):
        """闲聊路径完成后，本轮问答也写入记忆"""
        rag, clf, mem = make_mocks(intent="chitchat", answer="闲聊答案")
        graph = build_ask_graph(rag, clf, mem)
        graph.invoke(make_initial_state("你好", session_id="sid-002"))

        mem.add_turn.assert_called_once_with(
            session_id="sid-002",
            question="你好",
            answer="闲聊答案",
            intent="chitchat",
        )

    def test_no_session_id_skips_memory(self):
        """不传 session_id → add_turn 不被调用（无状态模式）"""
        rag, clf, mem = make_mocks(intent="knowledge_query")
        graph = build_ask_graph(rag, clf, mem)
        graph.invoke(make_initial_state("问题", session_id=None))

        mem.add_turn.assert_not_called()

    def test_get_history_called_for_rag_path(self):
        """RAG 路径从记忆加载历史"""
        rag, clf, mem = make_mocks(intent="knowledge_query")
        mem.get_history.return_value = "[历史对话]\n用户: 上一个问题\n助手: 上一个答案"
        graph = build_ask_graph(rag, clf, mem)
        graph.invoke(make_initial_state("追问", session_id="sid-003"))

        mem.get_history.assert_called_once_with("sid-003")

    def test_get_history_not_called_for_chitchat(self):
        """闲聊路径不走 load_history 节点，不调用 get_history"""
        rag, clf, mem = make_mocks(intent="chitchat")
        graph = build_ask_graph(rag, clf, mem)
        graph.invoke(make_initial_state("你好", session_id="sid-004"))

        mem.get_history.assert_not_called()

    def test_history_passed_to_rag_ask(self):
        """历史文本从记忆加载后正确传给 rag.ask()"""
        rag, clf, mem = make_mocks(intent="knowledge_query")
        history_text = "[历史对话]\n用户: 上一问\n助手: 上一答"
        mem.get_history.return_value = history_text
        graph = build_ask_graph(rag, clf, mem)
        graph.invoke(make_initial_state("继续问", session_id="sid-005"))

        rag.ask.assert_called_once_with("继续问", history=history_text)

    def test_no_session_id_passes_empty_history_to_ask(self):
        """无 session_id 时，history 为空字符串传给 rag.ask()"""
        rag, clf, mem = make_mocks(intent="knowledge_query")
        graph = build_ask_graph(rag, clf, mem)
        graph.invoke(make_initial_state("问题", session_id=None))

        rag.ask.assert_called_once_with("问题", history="")


# ==========================================
# 边界情况
# ==========================================

class TestEdgeCases:

    def test_empty_question_does_not_crash(self):
        rag, clf, mem = make_mocks(intent="knowledge_query")
        graph = build_ask_graph(rag, clf, mem)
        result = graph.invoke(make_initial_state(""))
        assert "answer" in result

    def test_rag_ask_exception_propagates(self):
        """rag.ask() 抛出异常时，图不吞掉异常（让 api.py 的 try/except 捕获）"""
        rag, clf, mem = make_mocks(intent="knowledge_query")
        rag.ask.side_effect = RuntimeError("模型超时")
        graph = build_ask_graph(rag, clf, mem)

        with pytest.raises(RuntimeError, match="模型超时"):
            graph.invoke(make_initial_state("问题"))

    def test_multiple_invocations_are_independent(self):
        """同一个编译图多次调用互不影响（无共享状态）"""
        rag, clf, mem = make_mocks(intent="knowledge_query")
        rag.ask.side_effect = [
            {"answer": "第一次答案", "sources": [], "provider": "deepseek"},
            {"answer": "第二次答案", "sources": [], "provider": "deepseek"},
        ]
        graph = build_ask_graph(rag, clf, mem)

        r1 = graph.invoke(make_initial_state("问题1", session_id="s1"))
        r2 = graph.invoke(make_initial_state("问题2", session_id="s2"))

        assert r1["answer"] == "第一次答案"
        assert r2["answer"] == "第二次答案"
