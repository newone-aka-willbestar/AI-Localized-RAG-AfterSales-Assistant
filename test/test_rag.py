import pytest
from rag import RAGChain

@pytest.mark.asyncio
async def test_rag_answer():
    rag = RAGChain()
    result = rag.answer("测试问题，请返回一个答案")
    assert "answer" in result
    assert isinstance(result["answer"], str)
    assert len(result["sources"]) >= 0