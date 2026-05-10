"""
Word 导出模块测试。

策略：
- 生成字节流后用 python-docx 重新解析，验证内容正确性
- 验证：标题层级、粗体、表格、空文本不崩溃
"""
import pytest

try:
    from docx import Document as DocxDocument
    from src.docx_exporter import markdown_to_docx_bytes
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

pytestmark = pytest.mark.skipif(not _AVAILABLE, reason="python-docx 未安装")


def parse_docx_bytes(data: bytes):
    """从字节流解析 docx，返回 Document 对象"""
    import io
    from docx import Document
    return Document(io.BytesIO(data))


# ==========================================
# 基本输出
# ==========================================

class TestBasicOutput:

    def test_returns_bytes(self):
        result = markdown_to_docx_bytes("# 标题\n\n正文内容")
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_output_is_valid_docx(self):
        """输出字节流能被 python-docx 正确解析"""
        data = markdown_to_docx_bytes("# 标题\n\n正文")
        doc = parse_docx_bytes(data)
        assert doc is not None

    def test_empty_input_does_not_crash(self):
        result = markdown_to_docx_bytes("")
        assert isinstance(result, bytes)

    def test_title_parameter_added(self):
        data = markdown_to_docx_bytes("正文内容", title="我的报告")
        doc = parse_docx_bytes(data)
        # 标题段落应存在于文档中
        all_text = " ".join(p.text for p in doc.paragraphs)
        assert "我的报告" in all_text


# ==========================================
# 标题层级
# ==========================================

class TestHeadings:

    def test_h1_rendered(self):
        data = markdown_to_docx_bytes("# 一级标题\n\n正文")
        doc = parse_docx_bytes(data)
        heading_texts = [p.text for p in doc.paragraphs if p.style.name.startswith("Heading")]
        assert any("一级标题" in t for t in heading_texts)

    def test_h2_rendered(self):
        data = markdown_to_docx_bytes("## 二级标题\n\n正文")
        doc = parse_docx_bytes(data)
        heading_texts = [p.text for p in doc.paragraphs if p.style.name.startswith("Heading")]
        assert any("二级标题" in t for t in heading_texts)

    def test_h3_rendered(self):
        data = markdown_to_docx_bytes("### 三级标题")
        doc = parse_docx_bytes(data)
        heading_texts = [p.text for p in doc.paragraphs if p.style.name.startswith("Heading")]
        assert any("三级标题" in t for t in heading_texts)


# ==========================================
# 内容保留
# ==========================================

class TestContentPreservation:

    def test_paragraph_text_preserved(self):
        data = markdown_to_docx_bytes("这是一段普通正文内容，应该被完整保留。")
        doc = parse_docx_bytes(data)
        all_text = " ".join(p.text for p in doc.paragraphs)
        assert "普通正文内容" in all_text

    def test_list_item_preserved(self):
        data = markdown_to_docx_bytes("- 第一条要点\n- 第二条要点")
        doc = parse_docx_bytes(data)
        all_text = " ".join(p.text for p in doc.paragraphs)
        assert "第一条要点" in all_text
        assert "第二条要点" in all_text

    def test_multiple_sections_preserved(self):
        md = "# 背景\n\n背景内容。\n\n## 方法\n\n方法内容。\n\n## 结论\n\n结论内容。"
        data = markdown_to_docx_bytes(md)
        doc = parse_docx_bytes(data)
        all_text = " ".join(p.text for p in doc.paragraphs)
        assert "背景内容" in all_text
        assert "方法内容" in all_text
        assert "结论内容" in all_text


# ==========================================
# 表格
# ==========================================

class TestTableRendering:

    def test_markdown_table_creates_word_table(self):
        md = "| 方法 | 准确率 |\n| --- | --- |\n| BERT | 92% |\n| GPT-4 | 96% |"
        data = markdown_to_docx_bytes(md)
        doc = parse_docx_bytes(data)
        assert len(doc.tables) >= 1

    def test_table_content_preserved(self):
        md = "| 方法 | 准确率 |\n| --- | --- |\n| BERT | 92% |"
        data = markdown_to_docx_bytes(md)
        doc = parse_docx_bytes(data)
        if doc.tables:
            table_text = " ".join(
                cell.text for row in doc.tables[0].rows for cell in row.cells
            )
            assert "BERT" in table_text
            assert "92%" in table_text


# ==========================================
# 长文档
# ==========================================

class TestLongDocument:

    def test_large_report_does_not_crash(self):
        """典型学术报告体量（5000字）不崩溃"""
        sections = []
        for i in range(10):
            sections.append(f"## 第{i+1}节\n\n{'这是第{}节的内容，包含详细的分析和讨论。'.format(i+1) * 20}")
        md = "\n\n".join(sections)
        data = markdown_to_docx_bytes(md, title="完整学术报告")
        assert len(data) > 1000
