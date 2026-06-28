from memory_agent.utils.sources import (
    build_source_preview,
    consolidate_sources_for_display,
    polish_assistant_text,
    truncate_preview,
)


def test_truncate_preview():
    assert truncate_preview("hello world", max_len=8) == "hello w…"


def test_build_pdf_preview():
    preview = build_source_preview(
        {"modality": "pdf", "page_start": 1, "page_end": 6, "total_pages": 63},
    )
    assert preview == "Pages 1–6 of 63"


def test_consolidate_preserves_retrieval_order():
    sources = [
        {"source": "a.md", "chunk_index": 2, "relevance_score": 0.9, "citation_index": 1},
        {"source": "a.md", "chunk_index": 0, "relevance_score": 0.5, "citation_index": 2},
        {"source": "b.pdf", "chunk_index": 1, "relevance_score": 0.8, "citation_index": 3},
    ]
    consolidated = consolidate_sources_for_display(sources, max_documents=4, max_chunks_per_document=2)
    assert len(consolidated) == 3
    assert consolidated[0]["citation_index"] == 1
    assert consolidated[2]["source"] == "b.pdf"


def test_format_source_label_pdf():
    from memory_agent.utils.sources import format_source_label

    label = format_source_label({
        "source": "report.pdf",
        "modality": "pdf",
        "page_start": 1,
        "page_end": 6,
    })
    assert label == "report.pdf (pages 1–6)"


def test_polish_assistant_text_removes_inline_sources_and_artifacts():
    raw = "Answer body here.\n\nSource: file1.pdf, file2.md"
    assert polish_assistant_text(raw) == "Answer body here."


def test_polish_trailing_list_artifact():
    assert polish_assistant_text("Some answer about docs.pdf, guide.md']") == (
        "Some answer about docs.pdf, guide.md"
    )
