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


def test_consolidate_one_chunk_per_document():
    sources = [
        {"source": "a.md", "chunk_index": 2, "relevance_score": 0.9, "preview": "A2"},
        {"source": "a.md", "chunk_index": 0, "relevance_score": 0.5, "preview": "A0"},
        {"source": "b.pdf", "chunk_index": 1, "relevance_score": 0.8, "preview": "B1"},
    ]
    consolidated = consolidate_sources_for_display(sources, max_documents=4)
    assert len(consolidated) == 2
    assert consolidated[0]["source"] == "a.md"
    assert consolidated[0]["preview"] == "A2"


def test_polish_assistant_text_removes_inline_sources_and_artifacts():
    raw = "Answer body here.\n\nSource: file1.pdf, file2.md"
    assert polish_assistant_text(raw) == "Answer body here."


def test_polish_trailing_list_artifact():
    assert polish_assistant_text("Some answer about docs.pdf, guide.md']") == (
        "Some answer about docs.pdf, guide.md"
    )
