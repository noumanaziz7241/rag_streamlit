from memory_agent.rag.loaders import load_uploaded_file


def test_load_text_markdown():
    content = b"# Title\n\nMemory Agent uses LangGraph and Pinecone."
    chunks = load_uploaded_file("notes.md", content)
    assert len(chunks) >= 1
    assert chunks[0].modality == "text"
    assert chunks[0].source == "notes.md"
    assert "LangGraph" in chunks[0].text


def test_empty_upload_returns_no_chunks():
    assert load_uploaded_file("empty.txt", b"") == []
