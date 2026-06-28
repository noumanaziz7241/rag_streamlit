from pathlib import Path

from memory_agent.rag.media_store import MediaStore
from memory_agent.rag.multimodal import GeminiMultimodalClient


def test_media_store_saves_relative_path(tmp_path):
    store = MediaStore(base_dir=str(tmp_path))
    stored = store.save("abc123", b"hello", "txt")
    assert stored == "abc123.txt"
    assert (tmp_path / "abc123.txt").read_bytes() == b"hello"
    assert store.exists("abc123.txt")
    assert store.try_load_text("abc123.txt") == "hello"


def test_media_store_legacy_absolute_path(tmp_path):
    store = MediaStore(base_dir=str(tmp_path))
    legacy = tmp_path / "legacy.txt"
    legacy.write_text("legacy content", encoding="utf-8")
    assert store.try_load_text(str(legacy)) == "legacy content"


def test_enrich_metadata_falls_back_to_text_preview_when_file_missing(tmp_path):
    client = GeminiMultimodalClient(media_store=MediaStore(base_dir=str(tmp_path)))
    content = client.enrich_metadata({
        "source": "memory_agent_overview.md",
        "modality": "text",
        "storage_path": "/nonexistent/local/data/uploads/deadbeef.txt",
        "text_preview": "Memory Agent Chat uses LangGraph and Pinecone.",
    })
    assert "LangGraph" in content
    assert "Pinecone" in content


def test_enrich_metadata_uses_local_file_when_present(tmp_path):
    store = MediaStore(base_dir=str(tmp_path))
    store.save("chunk1", b"full chunk body", "txt")
    client = GeminiMultimodalClient(media_store=store)
    content = client.enrich_metadata({
        "source": "doc.md",
        "modality": "text",
        "storage_path": "chunk1.txt",
        "text_preview": "short preview",
    })
    assert content == "full chunk body"
