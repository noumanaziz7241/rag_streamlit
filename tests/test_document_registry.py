from memory_agent.documents.registry import DocumentRegistry, content_hash


def test_register_and_list_documents(temp_db_path: str):
    registry = DocumentRegistry(temp_db_path)
    file_hash = content_hash(b"hello world")
    registry.register(
        doc_id="doc-1",
        source="hello.txt",
        modality="text",
        chunk_count=2,
        file_hash=file_hash,
    )
    docs = registry.list_documents()
    assert len(docs) == 1
    assert docs[0].source == "hello.txt"
    assert docs[0].chunk_count == 2


def test_duplicate_detection(temp_db_path: str):
    registry = DocumentRegistry(temp_db_path)
    file_hash = content_hash(b"same-content")
    registry.register("doc-1", "file.txt", "text", 1, file_hash)
    assert registry.is_duplicate("file.txt", file_hash) is True
    assert registry.is_duplicate("file.txt", content_hash(b"other")) is False


def test_delete_document(temp_db_path: str):
    registry = DocumentRegistry(temp_db_path)
    registry.register("doc-1", "file.txt", "text", 1, content_hash(b"x"))
    registry.delete("doc-1")
    assert registry.get("doc-1") is None
