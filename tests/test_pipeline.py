from memory_agent.documents.registry import DocumentRegistry
from memory_agent.rag.pipeline import ingest_file


class FakeIndex:
    def __init__(self):
        self.vectors: dict[str, dict] = {}
        self.deleted: list[str] = []

    def upsert_chunks(self, chunks):
        for chunk in chunks:
            self.vectors[chunk.chunk_id] = chunk
        return len(chunks)

    def delete_by_doc_id(self, doc_id: str):
        self.deleted.append(doc_id)
        self.vectors = {
            key: value
            for key, value in self.vectors.items()
            if value.metadata.get("doc_id") != doc_id
        }


def test_ingest_skips_duplicate_content(temp_db_path: str):
    registry = DocumentRegistry(temp_db_path)
    index = FakeIndex()
    raw = b"Memory Agent Chat overview"

    first_count, first_skipped = ingest_file(index, "notes.md", raw, registry=registry)
    second_count, second_skipped = ingest_file(index, "notes.md", raw, registry=registry)

    assert first_count >= 1
    assert first_skipped is False
    assert second_count == 0
    assert second_skipped is True


def test_ingest_replaces_changed_content(temp_db_path: str):
    registry = DocumentRegistry(temp_db_path)
    index = FakeIndex()

    ingest_file(index, "notes.md", b"version one", registry=registry)
    ingest_file(index, "notes.md", b"version two updated", registry=registry)

    assert len(registry.list_documents()) == 1
    assert index.deleted
