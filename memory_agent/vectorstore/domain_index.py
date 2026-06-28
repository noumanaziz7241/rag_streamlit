"""Multimodal domain vector index backed by Pinecone."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

from memory_agent.config import EMBEDDING_DIMENSION, NAMESPACE, get_config_value
from memory_agent.rag.embeddings import GeminiEmbeddingClient
from memory_agent.rag.media_store import MediaStore
from memory_agent.rag.multimodal import GeminiMultimodalClient
from memory_agent.rag.types import RagChunk


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def mmr_select(
    query_vector: List[float],
    candidates: List[Dict[str, Any]],
    k: int,
    lambda_mult: float,
) -> List[Dict[str, Any]]:
    """Maximal Marginal Relevance selection over vector candidates."""
    if not candidates:
        return []

    selected: List[Dict[str, Any]] = []
    remaining = candidates.copy()

    while remaining and len(selected) < k:
        best_score = float("-inf")
        best_item = None

        for item in remaining:
            relevance = item["score"]
            if not selected:
                mmr_score = relevance
            else:
                diversity = max(
                    cosine_similarity(item["vector"], chosen["vector"])
                    for chosen in selected
                )
                mmr_score = lambda_mult * relevance - (1 - lambda_mult) * diversity
            if mmr_score > best_score:
                best_score = mmr_score
                best_item = item

        if best_item is None:
            break
        selected.append(best_item)
        remaining.remove(best_item)

    return selected


class DomainVectorIndex:
    """Indexes and retrieves multimodal chunks with gemini-embedding-2."""

    def __init__(
        self,
        embedding_client: GeminiEmbeddingClient | None = None,
        media_store: MediaStore | None = None,
        multimodal_client: GeminiMultimodalClient | None = None,
    ):
        self.embedding_client = embedding_client or GeminiEmbeddingClient()
        self.media_store = media_store or MediaStore()
        self.multimodal_client = multimodal_client or GeminiMultimodalClient(self.media_store)

        self.pc = Pinecone(api_key=get_config_value("PINECONE_API_KEY"))
        self.index_name = get_config_value("PINECONE_INDEX_NAME")
        self._ensure_index()
        self.index = self.pc.Index(self.index_name)

    def _ensure_index(self) -> None:
        if not self.pc.has_index(self.index_name):
            self.pc.create_index(
                name=self.index_name,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

    def _embed_chunk(self, chunk: RagChunk) -> List[float]:
        if chunk.modality == "text":
            return self.embedding_client.embed_document_text(chunk.text, title=chunk.source)
        if chunk.media_bytes is None:
            raise ValueError(f"Chunk {chunk.chunk_id} is missing media bytes.")
        return self.embedding_client.embed_media(chunk.media_bytes, chunk.mime_type)

    def _build_vector_metadata(self, chunk: RagChunk, storage_path: str | None) -> Dict[str, Any]:
        preview = (chunk.text or "")[:900]
        metadata: Dict[str, Any] = {
            "source": chunk.source,
            "modality": chunk.modality,
            "mime_type": chunk.mime_type,
            "chunk_index": chunk.metadata.get("chunk_index", 0),
            "doc_id": chunk.metadata.get("doc_id", ""),
            "text_preview": preview,
        }
        if storage_path:
            metadata["storage_path"] = storage_path
        for key in ("page_start", "page_end", "total_pages", "note"):
            if key in chunk.metadata:
                metadata[key] = chunk.metadata[key]
        return metadata

    def upsert_chunks(self, chunks: Sequence[RagChunk]) -> int:
        if not chunks:
            return 0

        vectors = []
        for chunk in chunks:
            storage_path = None
            if chunk.has_media:
                extension = chunk.mime_type.split("/")[-1]
                storage_path = self.media_store.save(chunk.chunk_id, chunk.media_bytes, extension)
            # Text chunks: content is stored in Pinecone text_preview — no local file (cloud-safe).

            vectors.append({
                "id": chunk.chunk_id,
                "values": self._embed_chunk(chunk),
                "metadata": self._build_vector_metadata(chunk, storage_path),
            })

        self.index.upsert(vectors=vectors, namespace=NAMESPACE)
        return len(vectors)

    def delete_by_doc_id(self, doc_id: str) -> None:
        """Remove all vectors belonging to a document."""
        self.index.delete(
            filter={"doc_id": {"$eq": doc_id}},
            namespace=NAMESPACE,
        )

    def _mmr_select(
        self,
        query_vector: List[float],
        candidates: List[Dict[str, Any]],
        k: int,
        lambda_mult: float,
    ) -> List[Dict[str, Any]]:
        return mmr_select(query_vector, candidates, k, lambda_mult)

    def search(
        self,
        query: str,
        k: int = 8,
        fetch_k: int = 24,
        lambda_mult: float = 0.6,
        min_score: float = 0.0,
    ) -> Tuple[str, List[Document]]:
        query_vector = self.embedding_client.embed_query(query)
        response = self.index.query(
            vector=query_vector,
            top_k=fetch_k,
            include_metadata=True,
            include_values=True,
            namespace=NAMESPACE,
        )

        matches = getattr(response, "matches", None) or response.get("matches", [])
        candidates = []
        for match in matches:
            metadata = dict(getattr(match, "metadata", None) or match.get("metadata") or {})
            vector = getattr(match, "values", None) or match.get("values") or query_vector
            match_id = getattr(match, "id", None) or match.get("id")
            score = float(getattr(match, "score", None) or match.get("score", 0.0))
            candidates.append({
                "id": match_id,
                "score": score,
                "vector": vector,
                "metadata": metadata,
            })

        selected = self._mmr_select(query_vector, candidates, k=k, lambda_mult=lambda_mult)
        if min_score > 0:
            selected = [item for item in selected if item["score"] >= min_score]
        documents: List[Document] = []
        blocks: List[str] = []

        for citation_index, item in enumerate(selected, start=1):
            metadata = dict(item["metadata"])
            metadata["relevance_score"] = item["score"]
            metadata["citation_index"] = citation_index
            content = self.multimodal_client.enrich_metadata(metadata)
            source = metadata.get("source", "unknown")
            modality = metadata.get("modality", "text")
            blocks.append(
                f"[{citation_index}] Source: {source}\n"
                f"Modality: {modality}\n"
                f"Chunk: {metadata.get('chunk_index', 'n/a')}\n"
                f"Content: {content}"
            )
            documents.append(
                Document(
                    page_content=content,
                    metadata=metadata,
                )
            )

        return "\n\n---\n\n".join(blocks), documents
