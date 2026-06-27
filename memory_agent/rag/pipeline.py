"""Document ingestion and retrieval utilities for multimodal RAG."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from langchain_core.documents import Document

from memory_agent.rag.loaders import load_uploaded_file
from memory_agent.rag.types import RagChunk
from memory_agent.vectorstore.domain_index import DomainVectorIndex

RETRIEVAL_K = 8
RETRIEVAL_FETCH_K = 24
MMR_LAMBDA = 0.6


def ingest_file(index: DomainVectorIndex, filename: str, raw_bytes: bytes) -> int:
    """Load, embed, and index an uploaded file."""
    chunks = load_uploaded_file(filename, raw_bytes)
    return index.upsert_chunks(chunks)


def ingest_chunks(index: DomainVectorIndex, chunks: Sequence[RagChunk]) -> int:
    return index.upsert_chunks(chunks)


def retrieve_domain_documents(
    index: DomainVectorIndex,
    query: str,
) -> Tuple[str, List[Document]]:
    """Retrieve domain knowledge with MMR over multimodal embeddings."""
    return index.search(
        query=query,
        k=RETRIEVAL_K,
        fetch_k=RETRIEVAL_FETCH_K,
        lambda_mult=MMR_LAMBDA,
    )
