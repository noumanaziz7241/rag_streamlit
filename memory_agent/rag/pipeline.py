"""Document chunking, ingestion, and retrieval utilities for domain RAG."""

from __future__ import annotations

import hashlib
import io
import uuid
from typing import Iterable, List, Sequence, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 8
RETRIEVAL_FETCH_K = 24
MMR_LAMBDA = 0.6

TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
)


def load_uploaded_file(filename: str, raw_bytes: bytes) -> List[Document]:
    """Load an uploaded file into LangChain documents."""
    lower_name = filename.lower()

    if lower_name.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(raw_bytes))
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(
                    Document(
                        page_content=text,
                        metadata={"source": filename, "page": page_number},
                    )
                )
        return pages

    if lower_name.endswith((".txt", ".md", ".markdown", ".csv")):
        text = raw_bytes.decode("utf-8", errors="ignore")
        if not text.strip():
            return []
        return [Document(page_content=text, metadata={"source": filename})]

    raise ValueError(f"Unsupported file type: {filename}")


def chunk_documents(documents: Sequence[Document]) -> List[Document]:
    """Split documents into retrieval-friendly overlapping chunks."""
    chunks = TEXT_SPLITTER.split_documents(documents)
    doc_id = str(uuid.uuid4())

    enriched: List[Document] = []
    for index, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        chunk_id = hashlib.sha256(
            f"{source}:{index}:{chunk.page_content[:120]}".encode("utf-8")
        ).hexdigest()[:16]
        chunk.metadata.update(
            {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "chunk_index": index,
                "source": source,
            }
        )
        enriched.append(chunk)
    return enriched


def ingest_documents(vectorstore, documents: Sequence[Document]) -> int:
    """Chunk and index documents into the domain vector store."""
    chunks = chunk_documents(documents)
    if not chunks:
        return 0
    vectorstore.add_documents(chunks)
    return len(chunks)


def deduplicate_chunks(documents: Iterable[Document]) -> List[Document]:
    """Remove near-duplicate chunks while preserving retrieval order."""
    seen: set[str] = set()
    unique: List[Document] = []
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        chunk_index = doc.metadata.get("chunk_index", doc.page_content[:80])
        key = f"{source}:{chunk_index}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique


def format_retrieved_documents(documents: Sequence[Document]) -> str:
    """Serialize retrieved chunks with source metadata for the LLM."""
    blocks = []
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        page_suffix = f", page {page}" if page is not None else ""
        blocks.append(
            f"Source: {source}{page_suffix}\n"
            f"Chunk: {doc.metadata.get('chunk_index', 'n/a')}\n"
            f"Content: {doc.page_content}"
        )
    return "\n\n---\n\n".join(blocks)


def retrieve_domain_documents(vectorstore, query: str, namespace: str) -> Tuple[str, List[Document]]:
    """
    Retrieve domain knowledge using MMR for diverse, relevant chunks.

    MMR balances relevance and diversity, reducing redundant chunks from the
    same document that simple top-k similarity search often returns.
    """
    docs = vectorstore.max_marginal_relevance_search(
        query=query,
        k=RETRIEVAL_K,
        fetch_k=RETRIEVAL_FETCH_K,
        lambda_mult=MMR_LAMBDA,
        namespace=namespace,
    )
    docs = deduplicate_chunks(docs)
    return format_retrieved_documents(docs), docs
