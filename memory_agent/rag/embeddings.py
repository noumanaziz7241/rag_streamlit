"""Gemini Embedding 2 client for text and multimodal vectors."""

from __future__ import annotations

from typing import List, Optional

from google.genai import types
from langchain_core.embeddings import Embeddings

from memory_agent.config import (
    EMBEDDING_DIMENSION,
    GEMINI_EMBEDDING_MODEL,
)
from memory_agent.google.genai_client import get_genai_client


class GeminiEmbeddingClient:
    """Wrapper around gemini-embedding-2 for retrieval-oriented embeddings."""

    def __init__(self):
        self.client = get_genai_client()
        self.model = GEMINI_EMBEDDING_MODEL
        self.output_dimensionality = EMBEDDING_DIMENSION

    def _embed_parts(self, parts: List[types.Part]) -> List[float]:
        result = self.client.models.embed_content(
            model=self.model,
            contents=parts,
            config=types.EmbedContentConfig(
                output_dimensionality=self.output_dimensionality,
            ),
        )
        if not result.embeddings:
            raise ValueError("Gemini embedding API returned no vectors.")
        return list(result.embeddings[0].values)

    def embed_document_text(self, text: str, title: str) -> List[float]:
        formatted = f"title: {title} | text: {text}"
        return self._embed_parts([types.Part.from_text(text=formatted)])

    def embed_query(self, query: str) -> List[float]:
        formatted = f"task: search result | query: {query}"
        return self._embed_parts([types.Part.from_text(text=formatted)])

    def embed_media(self, data: bytes, mime_type: str) -> List[float]:
        return self._embed_parts([
            types.Part.from_bytes(data=data, mime_type=mime_type),
        ])


class GeminiTextEmbeddings(Embeddings):
    """LangChain-compatible text embeddings for memory index compatibility."""

    def __init__(self, client: Optional[GeminiEmbeddingClient] = None):
        self._client = client or GeminiEmbeddingClient()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._client.embed_document_text(text, title="document") for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._client.embed_query(text)
