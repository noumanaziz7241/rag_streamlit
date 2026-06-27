"""Pinecone vector store management for domain knowledge and memory."""

from __future__ import annotations

import os

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from memory_agent.config import EMBEDDING_DIMENSION, NAMESPACE, get_config_value
from memory_agent.rag.embeddings import GeminiEmbeddingClient, GeminiTextEmbeddings
from memory_agent.vectorstore.domain_index import DomainVectorIndex


class VectorStoreManager:
    """Manages Pinecone vector stores for domain knowledge and memory."""

    def __init__(self):
        os.environ["GOOGLE_API_KEY"] = get_config_value("GEMINI_API_KEY")

        self.embedding_client = GeminiEmbeddingClient()
        self.embeddings = GeminiTextEmbeddings(self.embedding_client)
        self.pc = Pinecone(api_key=get_config_value("PINECONE_API_KEY"))
        self.idx_domain = get_config_value("PINECONE_INDEX_NAME")
        self.idx_memory = get_config_value("PINECONE_MEMORY_INDEX_NAME")
        self.domain_index = DomainVectorIndex(
            embedding_client=self.embedding_client,
        )

    def get_domain_vectorstore(self) -> PineconeVectorStore:
        """Legacy accessor — prefer domain_index for multimodal RAG."""
        domain_index = self.pc.Index(self.idx_domain)
        return PineconeVectorStore(
            index=domain_index,
            embedding=self.embeddings,
            namespace=NAMESPACE,
        )

    def get_memory_vectorstore(self) -> PineconeVectorStore:
        """Get or create memory vector store."""
        if not self.pc.has_index(self.idx_memory):
            self.pc.create_index(
                name=self.idx_memory,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        memory_index = self.pc.Index(self.idx_memory)
        return PineconeVectorStore(
            index=memory_index,
            embedding=self.embeddings,
            namespace=NAMESPACE,
        )
