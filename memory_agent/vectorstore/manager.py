"""Pinecone vector store management for domain knowledge and memory."""

from __future__ import annotations

import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from memory_agent.config import NAMESPACE, get_config_value


class VectorStoreManager:
    """Manages Pinecone vector stores for domain knowledge and memory."""

    def __init__(self):
        os.environ["GOOGLE_API_KEY"] = get_config_value("GEMINI_API_KEY")

        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        self.pc = Pinecone(api_key=get_config_value("PINECONE_API_KEY"))
        self.idx_domain = get_config_value("PINECONE_INDEX_NAME")
        self.idx_memory = get_config_value("PINECONE_MEMORY_INDEX_NAME")

    def get_domain_vectorstore(self) -> PineconeVectorStore:
        """Get domain knowledge vector store."""
        domain_index = self.pc.Index(self.idx_domain)
        return PineconeVectorStore(
            index=domain_index,
            embedding=self.embeddings,
            namespace=NAMESPACE,
        )

    def get_memory_vectorstore(self) -> PineconeVectorStore:
        """Get or create memory vector store."""
        if not self.pc.has_index(self.idx_memory):
            example_embedding = self.embeddings.embed_query("test")
            embedding_dimension = len(example_embedding)

            self.pc.create_index(
                name=self.idx_memory,
                dimension=embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        memory_index = self.pc.Index(self.idx_memory)
        return PineconeVectorStore(
            index=memory_index,
            embedding=self.embeddings,
            namespace=NAMESPACE,
        )
