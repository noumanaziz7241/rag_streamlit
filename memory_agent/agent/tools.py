"""LangGraph agent tools for memory and domain retrieval."""

from __future__ import annotations

import uuid
from typing import List

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from memory_agent.rag.pipeline import retrieve_domain_documents
from memory_agent.vectorstore.manager import VectorStoreManager


class MemoryTools:
    """Tools for memory operations."""

    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vs_manager = vector_store_manager

    @staticmethod
    def get_user_thread(config: RunnableConfig) -> tuple[str, str]:
        """Extract user_id and thread_id from config."""
        cfg = config["configurable"]
        user_id = cfg.get("user_id")
        thread_id = cfg.get("thread_id")
        if user_id is None or thread_id is None:
            raise ValueError("Need both user_id and thread_id in config.")
        return user_id, thread_id

    def create_tools(self):
        """Create tool instances with access to vector store manager."""

        @tool
        def save_memory(memory_text: str, config: RunnableConfig) -> str:
            """Save a memory (e.g. user fact) into memory vector store."""
            user_id, thread_id = self.get_user_thread(config)
            doc = Document(
                page_content=memory_text,
                id=str(uuid.uuid4()),
                metadata={"user_id": user_id, "thread_id": thread_id},
            )
            memory_vs = self.vs_manager.get_memory_vectorstore()
            memory_vs.add_documents([doc])
            return memory_text

        @tool
        def recall_memory(query: str, config: RunnableConfig) -> List[str]:
            """Recall relevant user-specific memories from the memory vector store."""
            user_id, thread_id = self.get_user_thread(config)
            memory_vs = self.vs_manager.get_memory_vectorstore()
            docs = memory_vs.similarity_search(
                query,
                k=5,
                filter={
                    "user_id": {"$eq": user_id},
                    "thread_id": {"$eq": thread_id},
                },
            )
            return [d.page_content for d in docs]

        @tool(response_format="content_and_artifact")
        def retrieve_domain(query: str):
            """Retrieve domain / knowledge documents for a query."""
            serialized, docs = retrieve_domain_documents(
                self.vs_manager.domain_index,
                query=query,
            )
            return serialized, docs

        return [save_memory, recall_memory, retrieve_domain]
