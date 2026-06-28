"""LangGraph memory agent with checkpointed conversation threads."""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, Generator, List, Optional

import tiktoken
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage, get_buffer_string
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from memory_agent.agent.tools import MemoryTools
from memory_agent.config import DEFAULT_SESSION_ID, DEFAULT_USER_ID
from memory_agent.documents.registry import DocumentRegistry
from memory_agent.google.chat_model import create_chat_model
from memory_agent.models import SourceCitation
from memory_agent.rag.pipeline import ingest_file
from memory_agent.utils.message_content import extract_text_content
from memory_agent.utils.sources import (
    build_source_preview,
    consolidate_sources_for_display,
)
from memory_agent.vectorstore.manager import VectorStoreManager

TOOL_STATUS_LABELS = {
    "save_memory": "Saving to memory…",
    "recall_memory": "Recalling memories…",
    "retrieve_domain": "Searching knowledge base…",
}


class State(MessagesState):
    """Extended state with recall memories and per-turn UI metadata."""

    recall_memories: List[str]
    ui_tools: List[Dict[str, Any]]
    ui_sources: List[Dict[str, Any]]


class MemoryAgent:
    """Main agent class handling conversation flow."""

    def __init__(self, db_path: str = "chat_memory.db"):
        self.db_path = db_path
        self.vector_store_manager = VectorStoreManager()
        self.memory_tools = MemoryTools(self.vector_store_manager)
        self.tools = self.memory_tools.create_tools()
        self.document_registry = DocumentRegistry(db_path)
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
        self.checkpointer: Optional[SqliteSaver] = None

        self.model = create_chat_model()
        self.model_with_tools = self.model.bind_tools(self.tools)

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an assistant with memory + retrieval. You have tools:\n"
                "- save_memory: store personal/user-specific memory\n"
                "- recall_memory: fetch memories for current user/thread\n"
                "- retrieve_domain: fetch knowledge from your domain database\n\n"
                "Use the full conversation history when responding.\n"
                "Preloaded memories: {recall_memories}\n\n"
                "Guidelines:\n"
                " * Use recall_memory when incorporating personal info\n"
                " * Use retrieve_domain for factual or domain knowledge questions\n"
                " * Ground answers in retrieved documents when available\n"
                " * Cite retrieved documents inline with [1], [2], etc. matching the numbers in the tool output\n"
                " * Do not paste long excerpts, chunk text, or a separate Sources list in your reply\n"
                " * Keep answers concise; the UI shows full reference details separately\n"
                " * After using tools, respond using returned data",
            ),
            ("placeholder", "{messages}"),
        ])

        self.graph = self._build_graph(db_path)

    def _agent_fn(self, state: State) -> State:
        """Agent node — stream-aggregates so LangGraph can forward token events."""
        bound = self.prompt | self.model_with_tools
        recall_str = (
            "<recall_memory>\n"
            + "\n".join(state.get("recall_memories", []))
            + "\n</recall_memory>"
        )

        prediction = None
        for chunk in bound.stream({
            "messages": state["messages"],
            "recall_memories": recall_str,
        }):
            prediction = chunk if prediction is None else prediction + chunk

        if prediction is None:
            prediction = AIMessage(content="")

        if prediction is None:
            prediction = AIMessage(content="")

        clean_content = extract_text_content(prediction.content)
        tool_calls = getattr(prediction, "tool_calls", None)
        kwargs = dict(prediction.additional_kwargs or {})

        if clean_content and not tool_calls:
            kwargs["ui_tools"] = list(state.get("ui_tools", []))
            kwargs["ui_sources"] = self._prepare_sources_for_ui(
                self._dedupe_sources(state.get("ui_sources", []))
            )

        prediction = AIMessage(
            content=clean_content,
            tool_calls=tool_calls or [],
            additional_kwargs=kwargs,
        )

        return {"messages": [prediction]}

    def _load_recalls(self, state: State, config: RunnableConfig) -> State:
        """Load relevant memories and record preload activity for the UI."""
        convo = get_buffer_string(state["messages"])
        convo_trim = self.tokenizer.decode(self.tokenizer.encode(convo)[:2048])
        recs = self.tools[1].invoke(convo_trim, config)

        ui_tools: List[Dict[str, Any]] = []
        if recs:
            ui_tools.append({
                "tool": "recall_memory",
                "summary": f"Preloaded {len(recs)} relevant memories",
            })

        return {"recall_memories": recs, "ui_tools": ui_tools, "ui_sources": []}

    def _tools_fn(self, state: State, config: RunnableConfig) -> Dict[str, Any]:
        """Run tools and accumulate UI metadata for transparency panels."""
        result = ToolNode(self.tools).invoke(state, config)

        ui_tools = list(state.get("ui_tools", []))
        ui_sources = list(state.get("ui_sources", []))

        for tool_msg in result.get("messages", []):
            if not isinstance(tool_msg, ToolMessage):
                continue
            activity = self._parse_tool_message(tool_msg)
            ui_tools.append({
                "tool": activity["tool"],
                "summary": activity["summary"],
                "tool_call_id": tool_msg.tool_call_id,
            })
            ui_sources.extend(activity.get("sources", []))

        return {
            "messages": result["messages"],
            "ui_tools": ui_tools,
            "ui_sources": self._prepare_sources_for_ui(
                self._dedupe_sources(ui_sources)
            ),
        }

    @staticmethod
    def _route_tools(state: State):
        """Route to tools if tool calls exist."""
        msg = state["messages"][-1]
        if getattr(msg, "tool_calls", None):
            return "tools"
        return END

    def _build_graph(self, db_path: str):
        """Build and compile the LangGraph."""
        builder = StateGraph(State)
        builder.add_node("load_recalls", self._load_recalls)
        builder.add_node("agent", self._agent_fn)
        builder.add_node("tools", self._tools_fn)
        builder.add_edge(START, "load_recalls")
        builder.add_edge("load_recalls", "agent")
        builder.add_conditional_edges("agent", self._route_tools, ["tools", END])
        builder.add_edge("tools", "agent")

        conn = sqlite3.connect(db_path, check_same_thread=False)
        self.checkpointer = SqliteSaver(conn)

        return builder.compile(checkpointer=self.checkpointer)

    def _thread_config(self, user_id: str, session_id: str) -> Dict[str, Any]:
        return {
            "configurable": {
                "user_id": user_id,
                "thread_id": session_id,
            }
        }

    @staticmethod
    def _prepare_sources_for_ui(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return consolidate_sources_for_display(sources)

    @staticmethod
    def _dedupe_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        for source in sources:
            key = (str(source.get("source", "")), int(source.get("chunk_index", 0)))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(source)
        return deduped

    @staticmethod
    def _documents_to_sources(docs: List[Document]) -> List[Dict[str, Any]]:
        sources: List[Dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()

        for doc in docs:
            meta = doc.metadata or {}
            source = str(meta.get("source", "unknown"))
            chunk_index = int(meta.get("chunk_index", 0))
            key = (source, chunk_index)
            if key in seen:
                continue
            seen.add(key)

            citation_index = meta.get("citation_index")
            page_start = meta.get("page_start")
            page_end = meta.get("page_end")
            total_pages = meta.get("total_pages")

            citation = SourceCitation(
                source=source,
                modality=str(meta.get("modality", "text")),
                chunk_index=chunk_index,
                preview=build_source_preview(meta, doc.page_content or ""),
                storage_path=meta.get("storage_path"),
                relevance_score=float(meta.get("relevance_score", 0.0)),
                citation_index=int(citation_index) if citation_index is not None else None,
                page_start=int(page_start) if page_start is not None else None,
                page_end=int(page_end) if page_end is not None else None,
                total_pages=int(total_pages) if total_pages is not None else None,
            )
            sources.append(citation.to_dict())

        sources.sort(key=lambda item: int(item.get("citation_index") or 999))
        return sources

    def _parse_tool_message(self, msg: ToolMessage) -> Dict[str, Any]:
        name = msg.name or "unknown"
        if name == "save_memory":
            summary = f"Saved memory: {str(msg.content)[:120]}"
        elif name == "recall_memory":
            if isinstance(msg.content, list):
                count = len(msg.content)
            elif msg.content:
                count = len(str(msg.content).splitlines())
            else:
                count = 0
            summary = f"Recalled {count} memories"
        elif name == "retrieve_domain":
            docs = msg.artifact if getattr(msg, "artifact", None) else []
            sources = self._documents_to_sources(list(docs))
            unique_files = len({s["source"] for s in sources})
            summary = f"Retrieved {unique_files} document(s)"
            return {"tool": name, "summary": summary, "sources": sources}
        else:
            summary = f"Completed {name}"
        return {"tool": name, "summary": summary}

    def _collect_tools_from_messages(
        self,
        messages: List[Any],
        since_index: int = 0,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Fallback: rebuild tool/source metadata from ToolMessages only."""
        tools_used: List[Dict[str, Any]] = []
        sources: List[Dict[str, Any]] = []
        seen_source_keys: set[tuple[str, int]] = set()

        for msg in messages[since_index:]:
            if isinstance(msg, ToolMessage):
                activity = self._parse_tool_message(msg)
                tools_used.append({
                    "tool": activity["tool"],
                    "summary": activity["summary"],
                    "tool_call_id": msg.tool_call_id,
                })
                for source in activity.get("sources", []):
                    key = (source["source"], source["chunk_index"])
                    if key not in seen_source_keys:
                        seen_source_keys.add(key)
                        sources.append(source)

        return tools_used, sources

    def _build_display_history(self, messages: List[Any]) -> List[Dict[str, Any]]:
        history: List[Dict[str, Any]] = []
        turn_start = 0

        for index, msg in enumerate(messages):
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": extract_text_content(msg.content)})
                turn_start = index + 1
                continue

            if isinstance(msg, AIMessage) and msg.content:
                kwargs = msg.additional_kwargs or {}
                tools_used = kwargs.get("ui_tools")
                sources = kwargs.get("ui_sources")

                if tools_used is None or sources is None:
                    tools_used, sources = self._collect_tools_from_messages(messages, turn_start)

                history.append({
                    "role": "assistant",
                    "content": extract_text_content(msg.content),
                    "tools_used": list(tools_used or []),
                    "sources": list(sources or []),
                })

        return history

    def get_chat_history(
        self,
        user_id: str,
        session_id: str,
    ) -> List[Dict[str, Any]]:
        """Load displayable chat history with tool activity and citations."""
        snapshot = self.graph.get_state(self._thread_config(user_id, session_id))
        messages = snapshot.values.get("messages", [])
        return self._build_display_history(messages)

    def clear_thread(self, user_id: str, session_id: str) -> None:
        """Delete LangGraph checkpoint data for a session thread."""
        if self.checkpointer is not None:
            self.checkpointer.delete_thread(session_id)

    def ingest_uploaded_file(self, filename: str, raw_bytes: bytes) -> tuple[int, bool]:
        """Chunk and index an uploaded file into the domain vector store."""
        return ingest_file(
            self.vector_store_manager.domain_index,
            filename,
            raw_bytes,
            registry=self.document_registry,
        )

    def list_documents(self):
        return self.document_registry.list_documents()

    def delete_document(self, doc_id: str) -> bool:
        document = self.document_registry.get(doc_id)
        if document is None:
            return False
        self.vector_store_manager.domain_index.delete_by_doc_id(doc_id)
        self.document_registry.delete(doc_id)
        return True

    @staticmethod
    def _chunk_text(content: Any) -> str:
        return extract_text_content(content)

    def _iter_stream_events(
        self,
        state: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Generator[Dict[str, Any], None, None]:
        """Yield normalized events from LangGraph stream modes.

        Uses ``graph.stream()`` — LangGraph 1.2+ rejects sync ``stream_events(version="v2")``.
        """
        for mode, chunk in self.graph.stream(
            state,
            config=config,
            stream_mode=["updates", "messages"],
        ):
            if mode == "updates":
                for node_name, update in chunk.items():
                    if node_name == "load_recalls":
                        recalls = update.get("recall_memories", [])
                        for tool in update.get("ui_tools", []):
                            yield {"type": "tool_done", **tool}
                        if recalls:
                            yield {
                                "type": "status",
                                "message": f"Preloaded {len(recalls)} relevant memories",
                            }

                    elif node_name == "agent":
                        last_msg = update["messages"][-1]
                        if getattr(last_msg, "tool_calls", None):
                            for tool_call in last_msg.tool_calls:
                                yield {
                                    "type": "tool_start",
                                    "tool_call_id": tool_call.get("id"),
                                    "tool": tool_call.get("name", "unknown"),
                                    "summary": TOOL_STATUS_LABELS.get(
                                        tool_call.get("name", ""),
                                        f"Running {tool_call.get('name', 'tool')}…",
                                    ),
                                    "input": tool_call.get("args", {}),
                                }

                    elif node_name == "tools":
                        for tool_msg in update.get("messages", []):
                            if not isinstance(tool_msg, ToolMessage):
                                continue
                            activity = self._parse_tool_message(tool_msg)
                            yield {
                                "type": "tool_done",
                                "tool_call_id": tool_msg.tool_call_id,
                                "tool": activity["tool"],
                                "summary": activity["summary"],
                            }
                            for source in activity.get("sources", []):
                                yield {"type": "source", "source": source}

            elif mode == "messages":
                msg_chunk, metadata = chunk
                if metadata.get("langgraph_node") != "agent":
                    continue
                if isinstance(msg_chunk, AIMessageChunk):
                    if getattr(msg_chunk, "tool_call_chunks", None):
                        continue
                    text = self._chunk_text(msg_chunk.content)
                    if text:
                        yield {"type": "token", "content": text}

    def process_message_stream(
        self,
        message: str,
        user_id: str = DEFAULT_USER_ID,
        session_id: str = DEFAULT_SESSION_ID,
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream agent events: status, tool activity, tokens, and final metadata."""
        config = self._thread_config(user_id, session_id)
        state = {
            "messages": [HumanMessage(content=message)],
            "ui_tools": [],
            "ui_sources": [],
        }

        yield {"type": "status", "message": "Thinking…"}

        for event in self._iter_stream_events(state, config):
            yield event

        history = self.get_chat_history(user_id, session_id)
        last_assistant = next(
            (item for item in reversed(history) if item["role"] == "assistant"),
            None,
        )

        yield {
            "type": "done",
            "response": last_assistant["content"] if last_assistant else "",
            "tools_used": last_assistant.get("tools_used", []) if last_assistant else [],
            "sources": last_assistant.get("sources", []) if last_assistant else [],
        }

    def process_message(
        self,
        message: str,
        user_id: str = DEFAULT_USER_ID,
        session_id: str = DEFAULT_SESSION_ID,
    ) -> Dict[str, Any]:
        """Process a user message, appending to the checkpointed thread history."""
        final: Dict[str, Any] = {
            "response": "No response generated",
            "user_id": user_id,
            "session_id": session_id,
            "tools_used": [],
            "sources": [],
        }

        for event in self.process_message_stream(message, user_id, session_id):
            if event["type"] == "done":
                final["response"] = event.get("response") or final["response"]
                final["tools_used"] = event.get("tools_used", [])
                final["sources"] = event.get("sources", [])

        return final
