"""LangGraph memory agent with checkpointed conversation threads."""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional

import tiktoken
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from memory_agent.agent.tools import MemoryTools
from memory_agent.config import DEFAULT_SESSION_ID, DEFAULT_USER_ID, get_config_value
from memory_agent.rag.pipeline import ingest_file
from memory_agent.vectorstore.manager import VectorStoreManager


class State(MessagesState):
    """Extended state with recall memories."""

    recall_memories: List[str]


class MemoryAgent:
    """Main agent class handling conversation flow."""

    def __init__(self, db_path: str = "chat_memory.db"):
        self.db_path = db_path
        self.vector_store_manager = VectorStoreManager()
        self.memory_tools = MemoryTools(self.vector_store_manager)
        self.tools = self.memory_tools.create_tools()
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
        self.checkpointer: Optional[SqliteSaver] = None

        self.model = ChatDeepSeek(
            model="deepseek-chat",
            api_key=get_config_value("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
            temperature=0,
        )
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
                " * After using tools, respond using returned data",
            ),
            ("placeholder", "{messages}"),
        ])

        self.graph = self._build_graph(db_path)

    def _agent_fn(self, state: State) -> State:
        """Agent function to process messages."""
        bound = self.prompt | self.model_with_tools
        recall_str = "<recall_memory>\n" + "\n".join(state.get("recall_memories", [])) + "\n</recall_memory>"
        prediction = bound.invoke({
            "messages": state["messages"],
            "recall_memories": recall_str,
        })
        return {"messages": [prediction]}

    def _load_recalls(self, state: State, config: RunnableConfig) -> State:
        """Load relevant memories for current conversation."""
        convo = get_buffer_string(state["messages"])
        convo_trim = self.tokenizer.decode(self.tokenizer.encode(convo)[:2048])
        recs = self.tools[1].invoke(convo_trim, config)
        return {"recall_memories": recs}

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
        builder.add_node("tools", ToolNode(self.tools))
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

    def get_chat_history(
        self,
        user_id: str,
        session_id: str,
    ) -> List[Dict[str, str]]:
        """Load displayable chat history from the LangGraph checkpoint."""
        snapshot = self.graph.get_state(self._thread_config(user_id, session_id))
        messages = snapshot.values.get("messages", [])

        history: List[Dict[str, str]] = []
        for message in messages:
            if isinstance(message, HumanMessage):
                history.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage) and message.content:
                history.append({"role": "assistant", "content": message.content})
        return history

    def clear_thread(self, user_id: str, session_id: str) -> None:
        """Delete LangGraph checkpoint data for a session thread."""
        if self.checkpointer is not None:
            self.checkpointer.delete_thread(session_id)

    def ingest_uploaded_file(self, filename: str, raw_bytes: bytes) -> int:
        """Chunk and index an uploaded file into the domain vector store."""
        return ingest_file(self.vector_store_manager.domain_index, filename, raw_bytes)

    def process_message(
        self,
        message: str,
        user_id: str = DEFAULT_USER_ID,
        session_id: str = DEFAULT_SESSION_ID,
    ) -> Dict[str, Any]:
        """Process a user message, appending to the checkpointed thread history."""
        config = self._thread_config(user_id, session_id)
        state = {"messages": [HumanMessage(content=message)]}

        final_response = None
        for chunk in self.graph.stream(state, config=config):
            for _, updates in chunk.items():
                if "messages" in updates and updates["messages"]:
                    last_msg = updates["messages"][-1]
                    if isinstance(last_msg, AIMessage) and last_msg.content:
                        final_response = last_msg.content

        return {
            "response": final_response or "No response generated",
            "user_id": user_id,
            "session_id": session_id,
        }
