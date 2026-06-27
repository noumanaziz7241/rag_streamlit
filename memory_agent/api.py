"""Chat API facade over the memory agent and session store."""

from __future__ import annotations

from typing import Any, Dict, Generator, List

from memory_agent.agent.graph import MemoryAgent
from memory_agent.config import DEFAULT_SESSION_ID, DEFAULT_USER_ID
from memory_agent.models import ChatRequest, ChatResponse
from memory_agent.sessions.store import ChatSession, SessionStore


class ChatAPI:
    """API interface for the memory agent."""

    def __init__(self, db_path: str = "chat_memory.db"):
        self.db_path = db_path
        self.agent = MemoryAgent(db_path=db_path)
        self.sessions = SessionStore(db_path=db_path)
        self.sessions.ensure_default_session(DEFAULT_USER_ID, DEFAULT_SESSION_ID)

    def list_sessions(self, user_id: str = DEFAULT_USER_ID) -> List[ChatSession]:
        return self.sessions.list_sessions(user_id)

    def create_session(
        self,
        user_id: str = DEFAULT_USER_ID,
        title: str = "New chat",
    ) -> ChatSession:
        return self.sessions.create_session(user_id=user_id, title=title)

    def delete_session(self, session_id: str, user_id: str = DEFAULT_USER_ID) -> None:
        self.agent.clear_thread(user_id=user_id, session_id=session_id)
        self.sessions.delete_session(session_id)

    def get_history(
        self,
        user_id: str = DEFAULT_USER_ID,
        session_id: str = DEFAULT_SESSION_ID,
    ) -> List[Dict[str, Any]]:
        return self.agent.get_chat_history(user_id=user_id, session_id=session_id)

    def clear_session_history(
        self,
        user_id: str = DEFAULT_USER_ID,
        session_id: str = DEFAULT_SESSION_ID,
    ) -> None:
        self.agent.clear_thread(user_id=user_id, session_id=session_id)

    def ingest_file(self, filename: str, raw_bytes: bytes) -> int:
        return self.agent.ingest_uploaded_file(filename, raw_bytes)

    def _touch_session_after_chat(
        self,
        message: str,
        session_id: str,
    ) -> None:
        session = self.sessions.get_session(session_id)
        if session and session.title == "New chat":
            self.sessions.update_title(session_id, message)
        self.sessions.touch_session(session_id)

    def chat(self, request: ChatRequest) -> ChatResponse:
        try:
            result = self.agent.process_message(
                message=request.message,
                user_id=request.user_id,
                session_id=request.session_id,
            )

            self._touch_session_after_chat(request.message, request.session_id)

            return ChatResponse(
                response=result["response"],
                user_id=result["user_id"],
                session_id=result["session_id"],
                success=True,
                tools_used=result.get("tools_used", []),
                sources=result.get("sources", []),
            )
        except Exception as e:
            return ChatResponse(
                response="",
                user_id=request.user_id,
                session_id=request.session_id,
                success=False,
                error=str(e),
            )

    def chat_stream(
        self,
        message: str,
        user_id: str = DEFAULT_USER_ID,
        session_id: str = DEFAULT_SESSION_ID,
    ) -> Generator[Dict[str, Any], None, None]:
        """Yield streaming chat events for the UI."""
        try:
            for event in self.agent.process_message_stream(
                message=message,
                user_id=user_id,
                session_id=session_id,
            ):
                yield event

            self._touch_session_after_chat(message, session_id)
        except Exception as e:
            yield {"type": "error", "error": str(e)}

    def chat_dict(
        self,
        message: str,
        user_id: str = DEFAULT_USER_ID,
        session_id: str = DEFAULT_SESSION_ID,
    ) -> Dict[str, Any]:
        request = ChatRequest(
            message=message,
            user_id=user_id,
            session_id=session_id,
        )
        response = self.chat(request)

        return {
            "response": response.response,
            "user_id": response.user_id,
            "session_id": response.session_id,
            "success": response.success,
            "error": response.error,
            "tools_used": response.tools_used,
            "sources": response.sources,
        }
