"""Persistent chat session metadata backed by SQLite."""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ChatSession:
    session_id: str
    user_id: str
    title: str
    created_at: str
    updated_at: str


class SessionStore:
    """Stores chat session metadata separate from LangGraph checkpoints."""

    def __init__(self, db_path: str = "chat_memory.db"):
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chat_sessions_user
                ON chat_sessions (user_id, updated_at DESC)
                """
            )

    def create_session(
        self,
        user_id: str,
        title: str = "New chat",
        session_id: Optional[str] = None,
    ) -> ChatSession:
        session_id = session_id or str(uuid.uuid4())
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_sessions (session_id, user_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, user_id, title, now, now),
            )
        return ChatSession(
            session_id=session_id,
            user_id=user_id,
            title=title,
            created_at=now,
            updated_at=now,
        )

    def list_sessions(self, user_id: str) -> List[ChatSession]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id, user_id, title, created_at, updated_at
                FROM chat_sessions
                WHERE user_id = ?
                ORDER BY updated_at DESC
                """,
                (user_id,),
            ).fetchall()
        return [
            ChatSession(
                session_id=row[0],
                user_id=row[1],
                title=row[2],
                created_at=row[3],
                updated_at=row[4],
            )
            for row in rows
        ]

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT session_id, user_id, title, created_at, updated_at
                FROM chat_sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return ChatSession(
            session_id=row[0],
            user_id=row[1],
            title=row[2],
            created_at=row[3],
            updated_at=row[4],
        )

    def update_title(self, session_id: str, title: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE chat_sessions
                SET title = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (title[:80], _utc_now(), session_id),
            )

    def touch_session(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE chat_sessions
                SET updated_at = ?
                WHERE session_id = ?
                """,
                (_utc_now(), session_id),
            )

    def delete_session(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM chat_sessions WHERE session_id = ?",
                (session_id,),
            )

    def ensure_default_session(self, user_id: str, session_id: str) -> ChatSession:
        existing = self.get_session(session_id)
        if existing:
            return existing
        return self.create_session(user_id=user_id, session_id=session_id, title="Main chat")
