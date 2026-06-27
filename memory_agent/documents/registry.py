"""Document registry for indexed knowledge-base files."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def content_hash(raw_bytes: bytes) -> str:
    return hashlib.sha256(raw_bytes).hexdigest()


@dataclass
class IndexedDocument:
    doc_id: str
    source: str
    modality: str
    chunk_count: int
    content_hash: str
    indexed_at: str


class DocumentRegistry:
    """SQLite registry of files indexed into the domain vector store."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS indexed_documents (
                    doc_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    modality TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    indexed_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_indexed_documents_source
                ON indexed_documents (source)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_indexed_documents_hash
                ON indexed_documents (content_hash)
                """
            )

    def list_documents(self) -> List[IndexedDocument]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT doc_id, source, modality, chunk_count, content_hash, indexed_at
                FROM indexed_documents
                ORDER BY indexed_at DESC
                """
            ).fetchall()
        return [
            IndexedDocument(
                doc_id=row[0],
                source=row[1],
                modality=row[2],
                chunk_count=row[3],
                content_hash=row[4],
                indexed_at=row[5],
            )
            for row in rows
        ]

    def get_by_source(self, source: str) -> Optional[IndexedDocument]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT doc_id, source, modality, chunk_count, content_hash, indexed_at
                FROM indexed_documents
                WHERE source = ?
                ORDER BY indexed_at DESC
                LIMIT 1
                """,
                (source,),
            ).fetchone()
        if row is None:
            return None
        return IndexedDocument(
            doc_id=row[0],
            source=row[1],
            modality=row[2],
            chunk_count=row[3],
            content_hash=row[4],
            indexed_at=row[5],
        )

    def is_duplicate(self, source: str, file_hash: str) -> bool:
        existing = self.get_by_source(source)
        return existing is not None and existing.content_hash == file_hash

    def register(
        self,
        doc_id: str,
        source: str,
        modality: str,
        chunk_count: int,
        file_hash: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO indexed_documents
                (doc_id, source, modality, chunk_count, content_hash, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (doc_id, source, modality, chunk_count, file_hash, _utc_now()),
            )

    def delete(self, doc_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM indexed_documents WHERE doc_id = ?",
                (doc_id,),
            )

    def get(self, doc_id: str) -> Optional[IndexedDocument]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT doc_id, source, modality, chunk_count, content_hash, indexed_at
                FROM indexed_documents
                WHERE doc_id = ?
                """,
                (doc_id,),
            ).fetchone()
        if row is None:
            return None
        return IndexedDocument(
            doc_id=row[0],
            source=row[1],
            modality=row[2],
            chunk_count=row[3],
            content_hash=row[4],
            indexed_at=row[5],
        )
