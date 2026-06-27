"""Persist uploaded media bytes for retrieval-time multimodal understanding."""

from __future__ import annotations

from pathlib import Path

from memory_agent.config import UPLOADS_DIR


class MediaStore:
    """Filesystem store for binary RAG chunks (images, audio, video, PDF)."""

    def __init__(self, base_dir: str = UPLOADS_DIR):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, chunk_id: str, data: bytes, extension: str) -> str:
        """Save bytes and return a relative storage path."""
        suffix = extension if extension.startswith(".") else f".{extension}"
        path = self.base_dir / f"{chunk_id}{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path)

    def load(self, storage_path: str) -> bytes:
        return Path(storage_path).read_bytes()

    def load_text(self, storage_path: str) -> str:
        return Path(storage_path).read_text(encoding="utf-8", errors="ignore")
